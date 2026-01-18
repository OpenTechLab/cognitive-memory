# cognitive_memory/memory_centers.py
"""
Paměťová centra pro LTM (64D) a STM (16D).

Každé centrum obsahuje:
- K: klíč (64D nebo 16D)
- V: hodnota (d_v)
- h: intenzita/GS
- e: emoční vektor (4D)
- usage: čítač použití (pro pruning)
- age: stáří centra
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math


class MemoryCenters(nn.Module):
    """
    Sada paměťových center s RBF kernel operacemi.
    
    Podporuje:
    - RBF čtení (weighted sum based on distance)
    - RBF zápis (lokální update)
    - Homeostázu (decay)
    - Merge/prune pro správu kapacity
    """
    
    def __init__(
        self,
        n_centers: int,
        d_key: int,
        d_value: int,
        d_emotion: int = 4,
        sigma_read: float = 0.5,
        sigma_write: float = 0.4,
        leak: float = 1e-4,
        leak_emotion: float = 1e-4,
        leak_value: float = 1e-4,
        alpha_value: float = 0.03,
        alpha_emotion: float = 0.01,
        alpha_key: float = 0.0,
        device: str = "cpu"
    ):
        super().__init__()
        self.n_centers = n_centers
        self.d_key = d_key
        self.d_value = d_value
        self.d_emotion = d_emotion
        self.sigma_read = sigma_read
        self.sigma_write = sigma_write
        self.leak = leak
        self.leak_emotion = leak_emotion
        self.leak_value = leak_value
        self.alpha_value = alpha_value
        self.alpha_emotion = alpha_emotion
        self.alpha_key = alpha_key
        
        # Paměťová centra jako buffery (ne parametry - netrénujeme je)
        # Klíče - normalizované na jednotkovou normu
        self.register_buffer("K", F.normalize(torch.randn(n_centers, d_key, device=device), dim=-1))
        # Hodnoty
        self.register_buffer("V", torch.zeros(n_centers, d_value, device=device))
        # Intenzita (GS)
        self.register_buffer("h", torch.zeros(n_centers, device=device))
        # Emoce (neutrální hodnota = 1.0, jako PlantNet hormony)
        self.register_buffer("e", torch.ones(n_centers, d_emotion, device=device))
        # Usage counter (pro pruning)
        self.register_buffer("usage", torch.zeros(n_centers, dtype=torch.long, device=device))
        # Stáří (kroky od vytvoření)
        self.register_buffer("age", torch.zeros(n_centers, dtype=torch.long, device=device))
        # Aktivní maska (pro dynamické přidávání)
        self.register_buffer("active", torch.zeros(n_centers, dtype=torch.bool, device=device))
        
        self.total_step = 0
    
    def homeostasis_step(self):
        """
        Aplikuje decay na všechny aktivní centra.
        
        Emoce decayují směrem k neutrální hodnotě 1.0 (jako PlantNet hormony).
        Intenzita a hodnoty decayují směrem k 0.
        
        Volat jednou za interakci.
        """
        active_mask = self.active
        
        # Decay intenzity (směrem k 0)
        self.h[active_mask] *= (1.0 - self.leak)
        
        # Decay hodnot (směrem k 0)
        self.V[active_mask] *= (1.0 - self.leak_value)
        
        # Decay emocí SMĚREM K NEUTRÁLNÍ HODNOTĚ 1.0
        # e ← e + λ_e * (1.0 - e)  =  (1-λ_e)*e + λ_e*1.0
        # Toto zajistí exponenciální přiblížení k 1.0
        self.e[active_mask] = (
            (1.0 - self.leak_emotion) * self.e[active_mask] + 
            self.leak_emotion * 1.0
        )
        
        # Inkrementuj stáří
        self.age[active_mask] += 1
        
        self.total_step += 1
    
    def compute_rbf_weights(
        self,
        queries: torch.Tensor,
        top_k: int = 32,
        normalize: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vypočítá RBF kernel váhy pro dotazy vůči centrům.
        
        Args:
            queries: [B, T, d_key] normalizované dotazy
            top_k: počet nejbližších center
            normalize: aplikovat softmax normalizaci
            
        Returns:
            weights: [B, T, top_k] RBF váhy
            indices: [B, T, top_k] indexy vybraných center
        """
        B, T, D = queries.shape
        
        # Pouze aktivní centra
        active_indices = torch.where(self.active)[0]
        n_active = active_indices.shape[0]
        
        if n_active == 0:
            # Žádná aktivní centra
            return (
                torch.zeros(B, T, 0, device=queries.device),
                torch.zeros(B, T, 0, dtype=torch.long, device=queries.device)
            )
        
        K_active = self.K[active_indices]  # [n_active, d_key]
        h_active = self.h[active_indices]  # [n_active]
        
        # Cosine similarity (pro normalizované vektory = dot product)
        # [B, T, d_key] @ [d_key, n_active] -> [B, T, n_active]
        similarities = torch.matmul(queries, K_active.T)
        
        # Převod na vzdálenost: ||q - k||^2 = 2 - 2*cos (pro normalizované)
        distances_sq = 2.0 - 2.0 * similarities
        
        # RBF kernel
        rbf_weights = torch.exp(-distances_sq / (2 * self.sigma_read ** 2))
        
        # Top-k výběr
        effective_k = min(top_k, n_active)
        topk_weights, topk_local_indices = torch.topk(rbf_weights, effective_k, dim=-1)
        
        # Převod lokálních indexů na globální
        topk_indices = active_indices[topk_local_indices]
        
        if normalize:
            # Přimíchání intenzity do vah (log-space pro stabilitu)
            h_topk = self.h[topk_indices]  # [B, T, k]
            log_weights = torch.log(topk_weights + 1e-8) + torch.log(h_topk + 1e-8)
            weights = F.softmax(log_weights, dim=-1)
        else:
            weights = topk_weights
        
        return weights, topk_indices
    
    def read(
        self,
        queries: torch.Tensor,
        top_k: int = 32
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Čte z paměti pomocí RBF kernel.
        
        Args:
            queries: [B, T, d_key] normalizované dotazy
            top_k: počet center pro čtení
            
        Returns:
            r_V: [B, T, d_value] čtené hodnoty
            r_E: [B, T, d_emotion] čtené emoce
            weights: [B, T, top_k] váhy (pro debugging/gate)
            indices: [B, T, top_k] indexy center
        """
        B, T, _ = queries.shape
        
        weights, indices = self.compute_rbf_weights(queries, top_k, normalize=True)
        
        if weights.shape[-1] == 0:
            # Žádná aktivní centra
            return (
                torch.zeros(B, T, self.d_value, device=queries.device),
                torch.zeros(B, T, self.d_emotion, device=queries.device),
                weights,
                indices
            )
        
        # Gather hodnoty a emoce pro vybraná centra
        V_selected = self.V[indices]  # [B, T, k, d_value]
        e_selected = self.e[indices]  # [B, T, k, d_emotion]
        
        # Weighted sum
        r_V = torch.einsum('btk,btkv->btv', weights, V_selected)
        r_E = torch.einsum('btk,btke->bte', weights, e_selected)
        
        # Aktualizuj usage counter
        unique_indices = indices.unique()
        self.usage[unique_indices] += 1
        
        return r_V, r_E, weights, indices
    
    def write(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        emotions: torch.Tensor,
        intensities: torch.Tensor,
        top_k: int = 16,
        new_center_threshold: float = 0.3
    ) -> int:
        """
        Zapisuje do paměti pomocí RBF kernel.
        
        Args:
            keys: [N, d_key] klíče segmentů (normalizované)
            values: [N, d_value] hodnoty
            emotions: [N, d_emotion] emoce
            intensities: [N] síla zápisu (ω)
            top_k: počet center pro zápis
            new_center_threshold: práh pro vytvoření nového centra
            
        Returns:
            Počet nově vytvořených center
        """
        N = keys.shape[0]
        new_centers_created = 0
        
        for i in range(N):
            q = keys[i:i+1]  # [1, d_key]
            v = values[i]     # [d_value]
            eps = emotions[i] # [d_emotion]
            omega = intensities[i].item()
            
            if omega < 1e-6:
                continue
            
            # Najdi nejbližší centra
            weights, indices = self.compute_rbf_weights(
                q.unsqueeze(0),  # [1, 1, d_key]
                top_k=top_k,
                normalize=False
            )
            weights = weights.squeeze(0).squeeze(0)  # [k]
            indices = indices.squeeze(0).squeeze(0)  # [k]
            
            if weights.shape[0] == 0 or weights.sum() < new_center_threshold:
                # Nová oblast - vytvoř nové centrum
                new_idx = self._create_new_center(q.squeeze(0), v, eps, omega)
                if new_idx >= 0:
                    new_centers_created += 1
                continue
            
            # Normalizuj váhy lokálně
            weights_normalized = weights / (weights.sum() + 1e-8)
            
            # Update existujících center
            for j, idx in enumerate(indices):
                w = weights_normalized[j].item() * omega
                
                # Update intenzity
                self.h[idx] += w
                
                # Update hodnoty (exponential moving average)
                self.V[idx] += self.alpha_value * w * (v - self.V[idx])
                
                # Update emocí
                self.e[idx] += self.alpha_emotion * w * (eps - self.e[idx])
                
                # Update klíče (velmi opatrně nebo vůbec)
                if self.alpha_key > 0:
                    self.K[idx] = F.normalize(
                        self.K[idx] + self.alpha_key * w * (q.squeeze(0) - self.K[idx]),
                        dim=-1
                    )
        
        return new_centers_created
    
    def _create_new_center(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        emotion: torch.Tensor,
        intensity: float
    ) -> int:
        """
        Vytvoří nové centrum.
        
        Returns:
            Index nového centra, nebo -1 pokud není místo
        """
        # Najdi první neaktivní slot
        inactive_indices = torch.where(~self.active)[0]
        
        if inactive_indices.shape[0] == 0:
            # Není místo - potřeba prune nebo merge
            return -1
        
        idx = inactive_indices[0].item()
        
        self.K[idx] = F.normalize(key, dim=-1)
        self.V[idx] = value
        self.e[idx] = emotion
        self.h[idx] = intensity
        self.usage[idx] = 0
        self.age[idx] = 0
        self.active[idx] = True
        
        return idx
    
    def merge_similar(self, threshold: float = 0.95) -> int:
        """
        Sloučí podobná centra.
        
        Returns:
            Počet sloučených párů
        """
        active_indices = torch.where(self.active)[0]
        n_active = active_indices.shape[0]
        
        if n_active < 2:
            return 0
        
        merged = 0
        
        # Similarity matrix
        K_active = self.K[active_indices]
        sim = torch.matmul(K_active, K_active.T)
        
        # Najdi páry nad prahem (mimo diagonálu)
        sim.fill_diagonal_(0.0)
        
        while True:
            max_sim, flat_idx = sim.max(), sim.argmax()
            if max_sim < threshold:
                break
            
            i = flat_idx // n_active
            j = flat_idx % n_active
            idx_i = active_indices[i]
            idx_j = active_indices[j]
            
            # Slouč j do i
            h_i, h_j = self.h[idx_i], self.h[idx_j]
            total_h = h_i + h_j + 1e-8
            
            self.h[idx_i] = total_h
            self.K[idx_i] = F.normalize(
                (h_i * self.K[idx_i] + h_j * self.K[idx_j]) / total_h,
                dim=-1
            )
            self.V[idx_i] = (h_i * self.V[idx_i] + h_j * self.V[idx_j]) / total_h
            self.e[idx_i] = (h_i * self.e[idx_i] + h_j * self.e[idx_j]) / total_h
            
            # Deaktivuj j
            self.active[idx_j] = False
            
            # Aktualizuj similarity matrix (odstran j)
            sim[j, :] = 0.0
            sim[:, j] = 0.0
            sim[i, :] = 0.0  # Přestaň slučovat i v této iteraci
            sim[:, i] = 0.0
            
            merged += 1
        
        return merged
    
    def prune_weak(
        self,
        intensity_threshold: float = 0.001,
        min_age: int = 1000
    ) -> int:
        """
        Odstraní slabá a stará centra.
        
        Returns:
            Počet odstraněných center
        """
        active_indices = torch.where(self.active)[0]
        
        # Kritéria pro prune
        weak = self.h[active_indices] < intensity_threshold
        old = self.age[active_indices] > min_age
        unused = self.usage[active_indices] < 5
        
        to_prune = active_indices[weak & old & unused]
        
        for idx in to_prune:
            self.active[idx] = False
        
        return to_prune.shape[0]
    
    def apply_normalization(
        self,
        c_v: float = 2.0
    ):
        """
        Aplikuje normalizaci po konsolidaci (logaritmizace / saturace).
        """
        active_mask = self.active
        
        # Logaritmizace intenzity
        self.h[active_mask] = torch.log1p(self.h[active_mask])
        
        # Saturace hodnot
        V_norm = self.V[active_mask].norm(dim=-1, keepdim=True)
        self.V[active_mask] = self.V[active_mask] / (1.0 + V_norm / c_v)
        
        # Tanh na emoce
        self.e[active_mask] = torch.tanh(self.e[active_mask])
    
    def get_n_active(self) -> int:
        """Vrátí počet aktivních center."""
        return self.active.sum().item()
    
    def get_stats(self) -> Dict:
        """Vrátí statistiky."""
        active_mask = self.active
        return {
            "n_active": self.get_n_active(),
            "n_total": self.n_centers,
            "h_mean": self.h[active_mask].mean().item() if active_mask.any() else 0,
            "h_max": self.h[active_mask].max().item() if active_mask.any() else 0,
            "usage_mean": self.usage[active_mask].float().mean().item() if active_mask.any() else 0,
            "age_mean": self.age[active_mask].float().mean().item() if active_mask.any() else 0,
        }
    
    def state_dict_custom(self) -> dict:
        """Vrátí stav pro uložení."""
        return {
            "K": self.K.cpu(),
            "V": self.V.cpu(),
            "h": self.h.cpu(),
            "e": self.e.cpu(),
            "usage": self.usage.cpu(),
            "age": self.age.cpu(),
            "active": self.active.cpu(),
            "n_centers": self.n_centers,
            "d_key": self.d_key,
            "d_value": self.d_value,
            "d_emotion": self.d_emotion,
            "sigma_read": self.sigma_read,
            "sigma_write": self.sigma_write,
            "leak": self.leak,
            "leak_emotion": self.leak_emotion,
            "leak_value": self.leak_value,
            "alpha_value": self.alpha_value,
            "alpha_emotion": self.alpha_emotion,
            "alpha_key": self.alpha_key,
            "total_step": self.total_step,
        }
    
    @classmethod
    def from_state_dict(cls, state: dict, device: str = "cpu") -> 'MemoryCenters':
        """Načte centra ze stavu."""
        centers = cls(
            n_centers=state["n_centers"],
            d_key=state["d_key"],
            d_value=state["d_value"],
            d_emotion=state["d_emotion"],
            sigma_read=state["sigma_read"],
            sigma_write=state["sigma_write"],
            leak=state["leak"],
            leak_emotion=state["leak_emotion"],
            leak_value=state["leak_value"],
            alpha_value=state["alpha_value"],
            alpha_emotion=state["alpha_emotion"],
            alpha_key=state["alpha_key"],
            device=device
        )
        centers.K.copy_(state["K"].to(device))
        centers.V.copy_(state["V"].to(device))
        centers.h.copy_(state["h"].to(device))
        centers.e.copy_(state["e"].to(device))
        centers.usage.copy_(state["usage"].to(device))
        centers.age.copy_(state["age"].to(device))
        centers.active.copy_(state["active"].to(device))
        centers.total_step = state["total_step"]
        return centers

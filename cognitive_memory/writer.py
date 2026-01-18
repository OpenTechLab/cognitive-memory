# cognitive_memory/writer.py
"""
Writer modul pro zápis do Cognitive Memory.

Implementuje:
- Segmentaci hidden states
- Výpočet síly zápisu (novelty, surprise, emotion)
- Zápis do LTM/STM center
- Zápis do 3D terénu
- 3D→64D posilování

Matematika z plánu:
- k_s = norm(pool({X_t})) pro segment s
- q_s^64 = norm(W_q k_s)
- q_s^16 = norm(W_qs k_s)
- ω_s = η_0 * σ(c_n n_s + c_δ δ_s + c_a a_s + b_ω)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List

from .projections import ProjectionBundle
from .terrain_3d import Terrain3D
from .memory_centers import MemoryCenters


class MemoryWriter(nn.Module):
    """
    Zapisovač do kognitivní paměti.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        d_ltm_key: int = 64,
        d_stm_key: int = 16,
        d_value: int = 128,
        d_emotion: int = 4,
        segment_size: int = 32,
        write_strength_base: float = 0.1,
        write_novelty_weight: float = 0.4,
        write_surprise_weight: float = 0.3,
        write_emotion_weight: float = 0.3,
        write_bias: float = -1.0,
        terrain_boost: float = 0.1,
        terrain_gamma: float = 1.0,
        emotion_blend: float = 0.2
    ):
        super().__init__()
        self.segment_size = segment_size
        self.write_strength_base = write_strength_base
        self.write_novelty_weight = write_novelty_weight
        self.write_surprise_weight = write_surprise_weight
        self.write_emotion_weight = write_emotion_weight
        self.write_bias = write_bias
        self.terrain_boost = terrain_boost
        self.terrain_gamma = terrain_gamma
        self.emotion_blend = emotion_blend
        
        # Projekce
        self.proj = ProjectionBundle(
            d_model=d_model,
            d_ltm_key=d_ltm_key,
            d_stm_key=d_stm_key,
            d_value=d_value
        )
        
        # Pro výpočet novosti
        self.novelty_proj = nn.Linear(d_ltm_key, 1)
        nn.init.xavier_uniform_(self.novelty_proj.weight)
        nn.init.zeros_(self.novelty_proj.bias)
    
    def segment_hidden_states(
        self,
        hidden_states: torch.Tensor,
        pool_method: str = "mean"
    ) -> torch.Tensor:
        """
        Rozdělí hidden states na segmenty a pooluje.
        
        Args:
            hidden_states: [B, T, D] nebo [T, D]
            pool_method: "mean" nebo "last"
            
        Returns:
            segments: [B, n_segments, D] nebo [n_segments, D]
        """
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
        
        B, T, D = hidden_states.shape
        n_segments = (T + self.segment_size - 1) // self.segment_size
        
        segments = []
        for i in range(n_segments):
            start = i * self.segment_size
            end = min((i + 1) * self.segment_size, T)
            segment = hidden_states[:, start:end, :]
            
            if pool_method == "mean":
                pooled = segment.mean(dim=1)
            else:  # last
                pooled = segment[:, -1, :]
            
            segments.append(pooled)
        
        return torch.stack(segments, dim=1)  # [B, n_segments, D]
    
    def compute_write_strength(
        self,
        q_ltm: torch.Tensor,
        emotions: torch.Tensor,
        ltm_centers: MemoryCenters,
        surprise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Vypočítá sílu zápisu pro každý segment.
        
        Args:
            q_ltm: [B, N, 64] LTM klíče segmentů
            emotions: [B, N, 4] emoce segmentů
            ltm_centers: pro výpočet novosti
            surprise: [B, N] predikční chyba (volitelné)
            
        Returns:
            omega: [B, N] síla zápisu
        """
        B, N, _ = q_ltm.shape
        
        # Novost: jak daleko od existujících center
        # Použijeme RBF podobnost k nejbližšímu centru
        if ltm_centers.get_n_active() > 0:
            weights, indices = ltm_centers.compute_rbf_weights(
                q_ltm, top_k=1, normalize=False
            )
            max_similarity = weights.squeeze(-1)  # [B, N]
            novelty = 1.0 - max_similarity
        else:
            novelty = torch.ones(B, N, device=q_ltm.device)
        
        # Emoční salience: max odchylka od neutrální hodnoty (1.0)
        emotion_salience = (emotions - 1.0).abs().max(dim=-1).values  # [B, N]
        
        # Překvapení
        if surprise is None:
            surprise = torch.zeros_like(novelty)
        
        # Kombinace
        logit = (
            self.write_novelty_weight * novelty +
            self.write_surprise_weight * surprise +
            self.write_emotion_weight * emotion_salience +
            self.write_bias
        )
        
        omega = self.write_strength_base * torch.sigmoid(logit)
        
        return omega
    
    def write_to_memory(
        self,
        hidden_states: torch.Tensor,
        emotions: torch.Tensor,
        ltm_centers: MemoryCenters,
        stm_centers: MemoryCenters,
        ltm_terrain: Terrain3D,
        stm_terrain: Terrain3D,
        surprise: Optional[torch.Tensor] = None,
        ltm_threshold: float = 0.3,
        stm_threshold: float = 0.3
    ) -> Dict:
        """
        Hlavní funkce pro zápis do paměti.
        
        Args:
            hidden_states: [B, T, D] nebo [T, D] hidden states
            emotions: [4] nebo [B, 4] emoční stav (PlantNet hormony)
            ltm_centers: LTM paměťová centra
            stm_centers: STM paměťová centra
            ltm_terrain: LTM 3D terén
            stm_terrain: STM 3D terén
            surprise: [B, T] predikční chyba per token (volitelné)
            ltm_threshold: Práh pro vytvoření nového LTM centra
            stm_threshold: Práh pro vytvoření nového STM centra
            
        Returns:
            Dict se statistikami zápisu
        """
        # Segmentace
        segments = self.segment_hidden_states(hidden_states)  # [B, N, D]
        B, N, D = segments.shape
        
        # Rozšíření emocí na segmenty
        if emotions.dim() == 1:
            segment_emotions = emotions.unsqueeze(0).unsqueeze(0).expand(B, N, -1)
        else:
            segment_emotions = emotions.unsqueeze(1).expand(B, N, -1)
        
        # Segmentace překvapení
        if surprise is not None:
            surprise = self._segment_values(surprise, N)
        
        # Projekce
        k_pooled = F.normalize(segments, dim=-1)  # Normalizovaná reprezentace
        q_ltm = self.proj.project_to_ltm(segments)  # [B, N, 64]
        q_stm = self.proj.project_to_stm(segments)  # [B, N, 16]
        v = self.proj.project_to_value(segments)    # [B, N, d_value]
        
        # Síla zápisu
        omega = self.compute_write_strength(
            q_ltm, segment_emotions, ltm_centers, surprise
        )  # [B, N]
        
        # 3D→64D posilování (terén ovlivňuje sílu zápisu)
        omega, segment_emotions = self._apply_terrain_boost(
            omega, segment_emotions, q_ltm, ltm_terrain
        )
        
        # Zápis do center
        stats = {
            "n_segments": N,
            "omega_mean": omega.mean().item(),
            "omega_max": omega.max().item(),
        }
        
        # Flatten pro zápis
        for b in range(B):
            # LTM
            new_ltm = ltm_centers.write(
                keys=q_ltm[b],              # [N, 64]
                values=v[b],                # [N, d_value]
                emotions=segment_emotions[b],  # [N, 4]
                intensities=omega[b],       # [N]
                new_center_threshold=ltm_threshold
            )
            stats["new_ltm_centers"] = stats.get("new_ltm_centers", 0) + new_ltm
            
            # STM (s vyšší sílou)
            omega_stm = omega[b] * 3.0  # STM silnější
            new_stm = stm_centers.write(
                keys=q_stm[b],
                values=v[b],
                emotions=segment_emotions[b],
                intensities=omega_stm,
                new_center_threshold=stm_threshold
            )
            stats["new_stm_centers"] = stats.get("new_stm_centers", 0) + new_stm
        
        # Zápis do 3D terénu
        self._write_to_terrain(
            q_ltm, omega, segment_emotions,
            ltm_terrain, self.proj.ltm_to_3d
        )
        self._write_to_terrain(
            q_stm, omega * 3.0, segment_emotions,
            stm_terrain, self.proj.stm_to_3d
        )
        
        return stats
    
    def _segment_values(
        self,
        values: torch.Tensor,
        n_segments: int
    ) -> torch.Tensor:
        """Agreguje hodnoty per token na per segment."""
        B, T = values.shape
        result = []
        for i in range(n_segments):
            start = i * self.segment_size
            end = min((i + 1) * self.segment_size, T)
            segment_val = values[:, start:end].mean(dim=1)
            result.append(segment_val)
        return torch.stack(result, dim=1)  # [B, N]
    
    def _apply_terrain_boost(
        self,
        omega: torch.Tensor,
        emotions: torch.Tensor,
        q_ltm: torch.Tensor,
        terrain: Terrain3D
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        3D→64D posilování: terén ovlivňuje sílu zápisu a emoce.
        
        V místech s historií je systém citlivější.
        """
        B, N, _ = q_ltm.shape
        
        # Projekce do 3D
        z = self.proj.ltm_to_3d(q_ltm)  # [B, N, 3]
        
        # Vzorkování z terénu
        p_H, p_E = terrain.sample(z)  # [B, N], [B, N, 4]
        
        # Boost síly zápisu
        m = 1.0 + self.terrain_boost * torch.tanh(self.terrain_gamma * p_H)
        omega_boosted = omega * m  # [B, N]
        
        # Emoční stabilizace (blend s historickými emocemi)
        emotions_stabilized = (
            (1.0 - self.emotion_blend) * emotions +
            self.emotion_blend * p_E
        )
        
        return omega_boosted, emotions_stabilized
    
    def _write_to_terrain(
        self,
        keys: torch.Tensor,
        intensities: torch.Tensor,
        emotions: torch.Tensor,
        terrain: Terrain3D,
        to_3d_fn
    ):
        """Zapisuje do 3D terénu."""
        B, N, _ = keys.shape
        
        for b in range(B):
            # Projekce do 3D
            z = to_3d_fn(keys[b])  # [N, 3]
            
            terrain.splat(
                positions=z,
                intensities=intensities[b],
                emotions=emotions[b],
                sigma=terrain.alpha_h * 10,  # Proporcionální k difuzi
                eta=0.01
            )


class SegmentBuffer:
    """
    Buffer pro segmenty před zápisem.
    
    Umožňuje dávkový zápis pro efektivitu.
    """
    
    def __init__(self, max_size: int = 256):
        self.max_size = max_size
        self.buffer: List[Dict] = []
    
    def add(
        self,
        hidden_states: torch.Tensor,
        emotions: torch.Tensor,
        surprise: Optional[torch.Tensor] = None
    ):
        """Přidá segment do bufferu."""
        self.buffer.append({
            "hidden_states": hidden_states.detach(),
            "emotions": emotions.detach() if emotions is not None else None,
            "surprise": surprise.detach() if surprise is not None else None,
        })
        
        # Pokud je buffer plný, vrátí True
        return len(self.buffer) >= self.max_size
    
    def flush(self) -> List[Dict]:
        """Vyprázdní buffer a vrátí obsah."""
        content = self.buffer
        self.buffer = []
        return content
    
    def __len__(self):
        return len(self.buffer)

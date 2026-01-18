# cognitive_memory/consolidation.py
"""
Konsolidace STM → LTM (spánek).

Implementuje:
- Detekce únavy (saturace STM)
- Výběr významných STM center
- Přenos do LTM (16D → 64D)
- Přenos STM 3D → LTM 3D (blur)
- Normalizace STM (logaritmizace místo vymazání)

Matematika z plánu:
- F ← (1 - λ_F)F + Σ ω_s^stm
- Když F > Θ → spánek
- Vyber top-M center podle h_i^s
- Mapuj 16D → 64D: q^64 = norm(U @ K^s)
- STM terén → LTM terén: H^3 ← H^3 + ξ_H * blur(H_s^3)
- Normalizace: h^s ← log(1 + h^s)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

from .terrain_3d import Terrain3D
from .memory_centers import MemoryCenters
from .projections import ConsolidationProjection


class SleepConsolidator(nn.Module):
    """
    Konsolidátor pro přenos vzpomínek ze STM do LTM.
    """
    
    def __init__(
        self,
        d_stm_key: int = 16,
        d_ltm_key: int = 64,
        d_value: int = 128,
        d_emotion: int = 4,
        fatigue_leak: float = 0.01,
        fatigue_threshold: float = 10.0,
        consolidation_top_m: int = 128,
        consolidation_kappa: float = 0.05,
        consolidation_xi_h: float = 0.005,
        consolidation_xi_e: float = 0.003,
        normalization_rho_f: float = 0.2,
        normalization_c_v: float = 2.0,
        blur_sigma: float = 2.0
    ):
        super().__init__()
        self.fatigue_leak = fatigue_leak
        self.fatigue_threshold = fatigue_threshold
        self.consolidation_top_m = consolidation_top_m
        self.consolidation_kappa = consolidation_kappa
        self.consolidation_xi_h = consolidation_xi_h
        self.consolidation_xi_e = consolidation_xi_e
        self.normalization_rho_f = normalization_rho_f
        self.normalization_c_v = normalization_c_v
        self.blur_sigma = blur_sigma
        
        # 16D → 64D projekce
        self.stm_to_ltm = ConsolidationProjection(d_stm_key, d_ltm_key)
        
        # Aktuální únava
        self.register_buffer("fatigue", torch.tensor(0.0))
    
    def update_fatigue(self, write_strength: float):
        """
        Aktualizuje úroveň únavy.
        
        Args:
            write_strength: suma síly zápisů od posledního update
        """
        self.fatigue = (1.0 - self.fatigue_leak) * self.fatigue + write_strength
    
    def should_sleep(self) -> bool:
        """Rozhodne, zda je čas na spánek (konsolidaci)."""
        return self.fatigue.item() > self.fatigue_threshold
    
    def consolidate(
        self,
        stm_centers: MemoryCenters,
        ltm_centers: MemoryCenters,
        stm_terrain: Terrain3D,
        ltm_terrain: Terrain3D
    ) -> Dict:
        """
        Provede konsolidaci STM → LTM.
        
        Args:
            stm_centers: STM paměťová centra
            ltm_centers: LTM paměťová centra
            stm_terrain: STM 3D terén
            ltm_terrain: LTM 3D terén
            
        Returns:
            Dict se statistikami konsolidace
        """
        stats = {
            "pre_fatigue": self.fatigue.item(),
            "consolidated_centers": 0,
            "new_ltm_centers": 0,
        }
        
        # ========================================
        # Krok A: Výběr významných STM center
        # ========================================
        active_indices = torch.where(stm_centers.active)[0]
        if active_indices.shape[0] == 0:
            stats["status"] = "no_active_stm_centers"
            return stats
        
        # Seřaď podle intenzity
        h_active = stm_centers.h[active_indices]
        n_to_consolidate = min(self.consolidation_top_m, active_indices.shape[0])
        top_indices = torch.topk(h_active, n_to_consolidate).indices
        selected_indices = active_indices[top_indices]
        
        stats["consolidated_centers"] = selected_indices.shape[0]
        
        # ========================================
        # Krok B: Mapování 16D → 64D a zápis do LTM
        # ========================================
        for idx in selected_indices:
            # Získej STM data
            K_stm = stm_centers.K[idx]  # [16]
            V_stm = stm_centers.V[idx]  # [d_value]
            e_stm = stm_centers.e[idx]  # [4]
            h_stm = stm_centers.h[idx]  # scalar
            
            # Projekce do LTM prostoru
            K_ltm = self.stm_to_ltm(K_stm.unsqueeze(0)).squeeze(0)  # [64]
            
            # Síla zápisu (snížená)
            omega = self.consolidation_kappa * h_stm.item()
            
            # Zápis do LTM
            new_created = ltm_centers.write(
                keys=K_ltm.unsqueeze(0),
                values=V_stm.unsqueeze(0),
                emotions=e_stm.unsqueeze(0),
                intensities=torch.tensor([omega], device=K_ltm.device),
                new_center_threshold=0.5  # Konzervativnější při konsolidaci
            )
            stats["new_ltm_centers"] += new_created
        
        # ========================================
        # Krok C: Přenos STM terén → LTM terén
        # ========================================
        ltm_terrain.merge_from(
            stm_terrain,
            xi_h=self.consolidation_xi_h,
            xi_e=self.consolidation_xi_e,
            blur_sigma=self.blur_sigma
        )
        
        # ========================================
        # Krok D: Normalizace STM (ne vymazání!)
        # ========================================
        stm_centers.apply_normalization(c_v=self.normalization_c_v)
        
        # Reset únavy
        self.fatigue = self.normalization_rho_f * self.fatigue
        
        stats["post_fatigue"] = self.fatigue.item()
        stats["status"] = "success"
        
        return stats
    
    def get_fatigue_level(self) -> float:
        """Vrátí aktuální úroveň únavy (0-1 relativně k prahu)."""
        return min(1.0, self.fatigue.item() / self.fatigue_threshold)


class AutomaticConsolidator:
    """
    Automatický konsolidátor, který sleduje úroveň únavy
    a spouští konsolidaci když je potřeba.
    """
    
    def __init__(
        self,
        consolidator: SleepConsolidator,
        min_interval: int = 100,  # Minimální počet kroků mezi konsolidacemi
    ):
        self.consolidator = consolidator
        self.min_interval = min_interval
        self.steps_since_consolidation = 0
    
    def step(
        self,
        write_strength: float,
        stm_centers: MemoryCenters,
        ltm_centers: MemoryCenters,
        stm_terrain: Terrain3D,
        ltm_terrain: Terrain3D
    ) -> Optional[Dict]:
        """
        Jeden krok - aktualizuje únavu a případně konsoliduje.
        
        Returns:
            Dict se statistikami konsolidace, nebo None
        """
        self.consolidator.update_fatigue(write_strength)
        self.steps_since_consolidation += 1
        
        if (self.consolidator.should_sleep() and 
            self.steps_since_consolidation >= self.min_interval):
            
            stats = self.consolidator.consolidate(
                stm_centers, ltm_centers,
                stm_terrain, ltm_terrain
            )
            self.steps_since_consolidation = 0
            return stats
        
        return None

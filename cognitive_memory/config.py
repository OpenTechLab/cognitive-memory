# cognitive_memory/config.py
"""
Konfigurace pro Cognitive Memory System.

Všechny koeficienty pro roční provoz podle plánu:
- LTM (64D) s poločasem ~1 rok
- STM (16D) s poločasem dny-týdny
- 3D terény s pomalou difuzí
"""

from dataclasses import dataclass, field
from typing import Dict
import math


@dataclass
class MemoryConfig:
    """Hlavní konfigurace paměťového systému."""
    
    # ========================================
    # Dimenze
    # ========================================
    d_model: int = 256          # Dimenze hidden states transformeru
    d_memory_key: int = 64      # Dimenze LTM klíčů
    d_stm_key: int = 16         # Dimenze STM klíčů
    d_memory_value: int = 128   # Dimenze hodnot (d_v)
    d_emotion: int = 4          # Hormony: dopamin, serotonin, kortizol, oxytocin
    
    # 3D terén
    terrain_resolution: int = 48  # Rozlišení 3D gridu (48^3)
    
    # Počet center (počáteční)
    n_ltm_centers: int = 1024   # Počet LTM center
    n_stm_centers: int = 256    # Počet STM center
    
    # ========================================
    # LTM (64D) - poločas ~1 rok
    # ========================================
    # Při 50 interakcích/den, 1 rok = 18250 kroků
    # λ = 1 - 2^(-1/τ) kde τ = 18250
    ltm_leak: float = 3.8e-5          # λ_64 - extrémně pomalý leak
    ltm_leak_emotion: float = 5e-5    # λ_64^E
    ltm_leak_value: float = 3e-5      # λ_64^V
    
    ltm_alpha_value: float = 0.03     # α_V - update rychlost hodnot
    ltm_alpha_emotion: float = 0.01   # α_E - update rychlost emocí
    ltm_alpha_key: float = 0.0        # α_K - klíče obvykle nehýbat
    
    ltm_sigma_read: float = 0.5       # σ pro RBF čtení (0.3-0.7)
    ltm_sigma_write: float = 0.15     # σ pro RBF zápis (AGRESIVNÍ: 0.15)
    
    ltm_top_k_read: int = 32          # Kolik center při čtení
    ltm_top_k_write: int = 16         # Kolik center při zápisu
    
    ltm_new_center_threshold: float = 0.8  # τ_new - >80% shoda nutná, jinak nové (AGRESIVNÍ)
    
    # ========================================
    # LTM 3D terén
    # ========================================
    terrain_ltm_eta: float = 0.005    # η_3 - síla zápisu do terénu
    terrain_ltm_lambda: float = 5e-5  # λ_3 - leak (homeostáza)
    terrain_ltm_alpha_h: float = 0.002  # α_H - difuze intenzity
    terrain_ltm_alpha_e: float = 0.001  # α_E - difuze emocí
    terrain_ltm_sigma: float = 0.1    # σ_3 - šířka splat kernelu
    
    # ========================================
    # STM (16D) - poločas dny až týdny
    # ========================================
    stm_leak: float = 5e-3            # λ_stm - rychlejší leak
    stm_leak_emotion: float = 7e-3
    stm_leak_value: float = 4e-3
    
    stm_alpha_value: float = 0.1      # α_V^s - rychlejší update
    stm_alpha_emotion: float = 0.08   # α_E^s
    
    stm_sigma_read: float = 0.4
    stm_sigma_write: float = 0.2      # AGRESIVNÍ: 0.2
    
    stm_top_k_read: int = 16
    stm_top_k_write: int = 8
    
    # ========================================
    # STM 3D terén (rychlejší)
    # ========================================
    terrain_stm_eta: float = 0.02
    terrain_stm_lambda: float = 1e-3
    terrain_stm_alpha_h: float = 0.02
    terrain_stm_alpha_e: float = 0.01
    
    # ========================================
    # TerrainPrior
    # ========================================
    terrain_prior_beta: float = 0.02  # β_q - posun dotazu (malé!)
    terrain_prior_gate_bias: float = -3.0  # b_g - defaultně zavřeno
    
    # ========================================
    # Memory Attention & Gate
    # ========================================
    gate_bias: float = -2.5           # Defaultně paměť nevstupuje
    gate_prior_weight: float = 0.5    # u - váha terrain prior v gate
    
    # Které vrstvy transformeru používají paměť (poslední 30%)
    memory_layer_start_ratio: float = 0.7
    
    # ========================================
    # Zápis (Writer)
    # ========================================
    segment_size: int = 32            # Tokenů na segment (pro pooling)
    
    write_strength_base: float = 0.2  # η_0 - základní síla zápisu (AGRESIVNÍ: 0.2)
    write_novelty_weight: float = 0.6  # c_n (AGRESIVNÍ: 0.6)
    write_surprise_weight: float = 0.3  # c_δ
    write_emotion_weight: float = 0.3   # c_a
    write_bias: float = -0.2           # b_ω (AGRESIVNÍ: -0.2)
    
    # 3D→64D posilování při zápisu
    write_terrain_boost: float = 0.1   # ρ
    write_terrain_gamma: float = 1.0   # γ_H
    write_emotion_blend: float = 0.2   # ρ_E
    
    # ========================================
    # Únava & Konsolidace (Sleep)
    # ========================================
    fatigue_leak: float = 0.01        # λ_F
    fatigue_threshold: float = 5.0    # Θ - práh pro spánek (AGRESIVNÍ: 5.0)
    
    consolidation_top_m: int = 128    # M - kolik STM center konsolidovat
    consolidation_kappa: float = 0.8  # κ - přepočet intenzity (AGRESIVNÍ: 0.8)
    consolidation_xi_h: float = 0.005  # ξ_H - STM3D→LTM3D
    consolidation_xi_e: float = 0.003  # ξ_E
    
    normalization_rho_f: float = 0.2  # ρ_F - reset únavy po spánku
    normalization_c_v: float = 2.0    # c_V - saturace hodnot
    
    # ========================================
    # Merge/Prune (správa kapacity)
    # ========================================
    merge_similarity_threshold: float = 0.95  # τ_merge
    prune_intensity_threshold: float = 0.001  # τ_h
    prune_min_age: int = 1000         # Minimální stáří před prune
    max_centers_ltm: int = 4096       # Maximální počet LTM center
    max_centers_stm: int = 512        # Maximální počet STM center
    
    # ========================================
    # Persistence
    # ========================================
    state_file: str = "cognitive_memory_state.pt"
    
    def get_ltm_layer_indices(self, n_layers: int):
        """Vrací indexy vrstev, které používají paměť."""
        start = int(self.memory_layer_start_ratio * n_layers)
        return list(range(start, n_layers))


def compute_leak_from_halflife(halflife_steps: int) -> float:
    """Vypočítá leak koeficient z poločasu v krocích."""
    return 1.0 - math.pow(2.0, -1.0 / halflife_steps)


# Výchozí instance
DEFAULT_CONFIG = MemoryConfig()

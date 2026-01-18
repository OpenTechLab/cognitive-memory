# cognitive_memory/memory_block.py
"""
MemoryBlock - integrační blok pro transformer.

Implementuje paměťovou vrstvu mezi Self-Attention a MLP:

    Y = X + SA(LN(X))
    M = MemAttn(LN(Y))    # TerrainPrior + RBF read
    Y' = Y + g ⊙ W_m M
    X_out = Y' + MLP(LN(Y'))

Emoce se aplikují jako FiLM modulace: α(E)⊙u + β(E)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict

from .projections import ProjectionBundle
from .terrain_3d import Terrain3D
from .memory_centers import MemoryCenters
from .terrain_prior import DualTerrainPrior
from .memory_attention import DualMemoryAttention


class EmotionFiLM(nn.Module):
    """
    Feature-wise Linear Modulation based on emotions.
    
    Aplikuje: α(E) ⊙ x + β(E)
    """
    
    def __init__(
        self,
        d_emotion: int = 4,
        d_model: int = 256
    ):
        super().__init__()
        
        # γ (scale) a β (shift) z emocí
        self.gamma_proj = nn.Linear(d_emotion, d_model)
        self.beta_proj = nn.Linear(d_emotion, d_model)
        
        # Inicializace pro identity start
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.ones_(self.gamma_proj.bias)  # γ = 1 default
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)  # β = 0 default
    
    def forward(
        self,
        x: torch.Tensor,
        emotions: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
            emotions: [B, T, d_emotion]
            
        Returns:
            modulated: [B, T, d_model]
        """
        gamma = self.gamma_proj(emotions)  # [B, T, d_model]
        beta = self.beta_proj(emotions)    # [B, T, d_model]
        
        return gamma * x + beta


class MemoryBlock(nn.Module):
    """
    Paměťový blok pro integraci do TransformerBlock.
    
    Vkládá se mezi Self-Attention a MLP.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        d_ltm_key: int = 64,
        d_stm_key: int = 16,
        d_value: int = 128,
        d_emotion: int = 4,
        gate_bias: float = -2.5,
        gate_prior_weight: float = 0.5,
        terrain_prior_beta: float = 0.02,
        terrain_prior_gate_bias: float = -3.0,
        ltm_top_k: int = 32,
        stm_top_k: int = 16,
        use_emotion_film: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.use_emotion_film = use_emotion_film
        
        # Projekce
        self.proj = ProjectionBundle(
            d_model=d_model,
            d_ltm_key=d_ltm_key,
            d_stm_key=d_stm_key,
            d_value=d_value
        )
        
        # Terrain Prior
        self.terrain_prior = DualTerrainPrior(
            d_ltm_key=d_ltm_key,
            d_stm_key=d_stm_key,
            d_emotion=d_emotion,
            beta_ltm=terrain_prior_beta,
            beta_stm=terrain_prior_beta * 2,  # STM silnější
            gate_bias=terrain_prior_gate_bias
        )
        
        # Memory Attention
        self.memory_attention = DualMemoryAttention(
            d_model=d_model,
            d_ltm_key=d_ltm_key,
            d_stm_key=d_stm_key,
            d_value=d_value,
            d_emotion=d_emotion,
            gate_bias=gate_bias,
            gate_prior_weight=gate_prior_weight,
            ltm_top_k=ltm_top_k,
            stm_top_k=stm_top_k
        )
        
        # Emotion FiLM (pro MLP)
        if use_emotion_film:
            self.emotion_film = EmotionFiLM(d_emotion, d_model)
        
        # LayerNorm pro vstup
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        terrain_ltm: Terrain3D,
        terrain_stm: Terrain3D,
        ltm_centers: MemoryCenters,
        stm_centers: MemoryCenters,
        return_gate: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass paměťového bloku.
        
        Implementuje:
            M = MemAttn(LN(Y))
            Y' = Y + g ⊙ W_m M
        
        Args:
            x: [B, T, d_model] výstup z Self-Attention (před MLP)
            terrain_ltm: LTM terén
            terrain_stm: STM terén
            ltm_centers: LTM paměťová centra
            stm_centers: STM paměťová centra
            return_gate: zda vrátit gate hodnoty
            
        Returns:
            x_out: [B, T, d_model] vstup pro MLP
            emotions: [B, T, d_emotion] emoce pro FiLM (nebo None)
            gate: [B, T, 1] gate hodnoty (pokud return_gate)
        """
        # Normalizace
        x_normed = self.norm(x)
        
        # Projekce do paměťových prostorů
        q_ltm = self.proj.project_to_ltm(x_normed)  # [B, T, 64]
        q_stm = self.proj.project_to_stm(x_normed)  # [B, T, 16]
        
        # Terrain Prior - posouvá dotazy podle historie v terénu
        # Vrací posunuté dotazy pro LTM i STM
        q_tilde_ltm, q_tilde_stm, g_prior, terrain_signal = self.terrain_prior(
            q_ltm, q_stm, terrain_ltm, terrain_stm
        )
        
        # Memory Attention - čte z LTM i STM center
        M, E, g = self.memory_attention(
            x_normed, q_tilde_ltm, q_tilde_stm, g_prior,
            ltm_centers, stm_centers
        )
        
        # Gated residual: Y' = Y + g ⊙ W_m M
        x_out = x + g * M
        
        if return_gate:
            return x_out, E, g
        
        return x_out, E, None
    
    def apply_emotion_film(
        self,
        x: torch.Tensor,
        emotions: torch.Tensor
    ) -> torch.Tensor:
        """
        Aplikuje FiLM modulaci na MLP výstup.
        
        Args:
            x: [B, T, d_model] MLP výstup
            emotions: [B, T, d_emotion] emoce z Memory Attention
        """
        if self.use_emotion_film and emotions is not None:
            return self.emotion_film(x, emotions)
        return x


class CognitiveMemoryLayer(nn.Module):
    """
    Kompletní kognitivní paměťová vrstva.
    
    Obsahuje:
    - LTM centra + 3D terén
    - STM centra + 3D terén  
    - MemoryBlock pro každou relevantnív vrstvu transformeru
    - Writer pro zápis
    - Konsolidaci STM→LTM
    """
    
    def __init__(
        self,
        config,  # MemoryConfig
        device: str = "cpu"
    ):
        super().__init__()
        self.config = config
        self.device = device
        
        # LTM komponenty
        self.ltm_terrain = Terrain3D(
            resolution=config.terrain_resolution,
            n_emotions=config.d_emotion,
            alpha_h=config.terrain_ltm_alpha_h,
            alpha_e=config.terrain_ltm_alpha_e,
            leak=config.terrain_ltm_lambda,
            device=device
        )
        
        self.ltm_centers = MemoryCenters(
            n_centers=config.n_ltm_centers,
            d_key=config.d_memory_key,
            d_value=config.d_memory_value,
            d_emotion=config.d_emotion,
            sigma_read=config.ltm_sigma_read,
            sigma_write=config.ltm_sigma_write,
            leak=config.ltm_leak,
            leak_emotion=config.ltm_leak_emotion,
            leak_value=config.ltm_leak_value,
            alpha_value=config.ltm_alpha_value,
            alpha_emotion=config.ltm_alpha_emotion,
            alpha_key=config.ltm_alpha_key,
            device=device
        )
        
        # STM komponenty
        self.stm_terrain = Terrain3D(
            resolution=config.terrain_resolution,
            n_emotions=config.d_emotion,
            alpha_h=config.terrain_stm_alpha_h,
            alpha_e=config.terrain_stm_alpha_e,
            leak=config.terrain_stm_lambda,
            device=device
        )
        
        self.stm_centers = MemoryCenters(
            n_centers=config.n_stm_centers,
            d_key=config.d_stm_key,
            d_value=config.d_memory_value,
            d_emotion=config.d_emotion,
            sigma_read=config.stm_sigma_read,
            sigma_write=config.stm_sigma_write,
            leak=config.stm_leak,
            leak_emotion=config.stm_leak_emotion,
            leak_value=config.stm_leak_value,
            alpha_value=config.stm_alpha_value,
            alpha_emotion=config.stm_alpha_emotion,
            alpha_key=0.0,  # STM klíče nehýbat
            device=device
        )
        
        # MemoryBlock
        self.memory_block = MemoryBlock(
            d_model=config.d_model,
            d_ltm_key=config.d_memory_key,
            d_stm_key=config.d_stm_key,
            d_value=config.d_memory_value,
            d_emotion=config.d_emotion,
            gate_bias=config.gate_bias,
            gate_prior_weight=config.gate_prior_weight,
            terrain_prior_beta=config.terrain_prior_beta,
            terrain_prior_gate_bias=config.terrain_prior_gate_bias,
            ltm_top_k=config.ltm_top_k_read,
            stm_top_k=config.stm_top_k_read
        )
        
        # Únava
        self.fatigue = 0.0
    
    def forward(
        self,
        x: torch.Tensor,
        return_gate: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass celé kognitivní paměti.
        """
        return self.memory_block(
            x,
            self.ltm_terrain,
            self.stm_terrain,
            self.ltm_centers,
            self.stm_centers,
            return_gate=return_gate
        )
    
    def step_homeostasis(self):
        """Jeden krok homeostázy pro všechny komponenty."""
        self.ltm_terrain.step()
        self.stm_terrain.step()
        self.ltm_centers.homeostasis_step()
        self.stm_centers.homeostasis_step()
    
    def get_stats(self) -> Dict:
        """Vrátí statistiky celého systému."""
        return {
            "ltm_terrain": self.ltm_terrain.get_stats(),
            "stm_terrain": self.stm_terrain.get_stats(),
            "ltm_centers": self.ltm_centers.get_stats(),
            "stm_centers": self.stm_centers.get_stats(),
            "fatigue": self.fatigue,
        }

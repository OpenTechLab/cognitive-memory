# cognitive_memory/memory_attention.py
"""
MemoryAttention modul.

Implementuje RBF kernel čtení z paměťových center a finální gating.

Matematika z plánu:
- w_{b,t,i} = exp(-||q_tilde - K_i||^2 / 2σ^2)
- π_{b,t,i} = softmax_i(log(ε+h_i) + log(w_{b,t,i}))
- r^V = Σ π_i V_i
- r^E = Σ π_i e_i
- g = σ(W_g [X; r^V] + u * g_prior)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .memory_centers import MemoryCenters


class MemoryAttention(nn.Module):
    """
    Memory Attention s RBF kernel a gating.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        d_key: int = 64,
        d_value: int = 128,
        d_emotion: int = 4,
        gate_bias: float = -2.5,
        gate_prior_weight: float = 0.5,
        top_k: int = 32
    ):
        super().__init__()
        self.d_model = d_model
        self.d_key = d_key
        self.d_value = d_value
        self.d_emotion = d_emotion
        self.top_k = top_k
        
        # Gate: g = σ(W_g [X; r^V] + u * g_prior)
        self.W_g = nn.Linear(d_model + d_value, 1)
        nn.init.xavier_uniform_(self.W_g.weight)
        nn.init.constant_(self.W_g.bias, gate_bias)
        
        self.gate_prior_weight = nn.Parameter(torch.tensor(gate_prior_weight))
    
    def forward(
        self,
        x: torch.Tensor,
        q_tilde: torch.Tensor,
        g_prior: torch.Tensor,
        memory: MemoryCenters
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Čte z paměti a aplikuje gating.
        
        Args:
            x: [B, T, d_model] hidden states z SA
            q_tilde: [B, T, d_key] posunuté dotazy z TerrainPrior
            g_prior: [B, T, 1] gate prior z TerrainPrior
            memory: MemoryCenters instance
            
        Returns:
            M: [B, T, d_value] paměťový kontext
            E: [B, T, d_emotion] emoční kontext
            g: [B, T, 1] finální gate
        """
        B, T, _ = x.shape
        
        # RBF čtení z paměti
        r_V, r_E, weights, indices = memory.read(q_tilde, top_k=self.top_k)
        
        # Finální gate
        gate_input = torch.cat([x, r_V], dim=-1)  # [B, T, d_model + d_value]
        g = torch.sigmoid(
            self.W_g(gate_input) + 
            self.gate_prior_weight * g_prior
        )  # [B, T, 1]
        
        return r_V, r_E, g


class DualMemoryAttention(nn.Module):
    """
    Kombinované Memory Attention pro LTM i STM.
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
        ltm_top_k: int = 32,
        stm_top_k: int = 16
    ):
        super().__init__()
        
        self.ltm_attention = MemoryAttention(
            d_model=d_model,
            d_key=d_ltm_key,
            d_value=d_value,
            d_emotion=d_emotion,
            gate_bias=gate_bias,
            gate_prior_weight=gate_prior_weight,
            top_k=ltm_top_k
        )
        
        self.stm_attention = MemoryAttention(
            d_model=d_model,
            d_key=d_stm_key,
            d_value=d_value,
            d_emotion=d_emotion,
            gate_bias=gate_bias + 0.5,  # STM trochu otevřenější
            gate_prior_weight=gate_prior_weight,
            top_k=stm_top_k
        )
        
        # Kombinační váhy pro LTM vs STM
        self.ltm_weight = nn.Parameter(torch.tensor(0.7))
        self.stm_weight = nn.Parameter(torch.tensor(0.3))
        
        # Projekce zpátky do d_model
        self.output_proj = nn.Linear(d_value, d_model)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        q_tilde_ltm: torch.Tensor,
        q_tilde_stm: torch.Tensor,
        g_prior: torch.Tensor,
        ltm_centers: MemoryCenters,
        stm_centers: MemoryCenters
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, d_model] hidden states
            q_tilde_ltm: [B, T, 64] posunuté LTM dotazy
            q_tilde_stm: [B, T, 16] posunuté STM dotazy
            g_prior: [B, T, 1] kombinovaný gate prior
            ltm_centers: LTM centra
            stm_centers: STM centra
            
        Returns:
            M: [B, T, d_model] paměťový kontext (projektovaný)
            E: [B, T, d_emotion] emoční kontext
            g: [B, T, 1] finální gate
        """
        # LTM čtení
        r_V_ltm, r_E_ltm, g_ltm = self.ltm_attention(
            x, q_tilde_ltm, g_prior, ltm_centers
        )
        
        # STM čtení
        r_V_stm, r_E_stm, g_stm = self.stm_attention(
            x, q_tilde_stm, g_prior, stm_centers
        )
        
        # Kombinace
        ltm_w = torch.sigmoid(self.ltm_weight)
        stm_w = torch.sigmoid(self.stm_weight)
        total_w = ltm_w + stm_w + 1e-8
        
        r_V = (ltm_w * r_V_ltm + stm_w * r_V_stm) / total_w
        r_E = (ltm_w * r_E_ltm + stm_w * r_E_stm) / total_w
        g = (ltm_w * g_ltm + stm_w * g_stm) / total_w
        
        # Projekce do d_model
        M = self.output_proj(r_V)
        
        return M, r_E, g

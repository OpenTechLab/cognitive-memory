# cognitive_memory/terrain_prior.py
"""
TerrainPrior modul.

Čte z 3D terénu a vytváří:
1. Posunutý dotaz (q_tilde) - citlivější na "zlomy"
2. Gate prior (g_prior) - kdy má paměť zasahovat

Matematika z plánu:
- z = C(q) ∈ [-1,1]^3
- p_H = sample(H^3, z)
- p_E = sample(E^3, z)
- q_tilde = norm(q + β_q * W_pq @ [p_H; p_E])
- g_prior = σ(a_h * p_H + a_e^T @ p_E + b_g)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .terrain_3d import Terrain3D


class TerrainPrior(nn.Module):
    """
    TerrainPrior modul pro čtení z 3D terénu a modulaci dotazů.
    """
    
    def __init__(
        self,
        d_key: int = 64,
        d_emotion: int = 4,
        beta: float = 0.02,
        gate_bias: float = -3.0
    ):
        super().__init__()
        self.d_key = d_key
        self.d_emotion = d_emotion
        self.beta = beta
        
        # Projekce terénního signálu do query space
        # W_pq: [p_H; p_E] → d_key
        self.W_pq = nn.Linear(1 + d_emotion, d_key, bias=False)
        nn.init.xavier_uniform_(self.W_pq.weight, gain=0.1)  # Malá inicializace
        
        # Gate parametry
        # g_prior = σ(a_h * p_H + a_e^T @ p_E + b_g)
        self.a_h = nn.Parameter(torch.tensor(1.0))
        self.a_e = nn.Parameter(torch.ones(d_emotion) * 0.25)
        self.b_g = nn.Parameter(torch.tensor(gate_bias))
        
        # 64D → 3D projekce (může být sdílená s ProjectionBundle)
        self.to_terrain = nn.Linear(d_key, 3)
        nn.init.xavier_uniform_(self.to_terrain.weight, gain=0.5)
        nn.init.zeros_(self.to_terrain.bias)
    
    def forward(
        self,
        queries: torch.Tensor,
        terrain: Terrain3D,
        return_terrain_signal: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Aplikuje terrain prior na dotazy.
        
        Args:
            queries: [B, T, d_key] normalizované dotazy
            terrain: Terrain3D instance
            return_terrain_signal: zda vrátit raw terénní signál
            
        Returns:
            q_tilde: [B, T, d_key] posunuté dotazy
            g_prior: [B, T, 1] gate prior
            (p_H, p_E): volitelně raw signál z terénu
        """
        B, T, D = queries.shape
        
        # 1. Projekce do 3D
        z = torch.tanh(self.to_terrain(queries))  # [B, T, 3]
        
        # 2. Vzorkování z terénu
        p_H, p_E = terrain.sample(z)  # [B, T], [B, T, 4]
        
        # 3. Posun dotazu (citlivost na "zlom")
        terrain_signal = torch.cat([p_H.unsqueeze(-1), p_E], dim=-1)  # [B, T, 5]
        delta_q = self.W_pq(terrain_signal)  # [B, T, d_key]
        q_tilde = F.normalize(queries + self.beta * delta_q, dim=-1)
        
        # 4. Gate prior
        g_prior = torch.sigmoid(
            self.a_h * p_H + 
            torch.einsum('bte,e->bt', p_E, self.a_e) + 
            self.b_g
        ).unsqueeze(-1)  # [B, T, 1]
        
        if return_terrain_signal:
            return q_tilde, g_prior, (p_H, p_E)
        
        return q_tilde, g_prior, None


class DualTerrainPrior(nn.Module):
    """
    Kombinovaný TerrainPrior pro LTM i STM terény.
    
    Podle plánu zpracovává oba terény paralelně a kombinuje jejich prior.
    """
    
    def __init__(
        self,
        d_ltm_key: int = 64,
        d_stm_key: int = 16,
        d_emotion: int = 4,
        beta_ltm: float = 0.02,
        beta_stm: float = 0.05,  # STM má silnější vliv
        gate_bias: float = -3.0
    ):
        super().__init__()
        
        self.ltm_prior = TerrainPrior(
            d_key=d_ltm_key,
            d_emotion=d_emotion,
            beta=beta_ltm,
            gate_bias=gate_bias
        )
        
        self.stm_prior = TerrainPrior(
            d_key=d_stm_key,
            d_emotion=d_emotion,
            beta=beta_stm,
            gate_bias=gate_bias - 1.0  # STM trochu otevřenější
        )
        
        # Kombinační váhy (learnable)
        self.ltm_weight = nn.Parameter(torch.tensor(0.6))
        self.stm_weight = nn.Parameter(torch.tensor(0.4))
    
    def forward(
        self,
        q_ltm: torch.Tensor,
        q_stm: torch.Tensor,
        terrain_ltm: Terrain3D,
        terrain_stm: Terrain3D
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Aplikuje terrain prior na LTM i STM dotazy.
        
        Args:
            q_ltm: [B, T, 64] LTM dotazy
            q_stm: [B, T, 16] STM dotazy
            terrain_ltm: LTM terén
            terrain_stm: STM terén
            
        Returns:
            q_tilde_ltm: [B, T, 64] posunuté LTM dotazy
            q_tilde_stm: [B, T, 16] posunuté STM dotazy
            g_prior_combined: [B, T, 1] kombinovaný gate prior
            terrain_signal: [B, T, 10] signál z obou terénů [p_H_ltm, p_E_ltm, p_H_stm, p_E_stm]
        """
        # LTM prior
        q_tilde_ltm, g_prior_ltm, (p_H_ltm, p_E_ltm) = self.ltm_prior(
            q_ltm, terrain_ltm, return_terrain_signal=True
        )
        
        # STM prior
        q_tilde_stm, g_prior_stm, (p_H_stm, p_E_stm) = self.stm_prior(
            q_stm, terrain_stm, return_terrain_signal=True
        )
        
        # Kombinovaný gate prior (vážený průměr)
        # Podle plánu: gate se otevírá když LTM NEBO STM signalizuje historii
        ltm_w = torch.sigmoid(self.ltm_weight)
        stm_w = torch.sigmoid(self.stm_weight)
        g_prior_combined = (ltm_w * g_prior_ltm + stm_w * g_prior_stm) / (ltm_w + stm_w)
        
        # Kombinovaný terénní signál pro další zpracování
        terrain_signal = torch.cat([
            p_H_ltm.unsqueeze(-1), p_E_ltm,   # [B, T, 5]
            p_H_stm.unsqueeze(-1), p_E_stm    # [B, T, 5]
        ], dim=-1)  # [B, T, 10]
        
        return q_tilde_ltm, q_tilde_stm, g_prior_combined, terrain_signal


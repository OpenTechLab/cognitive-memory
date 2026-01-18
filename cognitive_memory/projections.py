# cognitive_memory/projections.py
"""
Projekční vrstvy pro Cognitive Memory.

Implementuje:
- D → 64D projekce (hidden states → LTM klíče)
- D → 16D projekce (hidden states → STM klíče)
- 64D → 3D projekce (LTM klíče → terén)
- 16D → 3D projekce (STM klíče → terén)
- 16D → 64D projekce (STM → LTM při konsolidaci)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MemoryProjection(nn.Module):
    """
    Projekce z hidden states do paměťového prostoru.
    
    f_q: X → q = norm(W_q X + b_q)
    """
    
    def __init__(
        self,
        d_input: int,
        d_output: int,
        bias: bool = True
    ):
        super().__init__()
        self.proj = nn.Linear(d_input, d_output, bias=bias)
        
        # Inicializace pro stabilitu
        nn.init.xavier_uniform_(self.proj.weight)
        if bias:
            nn.init.zeros_(self.proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_input] nebo [N, d_input]
            
        Returns:
            q: [..., d_output] normalizovaný vektor
        """
        projected = self.proj(x)
        return F.normalize(projected, dim=-1)


class TerrainProjection(nn.Module):
    """
    Projekce z paměťového prostoru do 3D terénu.
    
    C: k → z = tanh(W_c k + b_c)
    
    Výstup v [-1, 1]^3 pro grid_sample kompatibilitu.
    """
    
    def __init__(
        self,
        d_input: int,
        d_output: int = 3
    ):
        super().__init__()
        self.proj = nn.Linear(d_input, d_output)
        
        # Inicializace
        nn.init.xavier_uniform_(self.proj.weight, gain=0.5)
        nn.init.zeros_(self.proj.bias)
    
    def forward(self, k: torch.Tensor) -> torch.Tensor:
        """
        Args:
            k: [..., d_input] paměťový klíč
            
        Returns:
            z: [..., 3] pozice v terénu [-1, 1]^3
        """
        return torch.tanh(self.proj(k))


class ConsolidationProjection(nn.Module):
    """
    Projekce z STM (16D) do LTM (64D) prostoru.
    
    U: K_stm → norm(U @ K_stm)
    """
    
    def __init__(
        self,
        d_stm: int = 16,
        d_ltm: int = 64
    ):
        super().__init__()
        self.proj = nn.Linear(d_stm, d_ltm, bias=False)
        
        # Inicializace
        nn.init.orthogonal_(self.proj.weight)
    
    def forward(self, k_stm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            k_stm: [..., d_stm]
            
        Returns:
            k_ltm: [..., d_ltm] normalizovaný
        """
        return F.normalize(self.proj(k_stm), dim=-1)


class ValueProjection(nn.Module):
    """
    Projekce pro vytvoření paměťových hodnot.
    
    W_v: k_pooled → v = W_v k + b_v
    """
    
    def __init__(
        self,
        d_input: int,
        d_value: int
    ):
        super().__init__()
        self.proj = nn.Linear(d_input, d_value)
        
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class ProjectionBundle(nn.Module):
    """
    Všechny projekce pro paměťový systém.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        d_ltm_key: int = 64,
        d_stm_key: int = 16,
        d_value: int = 128
    ):
        super().__init__()
        
        # Hidden states → LTM klíče
        self.to_ltm_key = MemoryProjection(d_model, d_ltm_key)
        
        # Hidden states → STM klíče
        self.to_stm_key = MemoryProjection(d_model, d_stm_key)
        
        # Hidden states → hodnoty
        self.to_value = ValueProjection(d_model, d_value)
        
        # LTM klíče → 3D terén
        self.ltm_to_terrain = TerrainProjection(d_ltm_key, 3)
        
        # STM klíče → 3D terén
        self.stm_to_terrain = TerrainProjection(d_stm_key, 3)
        
        # STM → LTM (konsolidace)
        self.stm_to_ltm = ConsolidationProjection(d_stm_key, d_ltm_key)
    
    def project_to_ltm(self, x: torch.Tensor) -> torch.Tensor:
        """Hidden states → LTM klíče."""
        return self.to_ltm_key(x)
    
    def project_to_stm(self, x: torch.Tensor) -> torch.Tensor:
        """Hidden states → STM klíče."""
        return self.to_stm_key(x)
    
    def project_to_value(self, x: torch.Tensor) -> torch.Tensor:
        """Hidden states → hodnoty."""
        return self.to_value(x)
    
    def ltm_to_3d(self, k: torch.Tensor) -> torch.Tensor:
        """LTM klíče → 3D pozice."""
        return self.ltm_to_terrain(k)
    
    def stm_to_3d(self, k: torch.Tensor) -> torch.Tensor:
        """STM klíče → 3D pozice."""
        return self.stm_to_terrain(k)
    
    def consolidate_key(self, k_stm: torch.Tensor) -> torch.Tensor:
        """STM klíč → LTM klíč."""
        return self.stm_to_ltm(k_stm)

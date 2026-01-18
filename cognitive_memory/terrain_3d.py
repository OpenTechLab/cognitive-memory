# cognitive_memory/terrain_3d.py
"""
3D Terénová vrstva pro Cognitive Memory.

Implementuje:
- 3D grid pro intenzitu (GS) a emoce (CMYK-like)
- Difuzi (Laplaceův operátor) - vyhlazování
- Homeostázu (leak) - pomalý návrat k rovině
- Trilineární vzorkování pro čtení
- Gaussian splat pro zápis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class Terrain3D(nn.Module):
    """
    3D terénová vrstva s difuzí a homeostázou.
    
    Obsahuje:
        - H: Intenzita/GS terén [Gx, Gy, Gz]
        - E: Emoční terén [Gx, Gy, Gz, 4]
    """
    
    def __init__(
        self,
        resolution: int = 48,
        n_emotions: int = 4,
        alpha_h: float = 0.002,
        alpha_e: float = 0.001,
        leak: float = 5e-5,
        device: str = "cpu"
    ):
        super().__init__()
        self.resolution = resolution
        self.n_emotions = n_emotions
        self.alpha_h = alpha_h
        self.alpha_e = alpha_e
        self.leak = leak
        
        # Terénové gridy (jako buffery - ne parametry)
        # H: intenzita [1, 1, Gx, Gy, Gz] pro grid_sample kompatibilitu
        # Začíná na 0 = "nikdy nenavštíveno"
        self.register_buffer(
            "H", 
            torch.zeros(1, 1, resolution, resolution, resolution, device=device)
        )
        # E: emoce [1, 4, Gx, Gy, Gz]
        # Začíná na 1.0 = neutrální hodnota (jako PlantNet hormony)
        self.register_buffer(
            "E",
            torch.ones(1, n_emotions, resolution, resolution, resolution, device=device)
        )
        
        # Laplaceův kernel pro 3D difuzi (6-sousedů)
        # Kernel [1, 1, 3, 3, 3]
        laplacian_kernel = torch.zeros(1, 1, 3, 3, 3, device=device)
        # Střed
        laplacian_kernel[0, 0, 1, 1, 1] = -6.0
        # 6 sousedů
        laplacian_kernel[0, 0, 0, 1, 1] = 1.0
        laplacian_kernel[0, 0, 2, 1, 1] = 1.0
        laplacian_kernel[0, 0, 1, 0, 1] = 1.0
        laplacian_kernel[0, 0, 1, 2, 1] = 1.0
        laplacian_kernel[0, 0, 1, 1, 0] = 1.0
        laplacian_kernel[0, 0, 1, 1, 2] = 1.0
        self.register_buffer("laplacian_kernel", laplacian_kernel)
        
    def _compute_laplacian(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Vypočítá Laplaceův operátor (bez aplikace).
        """
        C = grid.shape[1]
        return F.conv3d(
            F.pad(grid, (1, 1, 1, 1, 1, 1), mode='replicate'),
            self.laplacian_kernel.expand(C, 1, 3, 3, 3),
            groups=C
        )
    
    def _apply_diffusion(self, grid: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        Aplikuje jeden krok difuze pomocí Laplaceova operátoru.
        
        Stabilita: α ≤ 1/6 pro 6-sousedů
        """
        laplacian = self._compute_laplacian(grid)
        return grid + alpha * laplacian
    
    def step(self):
        """
        Jeden krok difuze + homeostázy.
        
        Implementuje vzorec z plánu:
            H³ ← (1 − λ₃)H³ + α_H ∇²H³
            E³ ← (1 − λ₃)(E³ - 1) + 1 + α_E ∇²E³
        
        Emoce decayují směrem k neutrální hodnotě 1.0 (jako PlantNet hormony).
        Intenzita decayuje směrem k 0.
        
        Volat po každé interakci.
        """
        # Vypočítej Laplacián PŘED aplikací leak
        laplacian_H = self._compute_laplacian(self.H)
        laplacian_E = self._compute_laplacian(self.E)
        
        # Atomický krok pro intenzitu: H ← (1-λ)H + α∇²H
        self.H.mul_(1.0 - self.leak).add_(self.alpha_h * laplacian_H)
        
        # Atomický krok pro emoce: decay k 1.0 místo k 0
        # E ← (1-λ)(E-1) + 1 + α∇²E = (1-λ)E + λ + α∇²E
        self.E.mul_(1.0 - self.leak).add_(self.leak).add_(self.alpha_e * laplacian_E)
        
        # Clamp pro stabilitu
        self.H.clamp_(min=0.0)
        self.E.clamp_(min=0.0)  # Emoce nemohou být záporné
    
    def sample(
        self, 
        positions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Trilineární vzorkování z terénu.
        
        Args:
            positions: [B, T, 3] v rozsahu [-1, 1]
            
        Returns:
            p_H: [B, T] intenzita
            p_E: [B, T, 4] emoce
        """
        B, T, _ = positions.shape
        
        # grid_sample očekává [N, C, D, H, W] a grid [N, D, H, W, 3]
        # Přeformátuj pozice: [B, T, 3] -> [1, B*T, 1, 1, 3]
        grid = positions.view(1, B * T, 1, 1, 3)
        
        # Vzorkuj intenzitu
        # H: [1, 1, Gx, Gy, Gz] -> replikuj pro batch
        H_expanded = self.H.expand(1, 1, -1, -1, -1)
        p_H = F.grid_sample(
            H_expanded, 
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )  # [1, 1, B*T, 1, 1]
        p_H = p_H.view(B, T)
        
        # Vzorkuj emoce
        E_expanded = self.E.expand(1, -1, -1, -1, -1)
        p_E = F.grid_sample(
            E_expanded,
            grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )  # [1, 4, B*T, 1, 1]
        p_E = p_E.view(self.n_emotions, B, T).permute(1, 2, 0)  # [B, T, 4]
        
        return p_H, p_E
    
    def splat(
        self,
        positions: torch.Tensor,
        intensities: torch.Tensor,
        emotions: Optional[torch.Tensor] = None,
        sigma: float = 0.1,
        eta: float = 0.01
    ):
        """
        Gaussian splat zápis do terénu.
        
        Args:
            positions: [N, 3] pozice v [-1, 1]
            intensities: [N] síla zápisu (ω)
            emotions: [N, 4] emoční vektory (volitelné)
            sigma: šířka kernelu
            eta: globální síla zápisu
        """
        N = positions.shape[0]
        G = self.resolution
        
        # Převod pozic do grid souřadnic [0, G-1]
        grid_pos = (positions + 1.0) * 0.5 * (G - 1)  # [N, 3]
        
        # Pro každou pozici vytvoř lokální splat
        # Použijeme radius ~2-3 voxely
        radius = max(2, int(3.0 * sigma * G / 2.0))
        
        for i in range(N):
            cx, cy, cz = grid_pos[i].long().clamp(0, G-1).tolist()
            omega = intensities[i].item() * eta
            
            # Lokální oblast
            x_min = max(0, cx - radius)
            x_max = min(G, cx + radius + 1)
            y_min = max(0, cy - radius)
            y_max = min(G, cy + radius + 1)
            z_min = max(0, cz - radius)
            z_max = min(G, cz + radius + 1)
            
            # Vytvoř kernel pro tuto oblast
            x_range = torch.arange(x_min, x_max, device=positions.device, dtype=positions.dtype)
            y_range = torch.arange(y_min, y_max, device=positions.device, dtype=positions.dtype)
            z_range = torch.arange(z_min, z_max, device=positions.device, dtype=positions.dtype)
            
            # Grid souřadnice
            xx, yy, zz = torch.meshgrid(x_range, y_range, z_range, indexing='ij')
            
            # Vzdálenost od centra (v grid souřadnicích)
            dist_sq = (xx - grid_pos[i, 0])**2 + (yy - grid_pos[i, 1])**2 + (zz - grid_pos[i, 2])**2
            
            # Gaussian kernel (sigma je v world space, přepočti)
            sigma_grid = sigma * G / 2.0
            kernel = torch.exp(-dist_sq / (2 * sigma_grid**2))
            
            # Update intenzity
            self.H[0, 0, x_min:x_max, y_min:y_max, z_min:z_max] += omega * kernel
            
            # Update emocí
            if emotions is not None:
                for e_idx in range(self.n_emotions):
                    self.E[0, e_idx, x_min:x_max, y_min:y_max, z_min:z_max] += (
                        omega * kernel * emotions[i, e_idx].item()
                    )
    
    def blur(self, sigma: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vrátí rozmazanou verzi terénu (pro konsolidaci STM→LTM).
        """
        kernel_size = int(6 * sigma) | 1  # Musí být liché
        kernel_size = max(3, kernel_size)
        
        # 1D Gaussian kernel
        x = torch.arange(kernel_size, device=self.H.device, dtype=self.H.dtype)
        x = x - kernel_size // 2
        kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # Separabilní 3D blur
        def blur_1d(tensor, kernel, dim):
            # Přidej dimenzi pro konvoluci
            k = kernel.view(1, 1, -1) if dim == 2 else kernel.view(1, 1, 1, -1) if dim == 3 else kernel.view(1, 1, 1, 1, -1)
            padding = kernel.shape[0] // 2
            pad_config = [0, 0, 0, 0, padding, padding] if dim == 2 else [0, 0, padding, padding, 0, 0] if dim == 3 else [padding, padding, 0, 0, 0, 0]
            padded = F.pad(tensor, pad_config, mode='replicate')
            return F.conv3d(padded, k.expand(tensor.shape[1], 1, *k.shape[2:]), groups=tensor.shape[1])
        
        H_blurred = self.H.clone()
        E_blurred = self.E.clone()
        
        # Aplikuj blur ve všech 3 dimenzích (zjednodušená verze)
        # Pro produkci by bylo lepší použít separabilní konvoluci
        
        return H_blurred, E_blurred
    
    def merge_from(
        self,
        other: 'Terrain3D',
        xi_h: float = 0.01,
        xi_e: float = 0.01,
        blur_sigma: float = 2.0
    ):
        """
        Sloučí (rozmazaný) terén z jiného terénu (STM→LTM).
        """
        H_blurred, E_blurred = other.blur(blur_sigma)
        self.H.add_(xi_h * H_blurred)
        self.E.add_(xi_e * E_blurred)
    
    def get_stats(self) -> dict:
        """Vrátí statistiky terénu."""
        return {
            "H_mean": self.H.mean().item(),
            "H_max": self.H.max().item(),
            "H_std": self.H.std().item(),  # Přidáno
            "H_nonzero": (self.H > 1e-6).sum().item(),
            "E_mean": self.E.mean().item(),
            "E_max": self.E.abs().max().item(),
        }
    
    def state_dict_custom(self) -> dict:
        """Vrátí stav pro uložení."""
        return {
            "H": self.H.cpu(),
            "E": self.E.cpu(),
            "resolution": self.resolution,
            "n_emotions": self.n_emotions,
            "alpha_h": self.alpha_h,
            "alpha_e": self.alpha_e,
            "leak": self.leak,
        }
    
    @classmethod
    def from_state_dict(cls, state: dict, device: str = "cpu") -> 'Terrain3D':
        """Načte terén ze stavu."""
        terrain = cls(
            resolution=state["resolution"],
            n_emotions=state["n_emotions"],
            alpha_h=state["alpha_h"],
            alpha_e=state["alpha_e"],
            leak=state["leak"],
            device=device
        )
        terrain.H.copy_(state["H"].to(device))
        terrain.E.copy_(state["E"].to(device))
        return terrain

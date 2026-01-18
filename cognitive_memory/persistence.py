# cognitive_memory/persistence.py
"""
Persistence pro Cognitive Memory.

Ukládá a načítá:
- LTM centra + terén
- STM centra + terén
- Projekční vrstvy
- Konsolidátor stav
- Konfigurace
"""

import torch
import os
from typing import Optional, Dict
from pathlib import Path

from .config import MemoryConfig
from .terrain_3d import Terrain3D
from .memory_centers import MemoryCenters


def save_memory_state(
    filepath: str,
    config: MemoryConfig,
    ltm_centers: MemoryCenters,
    stm_centers: MemoryCenters,
    ltm_terrain: Terrain3D,
    stm_terrain: Terrain3D,
    projections: Optional[torch.nn.Module] = None,
    consolidator_state: Optional[Dict] = None,
    additional_state: Optional[Dict] = None
):
    """
    Uloží kompletní stav kognitivní paměti.
    
    Args:
        filepath: Cesta k souboru (.pt)
        config: Konfigurace
        ltm_centers: LTM paměťová centra
        stm_centers: STM paměťová centra
        ltm_terrain: LTM 3D terén
        stm_terrain: STM 3D terén
        projections: Volitelně projekční vrstvy
        consolidator_state: Volitelně stav konsolidátoru
        additional_state: Další data k uložení
    """
    state = {
        "version": "2.0",
        "config": {
            "d_model": config.d_model,
            "d_memory_key": config.d_memory_key,
            "d_stm_key": config.d_stm_key,
            "d_memory_value": config.d_memory_value,
            "d_emotion": config.d_emotion,
            "terrain_resolution": config.terrain_resolution,
            "n_ltm_centers": config.n_ltm_centers,
            "n_stm_centers": config.n_stm_centers,
        },
        "ltm_centers": ltm_centers.state_dict_custom(),
        "stm_centers": stm_centers.state_dict_custom(),
        "ltm_terrain": ltm_terrain.state_dict_custom(),
        "stm_terrain": stm_terrain.state_dict_custom(),
    }
    
    if projections is not None:
        state["projections"] = projections.state_dict()
    
    if consolidator_state is not None:
        state["consolidator"] = consolidator_state
    
    if additional_state is not None:
        state["additional"] = additional_state
    
    # Vytvoř adresář pokud neexistuje
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(state, filepath)
    
    # Statistiky
    ltm_stats = ltm_centers.get_stats()
    stm_stats = stm_centers.get_stats()
    
    print(f"CognitiveMemory: Saved to '{filepath}'")
    print(f"  LTM: {ltm_stats['n_active']}/{ltm_stats['n_total']} centers")
    print(f"  STM: {stm_stats['n_active']}/{stm_stats['n_total']} centers")


def load_memory_state(
    filepath: str,
    device: str = "cpu"
) -> Dict:
    """
    Načte stav kognitivní paměti.
    
    Args:
        filepath: Cesta k souboru
        device: Cílové zařízení
        
    Returns:
        Dict obsahující všechny komponenty
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Memory state file not found: {filepath}")
    
    state = torch.load(filepath, map_location=device)
    
    version = state.get("version", "1.0")
    if version != "2.0":
        print(f"Warning: Loading old memory state version {version}")
    
    # Rekonstruuj komponenty
    result = {
        "config_dict": state.get("config", {}),
        "ltm_centers": MemoryCenters.from_state_dict(state["ltm_centers"], device),
        "stm_centers": MemoryCenters.from_state_dict(state["stm_centers"], device),
        "ltm_terrain": Terrain3D.from_state_dict(state["ltm_terrain"], device),
        "stm_terrain": Terrain3D.from_state_dict(state["stm_terrain"], device),
        "projections_state": state.get("projections"),
        "consolidator_state": state.get("consolidator"),
        "additional": state.get("additional"),
    }
    
    # Statistiky
    ltm_stats = result["ltm_centers"].get_stats()
    stm_stats = result["stm_centers"].get_stats()
    
    print(f"CognitiveMemory: Loaded from '{filepath}'")
    print(f"  LTM: {ltm_stats['n_active']}/{ltm_stats['n_total']} centers")
    print(f"  STM: {stm_stats['n_active']}/{stm_stats['n_total']} centers")
    
    return result


def create_or_load_memory(
    config: MemoryConfig,
    device: str = "cpu"
) -> Dict:
    """
    Vytvoří novou paměť nebo načte existující.
    
    Args:
        config: Konfigurace
        device: Zařízení
        
    Returns:
        Dict s komponentami paměti
    """
    filepath = config.state_file
    
    if os.path.exists(filepath):
        try:
            return load_memory_state(filepath, device)
        except Exception as e:
            print(f"CognitiveMemory: Failed to load state: {e}")
            print("CognitiveMemory: Creating new memory...")
    
    # Vytvoř nové komponenty
    ltm_centers = MemoryCenters(
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
    
    stm_centers = MemoryCenters(
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
        alpha_key=0.0,
        device=device
    )
    
    ltm_terrain = Terrain3D(
        resolution=config.terrain_resolution,
        n_emotions=config.d_emotion,
        alpha_h=config.terrain_ltm_alpha_h,
        alpha_e=config.terrain_ltm_alpha_e,
        leak=config.terrain_ltm_lambda,
        device=device
    )
    
    stm_terrain = Terrain3D(
        resolution=config.terrain_resolution,
        n_emotions=config.d_emotion,
        alpha_h=config.terrain_stm_alpha_h,
        alpha_e=config.terrain_stm_alpha_e,
        leak=config.terrain_stm_lambda,
        device=device
    )
    
    print("CognitiveMemory: Created new memory")
    print(f"  LTM: {config.n_ltm_centers} centers, {config.d_memory_key}D keys")
    print(f"  STM: {config.n_stm_centers} centers, {config.d_stm_key}D keys")
    print(f"  Terrain: {config.terrain_resolution}^3 resolution")
    
    return {
        "config_dict": {},
        "ltm_centers": ltm_centers,
        "stm_centers": stm_centers,
        "ltm_terrain": ltm_terrain,
        "stm_terrain": stm_terrain,
        "projections_state": None,
        "consolidator_state": None,
        "additional": None,
    }


class MemoryCheckpointer:
    """
    Automatický checkpointer pro paměť.
    """
    
    def __init__(
        self,
        base_path: str,
        config: MemoryConfig,
        checkpoint_interval: int = 1000,  # Kroků
        max_checkpoints: int = 5
    ):
        self.base_path = Path(base_path)
        self.config = config
        self.checkpoint_interval = checkpoint_interval
        self.max_checkpoints = max_checkpoints
        
        self.steps_since_checkpoint = 0
        self.checkpoint_count = 0
        
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def step(
        self,
        ltm_centers: MemoryCenters,
        stm_centers: MemoryCenters,
        ltm_terrain: Terrain3D,
        stm_terrain: Terrain3D,
        **kwargs
    ) -> Optional[str]:
        """
        Jeden krok - případně vytvoří checkpoint.
        
        Returns:
            Cesta k checkpointu pokud byl vytvořen, jinak None
        """
        self.steps_since_checkpoint += 1
        
        if self.steps_since_checkpoint >= self.checkpoint_interval:
            checkpoint_path = self._create_checkpoint(
                ltm_centers, stm_centers,
                ltm_terrain, stm_terrain,
                **kwargs
            )
            self.steps_since_checkpoint = 0
            return checkpoint_path
        
        return None
    
    def _create_checkpoint(
        self,
        ltm_centers: MemoryCenters,
        stm_centers: MemoryCenters,
        ltm_terrain: Terrain3D,
        stm_terrain: Terrain3D,
        **kwargs
    ) -> str:
        """Vytvoří checkpoint."""
        self.checkpoint_count += 1
        filename = f"checkpoint_{self.checkpoint_count:05d}.pt"
        filepath = str(self.base_path / filename)
        
        save_memory_state(
            filepath,
            self.config,
            ltm_centers,
            stm_centers,
            ltm_terrain,
            stm_terrain,
            **kwargs
        )
        
        # Smaž staré checkpointy
        self._cleanup_old_checkpoints()
        
        return filepath
    
    def _cleanup_old_checkpoints(self):
        """Smaže nejstarší checkpointy pokud je jich moc."""
        checkpoints = sorted(self.base_path.glob("checkpoint_*.pt"))
        
        while len(checkpoints) > self.max_checkpoints:
            oldest = checkpoints.pop(0)
            oldest.unlink()
            print(f"CognitiveMemory: Deleted old checkpoint '{oldest}'")

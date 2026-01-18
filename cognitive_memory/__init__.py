# cognitive_memory/__init__.py
"""
Cognitive Memory System pro BioCortexAI.

Dvouvrstvá architektura perzistentní paměti:
- LTM (Long-Term Memory): 64D kernel-pole s poločasem ~1 rok
- STM (Short-Term Memory): 16D buffer s poločasem dny-týdny

Každá vrstva má:
- Paměťová centra (K, V, h, e)
- 3D terén s difuzí a homeostázou

Integrace do transformeru:
- MemoryBlock mezi Self-Attention a MLP
- TerrainPrior pro modulaci dotazů
- MemoryAttention s RBF kernel a gating

Konsolidace (spánek):
- STM → LTM přenos významných vzpomínek
- Normalizace STM místo vymazání
"""

from .config import MemoryConfig, DEFAULT_CONFIG
from .terrain_3d import Terrain3D
from .memory_centers import MemoryCenters
from .projections import (
    MemoryProjection,
    TerrainProjection,
    ConsolidationProjection,
    ValueProjection,
    ProjectionBundle
)
from .terrain_prior import TerrainPrior, DualTerrainPrior
from .memory_attention import MemoryAttention, DualMemoryAttention
from .memory_block import MemoryBlock, EmotionFiLM, CognitiveMemoryLayer
from .writer import MemoryWriter, SegmentBuffer
from .consolidation import SleepConsolidator, AutomaticConsolidator
from .persistence import (
    save_memory_state,
    load_memory_state, 
    create_or_load_memory,
    MemoryCheckpointer
)


__all__ = [
    # Config
    "MemoryConfig",
    "DEFAULT_CONFIG",
    
    # Core components
    "Terrain3D",
    "MemoryCenters",
    
    # Projections
    "MemoryProjection",
    "TerrainProjection", 
    "ConsolidationProjection",
    "ValueProjection",
    "ProjectionBundle",
    
    # Reading
    "TerrainPrior",
    "DualTerrainPrior",
    "MemoryAttention",
    "DualMemoryAttention",
    
    # Integration
    "MemoryBlock",
    "EmotionFiLM",
    "CognitiveMemoryLayer",
    
    # Writing
    "MemoryWriter",
    "SegmentBuffer",
    
    # Consolidation
    "SleepConsolidator",
    "AutomaticConsolidator",
    
    # Persistence
    "save_memory_state",
    "load_memory_state",
    "create_or_load_memory",
    "MemoryCheckpointer",
]


def create_cognitive_memory(
    d_model: int = 256,
    device: str = "cpu",
    state_file: str = "cognitive_memory_state.pt",
    **kwargs
) -> CognitiveMemoryLayer:
    """
    Factory funkce pro vytvoření kompletní kognitivní paměti.
    
    Args:
        d_model: Dimenze hidden states transformeru
        device: Zařízení (cpu/cuda)
        state_file: Cesta k souboru pro persistenci
        **kwargs: Další parametry pro MemoryConfig
        
    Returns:
        CognitiveMemoryLayer instance
    """
    config = MemoryConfig(
        d_model=d_model,
        state_file=state_file,
        **kwargs
    )
    
    # Pokus o načtení existujícího stavu
    memory_state = create_or_load_memory(config, device)
    
    # Vytvoř vrstvu
    layer = CognitiveMemoryLayer(config, device)
    
    # Nastav komponenty z načteného stavu
    if memory_state["ltm_centers"].get_n_active() > 0:
        layer.ltm_centers = memory_state["ltm_centers"]
        layer.stm_centers = memory_state["stm_centers"]
        layer.ltm_terrain = memory_state["ltm_terrain"]
        layer.stm_terrain = memory_state["stm_terrain"]
    
    return layer

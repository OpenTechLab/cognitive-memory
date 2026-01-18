# 🧠 Cognitive Memory

**Biologically Inspired Persistent Memory System for Large Language Models**

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

---

## 🌟 Overview

Cognitive Memory implements a **two-layer persistent memory architecture** for LLMs, inspired by biological memory systems:

| Biological Structure | System Component | Function |
|---------------------|------------------|----------|
| **Hippocampus** | STM (16D) | Fast encoding, limited capacity |
| **Neocortex** | LTM (64D) | Slow consolidation, long-term retention |
| **Sleep (SWS)** | Consolidation | Transfer memories from STM to LTM |

### ✨ Key Features

- 🧬 **Biologically Inspired**: Two-layer STM/LTM architecture with sleep-like consolidation
- 🌊 **3D Diffusion Terrains**: Spatial memory representation with homeostasis
- 🎯 **RBF Kernel Operations**: Smooth, localized memory read/write
- 💭 **Emotional Coloring**: 4D emotional vectors (dopamine, serotonin, cortisol, oxytocin)
- ⚡ **Transformer Integration**: Memory Attention blocks with gating mechanism
- 📦 **Persistent Storage**: Save/load complete memory state
- 🔧 **Configurable**: Coefficients calibrated for yearly operation

---

## 📦 Installation

### From Source

```bash
git clone https://github.com/OpenTechLab/cognitive-memory.git
cd cognitive-memory
pip install -e .
```

### Requirements

```bash
pip install torch>=2.0.0 numpy>=1.20.0 scipy>=1.7.0
```

---

## 🚀 Quick Start

```python
import torch
from cognitive_memory import CognitiveMemorySystem, MemoryConfig

# Create memory system
config = MemoryConfig(d_model=256, n_ltm_centers=1024)
memory = CognitiveMemorySystem(config, device="cuda")

# Read from memory during inference
hidden_states = torch.randn(1, 32, 256)
memory_context, emotions, gate = memory.read(hidden_states, layer_idx=7)

# Write to memory after generation  
segment_states = torch.randn(5, 256)
memory.write(segment_states=segment_states, emotions=torch.randn(5, 4))

# Automatic consolidation (STM → LTM)
if memory.should_consolidate():
    stats = memory.consolidate()
    
# Persist memory state
memory.save("memory_state.pt")
```

### Transformer Integration

```python
from cognitive_memory import MemoryBlock

class TransformerBlockWithMemory(nn.Module):
    def __init__(self, d_model, config):
        super().__init__()
        self.attn = Attention(...)
        self.memory_block = MemoryBlock(config)
        self.ffn = FeedForward(...)
    
    def forward(self, x, memory_state):
        y = x + self.attn(self.ln1(x))
        
        if memory_state is not None:
            m, emotions, gate = self.memory_block(self.ln_memory(y), memory_state)
            y = y + gate * m  # Gated residual
        
        return y + self.ffn(self.ln2(y))
```

---

## 📂 Project Structure

```
cognitive-memory/
├── README.md                  # This file (English)
├── README_CZ.md              # Czech documentation
├── LICENSE                    # CC BY-NC 4.0
├── setup.py                   # Installation script
├── requirements.txt           # Dependencies
│
├── cognitive_memory/          # Main package
│   ├── __init__.py           # Public API
│   ├── config.py             # Configuration
│   ├── memory_centers.py     # RBF kernel centers
│   ├── terrain_3d.py         # 3D diffusion terrains
│   ├── memory_block.py       # Transformer integration
│   ├── memory_attention.py   # Memory attention mechanism
│   ├── terrain_prior.py      # 3D→64D query bias
│   ├── writer.py             # Memory writing
│   ├── consolidation.py      # Sleep consolidation
│   ├── projections.py        # Dimension projections
│   └── persistence.py        # State save/load
│
└── tests/                     # Validation & benchmarks
    ├── README.md             # Test documentation
    ├── test_memory_fundamentals.py
    ├── stress_test_memory.py
    ├── ablation_study.py
    └── visualize_*.py
```

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         COGNITIVE MEMORY                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────┐         ┌──────────────────────┐      │
│  │   LTM (64D)          │◄────────┤   STM (16D)          │      │
│  │   half-life ~1yr     │  sleep  │   half-life ~weeks   │      │
│  │  ┌───────────────┐   │         │  ┌───────────────┐   │      │
│  │  │ RBF Centers   │   │         │  │ RBF Centers   │   │      │
│  │  │ K,V,h,e       │   │         │  │ K,V,h,e       │   │      │
│  │  └───────────────┘   │         │  └───────────────┘   │      │
│  │  ┌───────────────┐   │         │  ┌───────────────┐   │      │
│  │  │ 3D Terrain    │   │         │  │ 3D Terrain    │   │      │
│  │  │ (diffusion)   │   │         │  │ (fast diff.)  │   │      │
│  │  └───────────────┘   │         │  └───────────────┘   │      │
│  └──────────────────────┘         └──────────────────────┘      │
│                              ↓                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              TransformerBlock Memory Attention            │   │
│  │  Y = X + SA(LN(X))                                        │   │
│  │  M = MemAttn(LN(Y))  ← TerrainPrior + RBF read            │   │
│  │  Y' = Y + g ⊙ W_m M  ← Gated memory injection             │   │
│  │  X_out = Y' + MLP(LN(Y'))                                 │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Configuration

Default coefficients are calibrated for **~50 interactions/day** = **~18,250 steps/year**.

```python
from cognitive_memory import MemoryConfig

config = MemoryConfig(
    # Dimensions
    d_model=256,
    d_memory_key=64,        # LTM key dimension
    d_stm_key=16,           # STM key dimension
    
    # LTM (half-life ~1 year)
    ltm_leak=3.8e-5,
    ltm_sigma_read=0.5,
    ltm_sigma_write=0.15,   # Sharp traces
    
    # STM (half-life days-weeks)  
    stm_leak=5e-3,
    
    # Consolidation
    fatigue_threshold=5.0,
    consolidation_kappa=0.8,
)
```

---

## 🧪 Testing & Validation

```bash
# Basic tests (~3 sec)
python tests/test_memory_fundamentals.py

# Stress test (~5 min, 9000 interactions)
python tests/stress_test_memory.py

# Complete suite with visualizations
python tests/run_full_memory_suite.py
```

### Validated Metrics (2026-01-14)

| Metric | Value | Status |
|--------|-------|--------|
| LTM Active Centers | 459 | ✅ Excellent |
| Retention (1000 steps) | 100% | ✅ |
| Self-Reconstruction | 0.92 | ✅ Very good |
| Retrieval Accuracy | 100% | ✅ |

---

## 📚 Documentation

- **[cognitive_memory/README.md](cognitive_memory/README.md)** - Detailed API documentation
- **[tests/README.md](tests/README.md)** - Testing framework guide
- **Mathematical description** included in package README

---

## 📄 Scientific Papers

This implementation is based on the following peer-reviewed publications:

### Theoretical Foundation

**Persistent Memory for Decoder-Only Transformers: Latent Terrain, Diffusion, Homeostasis, and Emotional Stabilization**

> Proposes a persistent memory architecture combining sharp representations in latent space (64D) with a compressed 3D terrain layer featuring diffusion and homeostatic regulation. Memory traces carry emotional components that diffuse into terrain and stabilize responses during reading.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18198327.svg)](https://doi.org/10.5281/zenodo.18198327)

### Implementation & Validation

**Implementation of Persistent Latent Memory for Decoder Transformers**

> Presents the complete implementation with experimental validation simulating months of operation. Demonstrates key invariants: **constant data size** (~500 MB fixed volume) and **constant low read/write latency** (microseconds, independent of memory age). Ablation study confirms benefits of diffusion and STM layer.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18267378.svg)](https://doi.org/10.5281/zenodo.18267378)

---

## 📄 License

**CC BY-NC 4.0** (Creative Commons Attribution-NonCommercial 4.0)

✅ Allowed: Research, education, modification, distribution (with attribution)  
❌ Prohibited: Commercial use without license

For commercial licensing: **opentechlab@opentechlab.cz**

---

## 🤝 Citation

```bibtex
@software{cognitive_memory_2026,
  author = {Seidl, Michal},
  title = {Cognitive Memory: Biologically Inspired Persistent Memory for LLMs},
  year = {2026},
  publisher = {OpenTechLab Jablonec nad Nisou},
  version = {2.0-beta},
  url = {https://github.com/OpenTechLab/cognitive-memory}
}
```

---

## 📬 Contact

- **Email**: vyvoj@opentechlab.cz
- **Web**: [www.opentechlab.cz](https://www.opentechlab.cz)
- **Project**: BioCortexAI Framework

---

**Framework:** BioCortexAI v2.0-beta  
**Author:** Michal Seidl, OpenTechLab Jablonec nad Nisou s.r.o.  
**Status:** ✅ Production-ready

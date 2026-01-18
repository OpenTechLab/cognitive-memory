# 🧠 Cognitive Memory - Persistent Memory for LLMs

Biologically inspired long-term memory system for large language models. Implements a two-layer memory architecture (STM/LTM) with 3D diffusion terrains and RBF kernel operations.

> **Version:** 2.0-beta  
> **Status:** ✅ Fully functional and validated  
> **Origin:** BioCortexAI Framework  
> **License:** CC BY-NC 4.0

---

## 📖 Table of Contents

- [Concept and Inspiration](#-concept-and-inspiration)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Folder Structure](#-folder-structure)
- [Mathematical Description](#-mathematical-description)
- [Configuration](#-configuration)
- [Testing](#-testing)
- [Citation](#-citation)
- [Support](#-support)

---

## 💡 Concept and Inspiration

### Biological Analogy

Cognitive Memory is inspired by the biological memory system:

| Biological Structure | System Analogy | Function |
|---------------------|----------------|----------|
| **Hippocampus** | STM (16D) | Fast encoding, limited capacity |
| **Neocortex** | LTM (64D) | Slow consolidation, long-term retention |
| **Sleep (SWS)** | Consolidation | Transfer of memories from STM to LTM |

**Key Properties:**
- **Diffusion**: Traces spread over time (Laplacian operator)
- **Homeostasis**: Slow return to equilibrium state  
- **Grooves**: Repeated activity creates "paths" that are more sensitive to reactivation
- **Emotional Coloring**: 4D emotional vector (dopamine, serotonin, cortisol, oxytocin)

---

## 🏗️ Architecture

### Two-Layer System

```
┌─────────────────────────────────────────────────────────────────┐
│                         COGNITIVE MEMORY                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────┐         ┌──────────────────────┐      │
│  │   LTM (64D)          │◄────────┤   STM (16D)          │      │
│  │   half-life ~1yr     │  sleep  │   half-life ~weeks   │      │
│  │                      │         │                      │      │
│  │  ┌───────────────┐   │         │  ┌───────────────┐   │      │
│  │  │ RBF Centers   │   │         │  │ RBF Centers   │   │      │
│  │  │ K: [N, 64]    │   │         │  │ K: [M, 16]    │   │      │
│  │  │ V: [N, 128]   │   │         │  │ V: [M, 128]   │   │      │
│  │  │ h: [N]        │   │         │  │ h: [M]        │   │      │
│  │  │ e: [N, 4]     │   │         │  │ e: [M, 4]     │   │      │
│  │  └───────────────┘   │         │  └───────────────┘   │      │
│  │         ↕             │         │         ↕             │      │
│  │  ┌───────────────┐   │         │  ┌───────────────┐   │      │
│  │  │ 3D Terrain    │   │         │  │ 3D Terrain    │   │      │
│  │  │ H³: [48³]     │   │         │  │ H³: [48³]     │   │      │
│  │  │ E³: [48³, 4]  │   │         │  │ E³: [48³, 4]  │   │      │
│  │  │ (diffusion)   │   │         │  │ (fast diff.)  │   │      │
│  │  └───────────────┘   │         │  └───────────────┘   │      │
│  └──────────────────────┘         └──────────────────────┘      │
│            ↕                                  ↕                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              TransformerBlock Memory Attention            │   │
│  │  Y = X + SA(LN(X))                                        │   │
│  │  M = MemAttn(LN(Y))  ← TerrainPrior + RBF read            │   │
│  │  Y' = Y + g ⊙ W_m M                                       │   │
│  │  X_out = Y' + MLP(LN(Y'))                                 │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Data Structures

#### Memory Centers (RBF Kernel-Poles)

| Component | Dimension | Description |
|-----------|-----------|-------------|
| **K** | R^64 (LTM) / R^16 (STM) | Normalized key (semantic position) |
| **V** | R^128 | Value (what to recall) |
| **h** | R^+ | Intensity/depth of trace (GS) |
| **e** | R^4 | Emotional vector (dopamine, serotonin, cortisol, oxytocin) |
| **usage** | Z^+ | Usage counter (for pruning) |
| **age** | Z^+ | Center age |

#### 3D Terrains

```
Resolution: 48 × 48 × 48 voxels
H³ ∈ R^(48×48×48)       # Intensity (GS/"foam")
E³ ∈ R^(48×48×48×4)     # Emotional trace (4 hormones)
```

**Terrain Physics:**
- Diffusion: Laplacian operator (6-neighbors)
- Homeostasis: Exponential decay
- Projection: 64D → 3D via linear layer + tanh

---

## 📦 Installation

### Dependencies

```bash
pip install torch>=2.0.0
pip install numpy>=1.20.0
pip install scipy>=1.7.0
```

### Integration into Project

```python
from cognitive_memory import CognitiveMemorySystem, MemoryConfig

# Create configuration
config = MemoryConfig(
    d_model=256,              # Transformer dimension
    n_ltm_centers=1024,       # Number of LTM centers
    ltm_leak=3.8e-5,          # Leak for yearly operation
)

# Initialize system
memory = CognitiveMemorySystem(config, device="cuda")
```

---

## 🚀 Quick Start

### Basic Usage

```python
import torch
from cognitive_memory import CognitiveMemorySystem, MemoryConfig

# 1. Create memory system
config = MemoryConfig()
memory = CognitiveMemorySystem(config, device="cpu")

# 2. Reading from memory (during inference)
hidden_states = torch.randn(1, 32, 256)  # [B, T, D]

# Memory Attention (for Transformer layer)
memory_context, emotions, gate = memory.read(
    hidden_states,
    layer_idx=7  # Only upper layers
)

# 3. Writing to memory (after generation)
segment_states = torch.randn(5, 256)  # [N_segments, D]
current_emotions = torch.randn(5, 4)  # [N_segments, 4]

memory.write(
    segment_states=segment_states,
    emotions=current_emotions,
    surprise=0.3  # Prediction error (optional)
)

# 4. Consolidation (automatic when threshold reached)
if memory.should_consolidate():
    stats = memory.consolidate()
    print(f"Consolidated {stats['consolidated_centers']} memories")

# 5. Save/Load
memory.save("memory_state.pt")
memory = CognitiveMemorySystem.load("memory_state.pt", device="cpu")
```

### Integration into Transformer Layer

```python
import torch.nn as nn
from cognitive_memory import MemoryBlock, MemoryConfig

class TransformerBlockWithMemory(nn.Module):
    def __init__(self, d_model, config: MemoryConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = Attention(...)  # Standard self-attention
        
        # Memory Attention
        self.ln_memory = nn.LayerNorm(d_model)
        self.memory_block = MemoryBlock(config)
        
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(...)
    
    def forward(self, x, memory_state):
        # 1. Self-Attention
        y = x + self.attn(self.ln1(x))
        
        # 2. Memory Attention (with gate)
        if memory_state is not None:
            m, emotions, gate = self.memory_block(
                self.ln_memory(y), 
                memory_state
            )
            y = y + gate * m  # Gated residual
        
        # 3. Feed-Forward
        out = y + self.ffn(self.ln2(y), emotions)
        
        return out
```

---

## 📂 Folder Structure

```
cognitive_memory/
├── __init__.py                # Public API
├── config.py                  # Configuration (MemoryConfig)
├── memory_centers.py          # RBF kernel centers (LTM/STM)
├── terrain_3d.py              # 3D diffusion terrains
├── memory_block.py            # Memory Attention module
├── memory_attention.py        # Attention mechanism
├── terrain_prior.py           # TerrainPrior (3D→64D bias)
├── writer.py                  # Segment writing
├── consolidation.py           # Sleep consolidation (STM→LTM)
├── projections.py             # Projections (64D↔3D, 16D→64D)
├── persistence.py             # State saving/loading
├── README.md                  # Czech documentation
└── README_EN.md               # This documentation
```

### Module Descriptions

| Module | Function |
|--------|----------|
| `config.py` | Configuration class with coefficients for yearly operation |
| `memory_centers.py` | RBF kernel operations (read/write), homeostasis, merge/prune |
| `terrain_3d.py` | 3D grid with diffusion (Laplace), splat writing, sampling |
| `memory_block.py` | Complete Memory Attention + Gate for Transformer |
| `terrain_prior.py` | 3D→64D query shift, gate prior from terrain |
| `writer.py` | Segmentation, write strength calculation (novelty, surprise, emotion) |
| `consolidation.py` | Fatigue, STM→LTM consolidation, normalization |
| `projections.py` | Linear projections (64D→3D, 16D→64D) |
| `persistence.py` | Complete memory state save/load |

---

## 📐 Mathematical Description

### 1. Memory Reading (RBF Kernel Retrieval)

#### 1.1 TerrainPrior (3D→64D bias)

For query `q ∈ R^64`:

```
z = tanh(W_c @ q + b_c)  ∈ [-1,1]³    # Projection to 3D
p_H = sample(H³, z)  ∈ R              # Intensity sampling
p_E = sample(E³, z)  ∈ R^4            # Emotion sampling

g_prior = σ(a_H * p_H + a_E^T @ p_E + b_g)  # Gate prior

q̃ = norm(q + β_q * R([p_H; p_E]))    # Query shift
```

**Purpose:** Terrain says "you've been here before" → opens gate and shifts query

#### 1.2 RBF Kernel Reading

For each center `i`:

```
w_i = exp(-||q̃ - K_i||² / 2σ²)       # RBF kernel

π_i = softmax_i(log(ε + h_i) + log(w_i))  # Intensity strengthens weights

r_V = Σ π_i * V_i  ∈ R^d_v            # Read values
r_E = Σ π_i * e_i  ∈ R^4              # Read emotions
```

#### 1.3 Gate (final decision)

```
g = σ(W_g @ [x; r_V] + u * g_prior)   # Combine content + prior

x_out = x + g ⊙ W_m @ r_V             # Gated residual
```

### 2. Memory Writing

#### 2.1 Write Strength (adaptive)

For segment `s`:

```
ω_s = η₀ * σ(c_n * novelty + c_δ * surprise + c_a * emotion_salience + b_ω)
```

Where:
- **novelty**: `1 - max(sim(k_s, K_i))`
- **surprise**: Prediction error (entropy, KL divergence)
- **emotion_salience**: `||ε_s||` (emotion norm)

#### 2.2 RBF Writing to Centers

```
w̄_i = normalize(exp(-||k_s - K_i||² / 2σ_w²))  # RBF weights

h_i ← h_i + ω_s * w̄_i                          # Update intensity
V_i ← V_i + α_V * ω_s * w̄_i * (v_s - V_i)      # EMA values
e_i ← e_i + α_E * ω_s * w̄_i * (ε_s - e_i)      # EMA emotions
```

**Creating New Center:**
```
If max(w̄_i) < τ_new:
    K_new = k_s
    V_new = v_s
    h_new = ω_s
    e_new = ε_s
```

#### 2.3 Writing to 3D Terrain (splat)

```
z_s = tanh(W_c @ k_s)  ∈ [-1,1]³

ΔH³(u) = ω_s * exp(-||u - z_s||² / 2σ₃²)
ΔE³(u) = ω_s * exp(-||u - z_s||² / 2σ₃²) * ε_s

H³ ← H³ + η₃ * ΔH³
E³ ← E³ + η₃ * ΔE³
```

### 3. Diffusion and Homeostasis (each step)

#### 3.1 3D Terrain (foam)

Laplacian operator (6-neighbors):

```
∇²H³(i,j,k) = Σ (H³(neighbors) - H³(i,j,k))

H³ ← (1 - λ₃) * H³ + α_H * ∇²H³    # Diffusion + decay
E³ ← (1 - λ₃) * E³ + α_E * ∇²E³
```

#### 3.2 Centers (homeostasis)

```
h_i ← (1 - λ_64) * h_i                   # Intensity decay
V_i ← (1 - λ_64^V) * V_i                 # Value decay
e_i ← (1 - λ_64^E) * e_i + λ_64^E * 1    # Return to neutral (1.0)
```

### 4. Consolidation (sleep)

#### 4.1 Fatigue

```
F ← (1 - λ_F) * F + Σ ω_s^stm

If F > Θ → sleep
```

#### 4.2 STM → LTM Transfer

```
# Select top-M STM centers by h_i^s
C = TopM(h^s)

For each i ∈ C:
    k^64 = norm(U @ K_i^s)               # 16D → 64D projection
    ω_ltm = κ * h_i^s                    # Reduced strength
    
    LTM.write(k^64, V_i^s, e_i^s, ω_ltm)  # Write to LTM
```

#### 4.3 3D Terrain: STM → LTM

```
H_LTM³ ← H_LTM³ + ξ_H * blur(H_STM³)
E_LTM³ ← E_LTM³ + ξ_E * blur(E_STM³)
```

#### 4.4 STM Normalization (not deletion!)

```
h^s ← log(1 + h^s)                       # Logarithmization
V^s ← V^s / (1 + ||V^s|| / c_V)          # Saturation
e^s ← tanh(e^s)                          # Clipping

F ← ρ_F * F                              # Fatigue reset
```

### 5. Capacity Management

#### 5.1 Merging Similar Centers

```
If sim(K_i, K_j) > τ_merge:
    h_new = h_i + h_j
    K_new = norm((h_i * K_i + h_j * K_j) / h_new)
    V_new = (h_i * V_i + h_j * V_j) / h_new
    e_new = (h_i * e_i + h_j * e_j) / h_new
```

#### 5.2 Pruning Weak Centers

```
Remove center i if:
    h_i < τ_h  AND
    usage_i < τ_u  AND
    age_i > τ_age
```

---

## ⚙️ Configuration

### Coefficients for Yearly Operation

All default values are calibrated for **~50 interactions/day** = **~18,250 steps/year**.

#### LTM (64D) - Half-life ~1 year

```python
ltm_leak = 3.8e-5           # λ_64 - extremely slow decay
ltm_alpha_value = 0.03      # α_V - value update rate
ltm_alpha_emotion = 0.01    # α_E - emotion update rate
ltm_sigma_read = 0.5        # σ for RBF reading (0.3-0.7)
ltm_sigma_write = 0.15      # σ for RBF writing (AGGRESSIVE: sharp traces)
ltm_new_center_threshold = 0.8  # τ_new (AGGRESSIVE: only at >80% match)
```

**Meaning:**
- **Small leak** → memories last ~1 year
- **Small alpha** → values change slowly (stability)
- **Small sigma_write** → sharp, localized traces
- **High threshold** → creates new centers more often (granularity)

#### STM (16D) - Half-life days to weeks

```python
stm_leak = 5e-3             # λ_stm - faster decay
stm_alpha_value = 0.1       # α_V^s - faster update
stm_sigma_write = 0.2       # σ_w - sharper than LTM
```

#### 3D Terrains

```python
# LTM terrain (slow)
terrain_ltm_lambda = 5e-5   # λ_3 - decay
terrain_ltm_alpha_h = 0.002 # α_H - intensity diffusion
terrain_ltm_alpha_e = 0.001 # α_E - emotion diffusion

# STM terrain (fast)
terrain_stm_alpha_h = 0.02  # 10× faster diffusion
```

#### Consolidation

```python
fatigue_threshold = 5.0     # Θ - sleep threshold (AGGRESSIVE: 5.0)
consolidation_kappa = 0.8   # κ - intensity conversion (AGGRESSIVE: 0.8)
normalization_rho_f = 0.2   # ρ_F - fatigue reset (keeps 20%)
```

### Tuning Coefficients

#### For more granularity (more centers):

```python
config.ltm_new_center_threshold = 0.9    # Higher → more frequent new centers
config.ltm_sigma_write = 0.1             # Narrower → sharper traces
```

#### For more stable memory (fewer changes):

```python
config.ltm_alpha_value = 0.01            # Slower update
config.write_strength_base = 0.1         # Weaker writing
```

#### For faster forgetting:

```python
config.ltm_leak = 1e-4                   # Half-life ~3 months
config.terrain_ltm_lambda = 1e-4         # Faster terrain decay
```

---

## 🧪 Testing

### Validation Framework

In the `../memory-tests/` folder you'll find a complete test suite:

```bash
# Basic tests (3 sec)
python memory-tests/test_memory_fundamentals.py

# Qualitative tests (6 sec)
python memory-tests/memory_quality_test.py

# Stress test (~5 min, 9000 interactions = ~6 months)
python memory-tests/stress_test_memory.py

# Complete suite with visualizations (~10 min)
python memory-tests/run_full_memory_suite.py
```

### Available Tests

| Test | Purpose | Output |
|------|---------|--------|
| `test_memory_fundamentals.py` | Write/Read, retention, capacity | PASS/FAIL |
| `memory_quality_test.py` | Interference, consolidation | Metrics |
| `stress_test_memory.py` | Load test (9000 steps) | JSON + CSV |
| `ablation_study.py` | Ablation study (4 configurations) | Comparison |
| `visualize_*.py` | Topology, diffusion visualization | PNG graphs |

### Validated Metrics (2026-01-14)

#### Stress Test (9000 interactions)

| Metric | Value | Status |
|--------|-------|--------|
| LTM Active Centers | **459** | ✅ Excellent granularity |
| STM Active Centers | 141 | ✅ OK |
| Consolidation Events | 65 | ✅ Regular consolidation |
| h_max (intensity) | 42.0 | ✅ OK |

#### Fundamental Tests

| Test | Result |
|------|--------|
| Direct Write/Read | ✅ 100% similarity |
| Retention (1000 steps) | ✅ 100% |
| Capacity | ✅ 50 centers |
| Similarity Retrieval | ✅ 100% accuracy |

#### Memory Reconstruction

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Self-Reconstruction | **0.92** | B grade - Very good |
| Aging (Q4-Q1) | +0.20 | D grade - Old memories degrade |
| Self-Weight | 0.38 | Retrieval is "soft" (RBF) |

---

## 📚 Citation

If you use Cognitive Memory in your work, please cite:

```bibtex
@software{cognitive_memory_2026,
  author = {Seidl, Michal},
  title = {Cognitive Memory: Biologically Inspired Persistent Memory System for LLMs},
  year = {2026},
  publisher = {OpenTechLab Jablonec nad Nisou},
  version = {2.0-beta},
  url = {https://github.com/OpenTechLab/cognitive-memory}
}
```

### Theoretical Foundation

The system is based on these concepts:

- **Atkinson-Shiffrin model** (1968): STM/LTM separation
- **Complementary Learning Systems** (McClelland, McNaughton, O'Reilly, 1995): Hippocampus-neocortex interaction
- **Memory consolidation** (Sleep-wake cycle)
- **RBF networks** (Radial Basis Function kernels)
- **Reaction-diffusion systems** (Turing, 1952): Spatial diffusion

---

## 🛠️ Advanced Usage

### Custom Projections

```python
from cognitive_memory.projections import TerrainProjection

# Custom 64D → 3D projection (e.g., autoencoder)
class CustomProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.encoder(x)

# Usage
config.terrain_projection = CustomProjection()
```

### State Monitoring

```python
# Get detailed statistics
ltm_stats = memory.ltm_centers.get_stats()
print(f"LTM centers: {ltm_stats['n_active']}")
print(f"Average intensity: {ltm_stats['h_mean']:.3f}")
print(f"Max intensity: {ltm_stats['h_max']:.3f}")

# Terrain statistics
terrain_stats = memory.ltm_terrain.get_stats()
print(f"Total energy: {terrain_stats['total_energy']:.3f}")

# Fatigue level
fatigue = memory.consolidator.get_fatigue_level()
print(f"Fatigue: {fatigue * 100:.1f}%")
```

### Export Visualizations

```python
# Export 3D terrain as numpy array
terrain_h = memory.ltm_terrain.H.cpu().numpy()  # [48, 48, 48]
terrain_e = memory.ltm_terrain.E.cpu().numpy()  # [48, 48, 48, 4]

# Use matplotlib/plotly for visualization
import matplotlib.pyplot as plt
plt.imshow(terrain_h[:, :, 24], cmap='viridis')
plt.title("LTM Terrain - Central Slice")
plt.colorbar()
plt.savefig("terrain_slice.png")
```

---

## 🐛 Troubleshooting

| Problem | Possible Cause | Solution |
|---------|----------------|----------|
| Memory Error during initialization | Too large 3D grid | Reduce `terrain_resolution` to 32 or 24 |
| LTM fills up quickly | Too aggressive center creation | Reduce `ltm_new_center_threshold` to 0.5 |
| No consolidations | Low STM activity | Reduce `fatigue_threshold` to 3.0 |
| Reading too slow | Large `ltm_top_k_read` | Reduce to 16 or 8 |
| Gate always closed | Too negative `gate_bias` | Increase to -1.5 |
| Values "explode" | Too strong writing | Reduce `write_strength_base` to 0.1 |

### Debug Mode

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check for NaN
assert not torch.isnan(memory.ltm_centers.h).any(), "NaN in intensities!"
assert not torch.isnan(memory.ltm_terrain.H).any(), "NaN in terrain!"
```

---

## 📖 Additional Documentation

- **`BioCortexAI_Documentation_EN.md`**: Complete scientific documentation
- **`plan.md`**: Original mathematical design (developer notes)
- **`memory-tests/README.md`**: Testing framework and benchmarks

---

## 🤝 Support

### Issues

If you encounter a problem:
1. Verify you're using the latest version
2. Check [Troubleshooting](#-troubleshooting)
3. Open an issue with:
   - Python and PyTorch versions
   - Minimal reproducible example
   - Error message

### Contact

- **Email**: vyvoj@opentechlab.cz
- **Web**: [www.opentechlab.cz](https://www.opentechlab.cz)
- **Project**: BioCortexAI

---

## 📄 License

**CC BY-NC 4.0** (Creative Commons Attribution-NonCommercial 4.0)

✅ **Allowed:**
- Use for research and education
- Modification and distribution (with attribution)
- Private experiments

❌ **Prohibited:**
- Commercial use without license
- Patent claims

For commercial licensing, contact: opentechlab@opentechlab.cz

---

**Framework:** BioCortexAI v2.0-beta  
**Author:** Michal Seidl, OpenTechLab Jablonec nad Nisou s.r.o.  
**Status:** ✅ Production-ready

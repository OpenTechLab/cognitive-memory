# 🧪 Cognitive Memory - Testing & Validation Framework

Comprehensive toolset for validation, analysis, and visualization of the Cognitive Memory system.

> **Last updated:** 2026-01-14  
> **Status:** ✅ Fully functional - 454 LTM centers, 100% tests passing

---

## 📋 Folder Contents

### 🔬 Main Testing Scripts

| Script | Description | Runtime |
|--------|-------------|---------|
| `run_full_memory_suite.py` | **Orchestration script** - runs everything at once | ~5-10 min |
| `stress_test_memory.py` | Full stress test (9000 interactions) | ~5 min |
| `ablation_study.py` | **Ablation study** for paper (4 configurations) | ~10 min |
| `ablation_retrieval_test.py` | **Retrieval benchmark** for TerrainPrior | ~2 min |
| `test_memory_fundamentals.py` | Basic unit tests (R/W, retention, capacity) | ~3 sec |
| `memory_quality_test.py` | Qualitative tests (interference, consolidation) | ~6 sec |
| `audit_memory_content.py` | Content audit - memory discrimination | ~4 sec |
| `test_memory_reconstruction.py` | Reconstruction analysis by age | ~10 sec |

### 📊 Visualization Scripts

| Script | Description | Outputs |
|--------|-------------|---------|
| `visualize_stress_test.py` | Metric charts from stress test | `stress_test_results/visualizations/` |
| `visualize_centers_structure.py` | PCA projection + 3D LTM centers landscape | `terrain_visualizations/` |
| `visualize_memory_topology.py` | Clustermap, Dendrogram, t-SNE | `terrain_visualizations/` |

### 📁 Helper Files

| File | Description |
|------|-------------|
| `realistic_scenarios.py` | Ultra-realistic scenario with 500 topics |
| `analyze_memory_risks.py` | Detection of 5 known risks |

---

## 🚀 Quick Start

### Running the Complete Test Suite
```powershell
python run_full_memory_suite.py
```

This script executes:
1. Fundamental tests
2. Qualitative tests
3. Content audit
4. Stress test (if results don't exist)
5. All visualizations

### Manual Execution
```powershell
# Stress test
python stress_test_memory.py

# Visualizations
python visualize_stress_test.py
python visualize_centers_structure.py
python visualize_memory_topology.py

# Reconstruction test
python test_memory_reconstruction.py
```

---

## 📈 Test Results (2026-01-14)

### Stress Test (9000 interactions = ~6 months of operation)

| Metric | Value | Status |
|--------|-------|--------|
| LTM Active Centers | **459** | ✅ Excellent granularity |
| STM Active Centers | 141 | ✅ OK |
| Consolidation Events | 65 | ✅ Regular consolidation |
| h_max (intensity) | 42.0 | ✅ OK |
| h_mean | 11.5 | ✅ OK |

### Fundamental Tests

| Test | Result |
|------|--------|
| Direct Write/Read | ✅ 100% similarity |
| Retention (1000 steps) | ✅ 100% |
| Capacity | ✅ 50 centers |
| Similarity Retrieval | ✅ 100% accuracy |

### Qualitative Tests

| Test | Result |
|------|--------|
| Retention Rate | ✅ 100% |
| Anti-Interference | ✅ 100% |
| Capacity Score | ✅ 100% |
| Consolidation Survival | ✅ 100% |

### Memory Reconstruction

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Self-Reconstruction | **0.92** | B grade - Very good |
| Aging (Q4-Q1) | +0.20 | D grade - Old memories degrade |
| Self-Weight | 0.38 | Retrieval is "soft" (RBF) |

---

## 📊 Generated Visualizations

### `terrain_visualizations/`

| File | Description |
|------|-------------|
| `centers_structure_scatter.png` | 3D scatter plot of LTM centers (PCA) |
| `centers_ideal_map.png` | 2D heatmap of semantic space |
| `centers_ideal_3d_landscape.png` | 3D memory landscape |
| `topology_01_clustermap.png` | Similarity matrix (454×454) |
| `topology_02_dendrogram.png` | Hierarchical memory tree |
| `topology_03_tsne.png` | t-SNE manifold |
| `memory_reconstruction_analysis.png` | Reconstruction analysis |
| `memory_discrimination_audit.png` | Discrimination audit |

### `stress_test_results/visualizations/`

| File | Description |
|------|-------------|
| `01_overview.png` | Dashboard (6 panels) |
| `02_retention.png` | Retention analysis |
| `03_consolidation.png` | Consolidation timeline |

---

## ⚙️ Current Configuration (`cognitive_memory/config.py`)

### LTM Parameters (AGGRESSIVE PLASTICITY)
```python
ltm_leak: float = 3.8e-5           # Half-life ~1 year
ltm_sigma_read: float = 0.5        # RBF width for reading
ltm_sigma_write: float = 0.15      # RBF width for writing (sharp)
ltm_new_center_threshold: float = 0.8  # Threshold for new center
```

### Write
```python
write_strength_base: float = 0.2   # Base write strength
segment_size: int = 32             # Tokens per segment
```

### Consolidation
```python
fatigue_threshold: float = 5.0     # Threshold fatigue → sleep
consolidation_kappa: float = 0.8   # Intensity conversion
# Consolidation uses threshold 0.5 (conservative)
```

---

## 🔧 Production Recommendations

### 1. Monitor h_max
Some centers have intensity >140. Consider:
- Reduce `write_strength_base` to 0.15
- Implement log-scaling during retrieval

### 2. Memory Aging
Q1 (oldest) centers have reconstruction 0.75 vs Q4 (fresh) 0.95.
- Reduce `ltm_leak_value` for slower V value degradation
- Or accept as a feature (old memories naturally fade)

### 3. Self-Weight
Average weight on correct center is 38%. For sharper retrieval:
- Reduce `ltm_sigma_read` to 0.3-0.4

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Memory Error | Reduce `terrain_resolution` to 24 |
| Too slow | Reduce `n_ltm_centers` |
| LTM fills up quickly | Reduce `ltm_new_center_threshold` |
| No consolidations | Reduce `fatigue_threshold` |

---

## 📚 References

- `../cognitive_memory/config.py` - Main configuration
- `../cognitive_memory/memory_centers.py` - RBF centers
- `../cognitive_memory/consolidation.py` - Sleep consolidation

---

**Framework:** BioCortexAI v2.0-beta  
**Status:** ✅ Fully functional and validated

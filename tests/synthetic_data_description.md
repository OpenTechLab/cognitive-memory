# Description of Synthetic Data for Stress Test

This document provides a detailed description of how synthetic data is generated and used in the Cognitive Memory system stress test.

> **Version:** 2026-01-15  
> **Files:** `stress_test_memory.py`, `realistic_scenarios.py`

---

## Purpose of Synthetic Data

Synthetic data simulates **realistic annual operation of an LLM** without the need for actual user data. The goal is to:

1. **Validate memory stability** during long-term operation
2. **Test homeostasis mechanisms** (forgetting, consolidation)
3. **Measure capacity and interference** of memory centers
4. **Create reproducible benchmarks**

---

## Data Generation Architecture

### Scenario Hierarchy

```
StressTestScenario (abstract class)
├── RandomScenario      → Random embeddings (worst case)
├── ClusteredScenario   → Thematically clustered
├── TemporalScenario    → Time-structured (narrative)
└── RealisticMixedScenario → Ultra-realistic (500 topics)
```

---

## Scenario Types

### 1. RandomScenario (Worst Case)

**File:** `stress_test_memory.py`, lines 88-111

Generates **purely random embeddings** without any structure.

| Parameter | Value | Description |
|-----------|-------|-------------|
| Embedding | `torch.randn(d_model)` | Normalized random vector |
| Emotion | `torch.rand(4) + 0.5` | Uniform [0.5, 1.5] |
| Surprise | `uniform(0, 1)` | Random surprise |
| Cluster ID | `-1` | Unassigned |

**Purpose:** Tests maximum interference – what happens when memory receives completely unrelated inputs.

---

### 2. ClusteredScenario (Thematically clustered)

**File:** `stress_test_memory.py`, lines 114-157

Generates embeddings **grouped around N thematic centers**.

| Parameter | Value | Description |
|-----------|-------|-------------|
| Cluster Count | 5 (default) | `cluster_centers` parameter |
| Noise around center | σ = 0.3 | Gaussian noise |
| Emotion | Correlates with cluster | Systematic differences |
| Surprise | `uniform(0, 0.5)` | Low (routine) |

**Embedding Generation:**
```python
center = self.centers[cluster_id]  # Random center
noise = torch.randn(d_model) * 0.3
emb = center + noise
emb = emb / emb.norm()  # L2 normalization
```

**Purpose:** Simulates a user who focuses on several main topics.

---

### 3. TemporalScenario (Time-structured)

**File:** `stress_test_memory.py`, lines 160-198

Generates embeddings with **slow drift** – simulates narrative development.

| Parameter | Value | Description |
|-----------|-------|-------------|
| Drift | σ = 0.1 | Small change from previous |
| Emotion | Sinusoidal | Changes periodically |
| Surprise | `0.2 + 0.3 * |sin(phase)|` | Oscillates |

**Embedding Generation:**
```python
drift = torch.randn(d_model) * 0.1
self.current = self.current + drift
self.current = self.current / self.current.norm()
```

**Purpose:** Simulates the gradual development of a conversation where each step follows the previous one.

---

### 4. RealisticMixedScenario (Ultra-realistic)

**File:** `realistic_scenarios.py`, lines 30-321

The most sophisticated scenario simulating **actual LLM operation**.

---

## Detail: RealisticMixedScenario

### Basic Calibration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Topics | **500** | High user variability |
| Interactions/day | **50** | Average daily chatbot usage |
| Tokens/interaction | **50-200** | Realistic prompt + response |
| Topics/day | **~5** | 10% chance of new topic |
| Aha moment | **every ~18 interactions** | Poisson distribution |

### Generation Structure

#### A) Topic Selection

```python
# Zipf distribution of topic popularity
ranks = np.arange(1, 500 + 1)
topic_popularity = 1.0 / ranks  # Some topics are more frequent
```

**Selection Logic:**
1. **10% chance** → New topic (weighted by popularity)
2. **90% chance** → Continue current topic
3. **10% of continuation** → Switch between today's topics

#### B) Token Sequence Generation

Each interaction generates a **sequence of 50-200 tokens** (not a single embedding!):

```python
n_tokens = int(np.random.uniform(
    tokens_per_interaction * 0.5,  # 50
    tokens_per_interaction * 1.5   # 150
))
```

**Token-by-token generation:**
```python
for t in range(n_tokens):
    # Small drift (coherence within prompt)
    drift = torch.randn(d_model) * 0.15
    current_pos = current_pos + drift
    current_pos = current_pos / current_pos.norm()
    
    # Variability around topic
    noise = torch.randn(d_model) * 0.3
    token_emb = current_pos + noise
    token_emb = token_emb / token_emb.norm()
    
    sequence[t] = token_emb
```

#### C) Surprise

**Bimodal distribution** – most interactions are routine:

| Type | Probability | Distribution | Mean Value |
|------|-------------|--------------|------------|
| Routine | 80% | Beta(2, 5) | ~0.2 |
| Novelty | 20% | Beta(5, 2) | ~0.7 |

```python
if np.random.random() < 0.80:
    surprise = float(np.random.beta(2, 5))  # Low
else:
    surprise = float(np.random.beta(5, 2))  # High
```

#### D) Aha Moments (Significant events)

**Poisson model** with an average interval of 18 interactions:

```python
next_aha_at = interaction_count + max(1, int(np.random.exponential(18)))
```

During an Aha moment:
- **Surprise** is forced high: `Beta(6, 2)` → ~0.75
- **Emotions** are more intense (see below)

#### E) Emotional Profile

**4-dimensional emotion vector** (inspired by neurotransmitters):

| Index | Name | Description |
|-------|------|-------------|
| 0 | Dopamine | Reward, motivation |
| 1 | Serotonin | Satisfaction, happiness |
| 2 | Cortisol | Stress, tension |
| 3 | Oxytocin | Trust, social bonds |

**Valence (emotional charge):**
- 45% positive
- 45% negative  
- 10% neutral

**During Aha moment:**
```python
intensity = np.random.uniform(1.2, 2.5)

if valence == 'positive':
    base[0] += intensity * 0.8-1.2  # Dopamine ↑
    base[1] += intensity * 0.5-0.9  # Serotonin ↑
    base[3] += intensity * 0.6-1.0  # Oxytocin ↑
elif valence == 'negative':
    base[2] += intensity * 0.8-1.5  # Cortisol ↑
    base[0] -= intensity * 0.2       # Dopamine ↓
```

**During normal interaction:**
```python
intensity = np.random.uniform(0.1, 0.6)  # Milder
```

---

## Data Usage in Test

### Processing Flow

```
┌──────────────────────────────────────────────────────────────┐
│  StressTestRunner.run_scenario()                             │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  for interaction in range(9000):                             │
│      │                                                       │
│      ▼                                                       │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ scenario.generate_sequence(interaction)              │    │
│  │ → sequence [T, D], metadata                          │    │
│  └─────────────────────────────────────────────────────┘    │
│      │                                                       │
│      ▼                                                       │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ writer.write_to_memory(                              │    │
│  │     hidden_states = sequence,                        │    │
│  │     emotions = metadata["emotions"],                 │    │
│  │     surprise = metadata["surprise"],                 │    │
│  │     ltm_centers, stm_centers,                        │    │
│  │     ltm_terrain, stm_terrain                         │    │
│  │ )                                                    │    │
│  └─────────────────────────────────────────────────────┘    │
│      │                                                       │
│      ▼                                                       │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ HOMEOSTASIS (once per INTERACTION, not token!)       │    │
│  │ - ltm_centers.homeostasis_step()                     │    │
│  │ - stm_centers.homeostasis_step()                     │    │
│  │ - ltm_terrain.step()                                 │    │
│  │ - stm_terrain.step()                                 │    │
│  └─────────────────────────────────────────────────────┘    │
│      │                                                       │
│      ▼                                                       │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ CONSOLIDATION (automatic upon fatigue)               │    │
│  │ - consolidator.step(omega, stm, ltm, ...)            │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### Metadata Written to CSV

For each interaction, the following metrics are recorded:

| Column | Source | Description |
|--------|--------|-------------|
| `interaction` | Index | Interaction sequence number (0-8999) |
| `n_tokens` | Scenario | Number of tokens in interaction |
| `cluster_id` | Scenario | Topic ID (0-499) |
| `surprise` | Scenario | Surprise level (0-1) |
| `omega_mean` | Writer | Average write strength |
| `omega_max` | Writer | Maximum write strength |
| `new_ltm_centers` | Writer | Number of newly created LTM centers |
| `new_stm_centers` | Writer | Number of newly created STM centers |
| `n_active` | LTM | Number of active LTM centers |
| `stm_active` | STM | Number of active STM centers |
| `H_mean` | Terrain | Average terrain height |
| `H_max` | Terrain | Maximum terrain height |
| `fatigue` | Consolidator | System fatigue level |
| `is_aha_moment` | Scenario | True if Aha moment |

---

## Mathematical Properties of Data

### Embedding Distribution

| Scenario | Distribution in Embedding Space |
|----------|---------------------------------|
| Random | Uniform on unit sphere |
| Clustered | Gaussian around N centers (σ=0.3) |
| Temporal | Random walk (drift σ=0.1) |
| Realistic | Hierarchical: Zipf topics × Gaussian |

### Normalization

**All embeddings are L2 normalized:**
```python
emb = emb / emb.norm()
```

This ensures:
- All vectors lie on the unit sphere
- Cosine similarity = dot product
- Stable RBF kernel activation

### Time Structure (RealisticMixed)

```
Day 0:  [T1, T1, T1, T2, T1, T1, T3, T3, ...]  (50 interactions)
        └─────────────────────────────────┘
         ~5 different topics, switch 10%

Day 1:  [T4, T4, T1, T4, T4, T5, T5, ...]
        └─────────────────────────────────┘
         Reset topic_today, new combo
```

---

## Test Configuration

### Production Parameters (main())

```python
config = MemoryConfig(
    d_model=256,                    # Embedding dimension
    terrain_resolution=32,          # 32³ grid
    n_ltm_centers=2048,             # Max LTM centers
    n_stm_centers=512,              # Max STM centers
)

n_interactions = 9000               # ~6 months of operation
tokens_per_interaction = 100        # ~100 tokens/interaction
save_interval = 500                 # Snapshot every 500 interactions
```

### Conversion to Real Time

| Metric | Value | Calculation |
|--------|-------|-------------|
| Days | 180 | 9000 / 50 |
| Months | 6 | 180 / 30 |
| Years | 0.49 | 180 / 365 |
| Total tokens | ~900,000 | 9000 × 100 |

---

## Validation Metrics

### What the Test Measures

1. **Retention** – How memory holds memories over time
2. **Interference** – Whether unrelated topics overlap
3. **Capacity** – How many active centers are created
4. **Stability** – Whether the system degrades
5. **Consolidation** – Transfer STM → LTM

### Expected Results (benchmark)

| Metric | Expected | Current (2026-01-15) |
|--------|----------|----------------------|
| LTM Active | 300-600 | 459 |
| h_max | < 200 | 42.0 |
| Consolidation events | > 50 | 65 |
| Retention rate | > 90% | 100% |

---

## References

- `stress_test_memory.py` – Main test runner
- `realistic_scenarios.py` – Realistic data generator
- `../cognitive_memory/config.py` – System configuration
- `README.md` – Test framework overview

---

**Framework:** BioCortexAI v2.0-beta  
**Documentation Created:** 2026-01-15

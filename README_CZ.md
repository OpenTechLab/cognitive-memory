# 🧠 Cognitive Memory - Persistentní paměť pro LLM

Biologicky inspirovaný systém dlouhodobé paměti pro velké jazykové modely. Implementuje dvouvrstvou paměťovou architekturu (STM/LTM) s 3D difuzními terény a RBF kernel operacemi.

> **Verze:** 2.0-beta  
> **Status:** ✅ Plně funkční a validováno  
> **Původ:** BioCortexAI Framework  
> **Licence:** CC BY-NC 4.0

---

## 📖 Obsah

- [Koncept a Inspirace](#-koncept-a-inspirace)
- [Architektura](#-architektura)
- [Instalace](#-instalace)
- [Rychlý start](#-rychlý-start)
- [Struktura složky](#-struktura-složky)
- [Matematický popis](#-matematický-popis)
- [Konfigurace](#-konfigurace)
- [Testování](#-testování)
- [Citace](#-citace)
- [Podpora](#-podpora)

---

## 💡 Koncept a Inspirace

### Biologická analogie

Cognitive Memory je inspirována biologickým paměťovým systémem:

| Biologická struktura | Analogie v systému | Funkce |
|---------------------|-------------------|---------|
| **Hippocampus** | STM (16D) | Rychlé kódování, omezená kapacita |
| **Neokortex** | LTM (64D) | Pomalá konsolidace, dlouhodobá retence |
| **Spánek (SWS)** | Konsolidace | Přenos vzpomínek ze STM do LTM |

**Klíčové vlastnosti:**
- **Difuze**: Stopy se v čase rozptylují (Laplaceův operátor)
- **Homeostáza**: Pomalý návrat k rovnovážnému stavu  
- **Zlomy**: Opakovaná aktivita vytváří "koleje", které jsou citlivější k opětovné aktivaci
- **Emoční zbarvení**: 4D emoční vektor (dopamin, serotonin, kortizol, oxytocin)

---

## 🏗️ Architektura

### Dvouvrstvý systém

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

### Datové struktury

#### Paměťová centra (RBF Kernel-Poles)

| Komponenta | Rozměr | Popis |
|-----------|--------|-------|
| **K** | R^64 (LTM) / R^16 (STM) | Normalizovaný klíč (sémantická pozice) |
| **V** | R^128 | Hodnota (co vybavit) |
| **h** | R^+ | Intenzita/hloubka stopy (GS) |
| **e** | R^4 | Emoční vektor (dopamin, serotonin, kortizol, oxytocin) |
| **usage** | Z^+ | Čítač použití (pro pruning) |
| **age** | Z^+ | Stáří centra |

#### 3D terény

```
Rozlišení: 48 × 48 × 48 voxelů
H³ ∈ R^(48×48×48)       # Intenzita (GS/"pěna")
E³ ∈ R^(48×48×48×4)     # Emoční stopa (4 hormony)
```

**Fyzika terénu:**
- Difuze: Laplaceův operátor (6-sousedů)
- Homeostáza: Exponenciální decay
- Projekce: 64D → 3D přes lineární vrstvu + tanh

---

## 📦 Instalace

### Závislosti

```bash
pip install torch>=2.0.0
pip install numpy>=1.20.0
pip install scipy>=1.7.0
```

### Integrace do projektu

```python
from cognitive_memory import CognitiveMemorySystem, MemoryConfig

# Vytvoř konfiguraci
config = MemoryConfig(
    d_model=256,              # Dimenze transformeru
    n_ltm_centers=1024,       # Počet LTM center
    ltm_leak=3.8e-5,          # Leak pro roční provoz
)

# Inicializuj systém
memory = CognitiveMemorySystem(config, device="cuda")
```

---

## 🚀 Rychlý start

### Základní použití

```python
import torch
from cognitive_memory import CognitiveMemorySystem, MemoryConfig

# 1. Vytvoř paměťový systém
config = MemoryConfig()
memory = CognitiveMemorySystem(config, device="cpu")

# 2. Čtení z paměti (během inference)
hidden_states = torch.randn(1, 32, 256)  # [B, T, D]

# Memory Attention (pro Transformer vrstvu)
memory_context, emotions, gate = memory.read(
    hidden_states,
    layer_idx=7  # Jen horní vrstvy
)

# 3. Zápis do paměti (po generování)
segment_states = torch.randn(5, 256)  # [N_segments, D]
current_emotions = torch.randn(5, 4)  # [N_segments, 4]

memory.write(
    segment_states=segment_states,
    emotions=current_emotions,
    surprise=0.3  # Predikční chyba (volitelně)
)

# 4. Konsolidace (automatická při dosažení prahu)
if memory.should_consolidate():
    stats = memory.consolidate()
    print(f"Konsolidováno {stats['consolidated_centers']} vzpomínek")

# 5. Uložení/načtení
memory.save("memory_state.pt")
memory = CognitiveMemorySystem.load("memory_state.pt", device="cpu")
```

### Integrace do Transformer vrstvy

```python
import torch.nn as nn
from cognitive_memory import MemoryBlock, MemoryConfig

class TransformerBlockWithMemory(nn.Module):
    def __init__(self, d_model, config: MemoryConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = Attention(...)  # Standardní self-attention
        
        # Memory Attention
        self.ln_memory = nn.LayerNorm(d_model)
        self.memory_block = MemoryBlock(config)
        
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(...)
    
    def forward(self, x, memory_state):
        # 1. Self-Attention
        y = x + self.attn(self.ln1(x))
        
        # 2. Memory Attention (s gate)
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

## 📂 Struktura složky

```
cognitive_memory/
├── __init__.py                # Public API
├── config.py                  # Konfigurace (MemoryConfig)
├── memory_centers.py          # RBF kernel centra (LTM/STM)
├── terrain_3d.py              # 3D difuzní terény
├── memory_block.py            # Memory Attention modul
├── memory_attention.py        # Attention mechanismus
├── terrain_prior.py           # TerrainPrior (3D→64D bias)
├── writer.py                  # Zápis segmentů
├── consolidation.py           # Sleep konsolidace (STM→LTM)
├── projections.py             # Projekce (64D↔3D, 16D→64D)
├── persistence.py             # Ukládání/načítání stavu
└── README.md                  # Tato dokumentace
```

### Popis modulů

| Modul | Funkce |
|-------|--------|
| `config.py` | Konfigurační třída s koeficienty pro roční provoz |
| `memory_centers.py` | RBF kernel operace (čtení/zápis), homeostáza, merge/prune |
| `terrain_3d.py` | 3D grid s difuzí (Laplace), splat zápis, sampling |
| `memory_block.py` | Kompletní Memory Attention + Gate pro Transformer |
| `terrain_prior.py` | 3D→64D posun dotazu, gate prior z terénu |
| `writer.py` | Segmentace, výpočet síly zápisu (novost, surprise, emoce) |
| `consolidation.py` | Únava, konsolidace STM→LTM, normalizace |
| `projections.py` | Lineární projekce (64D→3D, 16D→64D) |
| `persistence.py` | Uložení/načtení kompletního stavu paměti |

---

## 📐 Matematický popis

### 1. Čtení z paměti (RBF Kernel Retrieval)

#### 1.1 TerrainPrior (3D→64D bias)

Pro dotaz `q ∈ R^64`:

```
z = tanh(W_c @ q + b_c)  ∈ [-1,1]³    # Projekce do 3D
p_H = sample(H³, z)  ∈ R              # Sampling intenzity
p_E = sample(E³, z)  ∈ R^4            # Sampling emocí

g_prior = σ(a_H * p_H + a_E^T @ p_E + b_g)  # Gate prior

q̃ = norm(q + β_q * R([p_H; p_E]))    # Posun dotazu
```

**Účel:** Terén říká "tady jsi už byl" → otevře gate a posune dotaz

#### 1.2 RBF Kernel čtení

Pro každé centrum `i`:

```
w_i = exp(-||q̃ - K_i||² / 2σ²)       # RBF kernel

π_i = softmax_i(log(ε + h_i) + log(w_i))  # Intenzita posiluje váhy

r_V = Σ π_i * V_i  ∈ R^d_v            # Čtené hodnoty
r_E = Σ π_i * e_i  ∈ R^4              # Čtené emoce
```

#### 1.3 Gate (finální rozhodnutí)

```
g = σ(W_g @ [x; r_V] + u * g_prior)   # Kombinace obsahu + prior

x_out = x + g ⊙ W_m @ r_V             # Gated residual
```

### 2. Zápis do paměti

#### 2.1 Síla zápisu (adaptivní)

Pro segment `s`:

```
ω_s = η₀ * σ(c_n * novelty + c_δ * surprise + c_a * emotion_salience + b_ω)
```

Kde:
- **novelty**: `1 - max(sim(k_s, K_i))`
- **surprise**: Predikční chyba (entropie, KL divergence)
- **emotion_salience**: `||ε_s||` (norma emocí)

#### 2.2 RBF zápis do center

```
w̄_i = normalize(exp(-||k_s - K_i||² / 2σ_w²))  # RBF váhy

h_i ← h_i + ω_s * w̄_i                          # Update intenzity
V_i ← V_i + α_V * ω_s * w̄_i * (v_s - V_i)      # EMA hodnot
e_i ← e_i + α_E * ω_s * w̄_i * (ε_s - e_i)      # EMA emocí
```

**Přidání nového centra:**
```
Pokud max(w̄_i) < τ_new:
    K_new = k_s
    V_new = v_s
    h_new = ω_s
    e_new = ε_s
```

#### 2.3 Zápis do 3D terénu (splat)

```
z_s = tanh(W_c @ k_s)  ∈ [-1,1]³

ΔH³(u) = ω_s * exp(-||u - z_s||² / 2σ₃²)
ΔE³(u) = ω_s * exp(-||u - z_s||² / 2σ₃²) * ε_s

H³ ← H³ + η₃ * ΔH³
E³ ← E³ + η₃ * ΔE³
```

### 3. Difuze a homeostáza (každý krok)

#### 3.1 3D terén (pěna)

Laplaceův operátor (6-sousedů):

```
∇²H³(i,j,k) = Σ (H³(neighbors) - H³(i,j,k))

H³ ← (1 - λ₃) * H³ + α_H * ∇²H³    # Difuze + decay
E³ ← (1 - λ₃) * E³ + α_E * ∇²E³
```

#### 3.2 Centra (homeostáza)

```
h_i ← (1 - λ_64) * h_i                   # Decay intenzity
V_i ← (1 - λ_64^V) * V_i                 # Decay hodnot
e_i ← (1 - λ_64^E) * e_i + λ_64^E * 1    # Návrat k neutrální (1.0)
```

### 4. Konsolidace (spánek)

#### 4.1 Únava

```
F ← (1 - λ_F) * F + Σ ω_s^stm

Pokud F > Θ → spánek
```

#### 4.2 Přenos STM → LTM

```
# Vyber top-M STM center podle h_i^s
C = TopM(h^s)

Pro každé i ∈ C:
    k^64 = norm(U @ K_i^s)               # 16D → 64D projekce
    ω_ltm = κ * h_i^s                    # Snížená síla
    
    LTM.write(k^64, V_i^s, e_i^s, ω_ltm)  # Zápis do LTM
```

#### 4.3 3D terén: STM → LTM

```
H_LTM³ ← H_LTM³ + ξ_H * blur(H_STM³)
E_LTM³ ← E_LTM³ + ξ_E * blur(E_STM³)
```

#### 4.4 Normalizace STM (ne vymazání!)

```
h^s ← log(1 + h^s)                       # Logaritmizace
V^s ← V^s / (1 + ||V^s|| / c_V)          # Saturace
e^s ← tanh(e^s)                          # Clipping

F ← ρ_F * F                              # Reset únavy
```

### 5. Správa kapacity

#### 5.1 Merge podobných center

```
Pokud sim(K_i, K_j) > τ_merge:
    h_new = h_i + h_j
    K_new = norm((h_i * K_i + h_j * K_j) / h_new)
    V_new = (h_i * V_i + h_j * V_j) / h_new
    e_new = (h_i * e_i + h_j * e_j) / h_new
```

#### 5.2 Prune slabých center

```
Odstranit centrum i pokud:
    h_i < τ_h  AND
    usage_i < τ_u  AND
    age_i > τ_age
```

---

## ⚙️ Konfigurace

### Koeficienty pro roční provoz

Všechny defaultní hodnoty jsou kalibrovány pro **~50 interakcí/den** = **~18 250 kroků/rok**.

#### LTM (64D) - Poločas ~1 rok

```python
ltm_leak = 3.8e-5           # λ_64 - extrémně pomalý decay
ltm_alpha_value = 0.03      # α_V - update rychlost hodnot
ltm_alpha_emotion = 0.01    # α_E - update rychlost emocí
ltm_sigma_read = 0.5        # σ pro RBF čtení (0.3-0.7)
ltm_sigma_write = 0.15      # σ pro RBF zápis (AGRESIVNÍ: ostré stopy)
ltm_new_center_threshold = 0.8  # τ_new (AGRESIVNÍ: jen při >80% shodě)
```

**Význam:**
- **Malý leak** → vzpomínky vydrží ~1 rok
- **Malý alpha** → hodnoty se mění pomalu (stabilita)
- **Malý sigma_write** → ostré, lokalizované stopy
- **Vysoký threshold** → častěji vytváří nová centra (granularita)

#### STM (16D) - Poločas dny až týdny

```python
stm_leak = 5e-3             # λ_stm - rychlejší decay
stm_alpha_value = 0.1       # α_V^s - rychlejší update
stm_sigma_write = 0.2       # σ_w - ostřejší než LTM
```

#### 3D terény

```python
# LTM terén (pomalý)
terrain_ltm_lambda = 5e-5   # λ_3 - decay
terrain_ltm_alpha_h = 0.002 # α_H - difuze intenzity
terrain_ltm_alpha_e = 0.001 # α_E - difuze emocí

# STM terén (rychlý)
terrain_stm_alpha_h = 0.02  # 10× rychlejší difuze
```

#### Konsolidace

```python
fatigue_threshold = 5.0     # Θ - práh pro spánek (AGRESIVNÍ: 5.0)
consolidation_kappa = 0.8   # κ - přepočet intenzity (AGRESIVNÍ: 0.8)
normalization_rho_f = 0.2   # ρ_F - reset únavy (zachová 20%)
```

### Ladění koeficientů

#### Pro více granularity (více center):

```python
config.ltm_new_center_threshold = 0.9    # Vyšší → častější nová centra
config.ltm_sigma_write = 0.1             # Užší → ostřejší stopy
```

#### Pro stabilnější paměť (méně změn):

```python
config.ltm_alpha_value = 0.01            # Pomalejší update
config.write_strength_base = 0.1         # Slabší zápis
```

#### Pro rychlejší zapomínání:

```python
config.ltm_leak = 1e-4                   # Poločas ~3 měsíce
config.terrain_ltm_lambda = 1e-4         # Rychlejší decay terénu
```

---

## 🧪 Testování

### Validační framework

V složce `../memory-tests/` najdeš kompletní test suite:

```bash
# Základní testy (3 sec)
python memory-tests/test_memory_fundamentals.py

# Kvalitativní testy (6 sec)
python memory-tests/memory_quality_test.py

# Stress test (~5 min, 9000 interakcí = ~6 měsíců)
python memory-tests/stress_test_memory.py

# Kompletní suite s vizualizacemi (~10 min)
python memory-tests/run_full_memory_suite.py
```

### Dostupné testy

| Test | Účel | Výstup |
|------|------|--------|
| `test_memory_fundamentals.py` | Write/Read, retention, kapacita | PASS/FAIL |
| `memory_quality_test.py` | Interference, konsolidace | Metriky |
| `stress_test_memory.py` | Zátěžový test (9000 kroků) | JSON + CSV |
| `ablation_study.py` | Ablační studie (4 konfigurace) | Srovnání |
| `visualize_*.py` | Vizualizace topologie, difuze | PNG grafy |

### Validované metriky (2026-01-14)

#### Stress Test (9000 interakcí)

| Metrika | Hodnota | Status |
|---------|---------|--------|
| LTM Active Centers | **459** | ✅ Vynikající granularita |
| STM Active Centers | 141 | ✅ OK |
| Consolidation Events | 65 | ✅ Pravidelná konsolidace |
| h_max (intenzita) | 42.0 | ✅ OK |

#### Fundamentální testy

| Test | Výsledek |
|------|----------|
| Direct Write/Read | ✅ 100% similarity |
| Retention (1000 kroků) | ✅ 100% |
| Capacity | ✅ 50 center |
| Similarity Retrieval | ✅ 100% accuracy |

#### Rekonstrukce vzpomínek

| Metrika | Hodnota | Interpretace |
|---------|---------|--------------|
| Self-Reconstruction | **0.92** | B grade - Velmi dobrá |
| Stárnutí (Q4-Q1) | +0.20 | D grade - Staré vzpomínky degradují |
| Self-Weight | 0.38 | Retrieval je "měkký" (RBF) |

---

## 📚 Citace

Pokud používáš Cognitive Memory ve své práci, prosím cituj:

```bibtex
@software{cognitive_memory_2026,
  author = {Seidl, Michal},
  title = {Cognitive Memory: Biologicky inspirovaný systém persistentní paměti pro LLM},
  year = {2026},
  publisher = {OpenTechLab Jablonec nad Nisou},
  version = {2.0-beta},
  url = {https://github.com/OpenTechLab/cognitive-memory}
}
```

### Teoretický základ

Systém vychází z těchto konceptů:

- **Atkinson-Shiffrin model** (1968): Rozdělení STM/LTM
- **Complementary Learning Systems** (McClelland, McNaughton, O'Reilly, 1995): Hippocampus-neocortex interakce
- **Memory consolidation** (Sleep-wake cycle)
- **RBF networks** (Radial Basis Function kernels)
- **Reaction-diffusion systems** (Turing, 1952): Difuze v prostoru

---

## 🛠️ Pokročilé použití

### Přizpůsobení projekcí

```python
from cognitive_memory.projections import TerrainProjection

# Vlastní 64D → 3D projekce (např. autoenkodér)
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

# Použití
config.terrain_projection = CustomProjection()
```

### Monitorování stavu

```python
# Získej detailní statistiky
ltm_stats = memory.ltm_centers.get_stats()
print(f"LTM centra: {ltm_stats['n_active']}")
print(f"Průměrná intenzita: {ltm_stats['h_mean']:.3f}")
print(f"Max intenzita: {ltm_stats['h_max']:.3f}")

# Terénní statistiky
terrain_stats = memory.ltm_terrain.get_stats()
print(f"Celková energie: {terrain_stats['total_energy']:.3f}")

# Úroveň únavy
fatigue = memory.consolidator.get_fatigue_level()
print(f"Únava: {fatigue * 100:.1f}%")
```

### Export vizualizací

```python
# Export 3D terén jako numpy array
terrain_h = memory.ltm_terrain.H.cpu().numpy()  # [48, 48, 48]
terrain_e = memory.ltm_terrain.E.cpu().numpy()  # [48, 48, 48, 4]

# Použij matplotlib/plotly pro vizualizaci
import matplotlib.pyplot as plt
plt.imshow(terrain_h[:, :, 24], cmap='viridis')
plt.title("LTM Terrain - Central Slice")
plt.colorbar()
plt.savefig("terrain_slice.png")
```

---

## 🐛 Troubleshooting

| Problém | Možná příčina | Řešení |
|---------|---------------|--------|
| Memory Error při inicializaci | Příliš velký 3D grid | Snížit `terrain_resolution` na 32 nebo 24 |
| LTM se rychle plní | Příliš agresivní vytváření center | Snížit `ltm_new_center_threshold` na 0.5 |
| Žádné konsolidace | Nízká aktivita STM | Snížit `fatigue_threshold` na 3.0 |
| Příliš pomalé čtení | Velký `ltm_top_k_read` | Snížit na 16 nebo 8 |
| Gate vždy zavřený | Příliš záporný `gate_bias` | Zvýšit na -1.5 |
| Hodnoty "explodují" | Příliš silný zápis | Snížit `write_strength_base` na 0.1 |

### Debug mód

```python
# Zapni verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Kontrola NaN
assert not torch.isnan(memory.ltm_centers.h).any(), "NaN v intenzitách!"
assert not torch.isnan(memory.ltm_terrain.H).any(), "NaN v terénu!"
```

---

## 📖 Další dokumentace

- **`BioCortexAI_Documentation_EN.md`**: Kompletní vědecká dokumentace
- **`plan.md`**: Původní matematický návrh (developer notes)
- **`memory-tests/README.md`**: Testovací framework a benchmark

---

## 🤝 Podpora

### Issues

Pokud narazíš na problém:
1. Ověř, že používáš nejnovější verzi
2. Zkontroluj [Troubleshooting](#-troubleshooting)
3. Otevři issue s:
   - Verzí Python a PyTorch
   - Minimálním reprodukčním příkladem
   - Chybovým hlášením

### Kontakt

- **Email**: vyvoj@opentechlab.cz
- **Web**: [www.opentechlab.cz](https://www.opentechlab.cz)
- **Projekt**: BioCortexAI

---

## 📄 Licence

**CC BY-NC 4.0** (Creative Commons Attribution-NonCommercial 4.0)

✅ **Povoleno:**
- Použití pro výzkum a vzdělávání
- Modifikace a distribuce (s uvedením autora)
- Soukromé experimenty

❌ **Zakázáno:**
- Komerční využití bez licence
- Patent claims

Pro komerční licenci kontaktujte: opentechlab@opentechlab.cz

---

**Framework:** BioCortexAI v2.0-beta  
**Author:** Michal Seidl, OpenTechLab Jablonec nad Nisou s.r.o.  
**Status:** ✅ Production-ready

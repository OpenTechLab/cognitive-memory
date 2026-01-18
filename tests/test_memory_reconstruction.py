"""
Test rekonstrukce vzpomínek po dlouhém běhu.

Tento skript analyzuje kvalitu vybavování vzpomínek jako funkci jejich "stáří"
(aproximováno intenzitou h - čerstvější vzpomínky mají vyšší h).

Měří:
1. Přesnost rekonstrukce (cosine similarity mezi uloženou a vybavenou hodnotou V)
2. Self-discrimination (dostanu zpět sám sebe, nebo průměr s ostatními?)
3. Splývání podle stáří (mají staré vzpomínky horší rekonstrukci?)
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Přidej parent do path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cognitive_memory.memory_centers import MemoryCenters


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spočítá kosinovou podobnost mezi dvěma vektory."""
    a = a.flatten()
    b = b.flatten()
    return (torch.dot(a, b) / (torch.norm(a) * torch.norm(b) + 1e-8)).item()


def test_reconstruction():
    print("=" * 60)
    print("🔬 TEST REKONSTRUKCE VZPOMÍNEK")
    print("=" * 60)
    
    # Najdi nejnovější snapshot
    base_dir = Path("stress_test_results")
    snapshots = list(base_dir.rglob("memory.pt"))
    if not snapshots:
        print("❌ Nenalezen žádný snapshot!")
        return
    
    latest = max(snapshots, key=lambda p: p.stat().st_mtime)
    print(f"📂 Načítám: {latest}")
    
    state = torch.load(str(latest), map_location='cpu', weights_only=False)
    
    # Extrakce LTM center
    centers_state = state['ltm_centers']
    
    # Rekonstrukce MemoryCenters objektu
    n_centers = centers_state['K'].shape[0]
    d_key = centers_state['K'].shape[1]
    d_value = centers_state['V'].shape[1]
    d_emotion = centers_state['e'].shape[1] if len(centers_state['e'].shape) > 1 else 4
    
    # Vytvoř prázdnou instanci a naplň ji daty
    centers = MemoryCenters(
        n_centers=n_centers,
        d_key=d_key,
        d_value=d_value,
        d_emotion=d_emotion,
        sigma_read=0.5,  # Default
    )
    
    # Načti buffery
    centers.K = centers_state['K']
    centers.V = centers_state['V']
    centers.h = centers_state['h']
    centers.e = centers_state['e']
    centers.active = centers_state['active']
    centers.usage = centers_state.get('usage', torch.zeros(n_centers))
    centers.age = centers_state.get('age', torch.zeros(n_centers))
    
    # Filtruj aktivní centra
    active_mask = centers.active.bool()
    n_active = active_mask.sum().item()
    print(f"✅ Načteno {n_active} aktivních center")
    
    if n_active < 10:
        print("⚠️ Příliš málo center pro analýzu!")
        return
    
    # Získej data aktivních center
    active_indices = torch.where(active_mask)[0]
    K_active = centers.K[active_mask]
    V_active = centers.V[active_mask]
    h_active = centers.h[active_mask]
    
    # =============================================
    # TEST 1: Self-Reconstruction
    # =============================================
    print("\n🧪 TEST 1: Self-Reconstruction")
    print("-" * 40)
    
    reconstruction_scores = []
    weight_concentrations = []
    
    # Testujeme na vzorku (max 200 center)
    sample_size = min(200, n_active)
    sample_indices = torch.randperm(n_active)[:sample_size]
    
    for i, idx in enumerate(sample_indices):
        # Query = klíč tohoto centra
        query = K_active[idx:idx+1].unsqueeze(0)  # [1, 1, d_key]
        
        # Čtení z paměti
        r_V, r_E, weights, read_indices = centers.read(query, top_k=32)
        
        # Porovnej vybavenou hodnotu s uloženou
        original_V = V_active[idx]
        retrieved_V = r_V.squeeze()
        
        similarity = cosine_similarity(original_V, retrieved_V)
        reconstruction_scores.append(similarity)
        
        # Koncentrace váhy na správném centru
        # Najdi, zda je idx v read_indices
        global_idx = active_indices[idx]
        if global_idx in read_indices:
            pos = (read_indices.squeeze() == global_idx).nonzero()
            if len(pos) > 0:
                self_weight = weights.squeeze()[pos[0]].item()
            else:
                self_weight = 0.0
        else:
            self_weight = 0.0
        weight_concentrations.append(self_weight)
        
        if (i + 1) % 50 == 0:
            print(f"   Zpracováno {i+1}/{sample_size}...")
    
    avg_reconstruction = np.mean(reconstruction_scores)
    avg_self_weight = np.mean(weight_concentrations)
    
    print(f"\n📊 Výsledky Self-Reconstruction:")
    print(f"   Průměrná podobnost V: {avg_reconstruction:.4f}")
    print(f"   Průměrná váha self: {avg_self_weight:.4f}")
    
    # =============================================
    # TEST 2: Rekonstrukce podle stáří (h)
    # =============================================
    print("\n🧪 TEST 2: Rekonstrukce podle stáří (intenzita h)")
    print("-" * 40)
    
    # Seřaď centra podle h
    h_sorted_indices = torch.argsort(h_active)
    
    # Rozděl na kvartily
    quartile_size = n_active // 4
    quartiles = {
        "Q1 (nejstarší/nejslabší)": h_sorted_indices[:quartile_size],
        "Q2": h_sorted_indices[quartile_size:2*quartile_size],
        "Q3": h_sorted_indices[2*quartile_size:3*quartile_size],
        "Q4 (nejčerstvější/nejsilnější)": h_sorted_indices[3*quartile_size:],
    }
    
    quartile_scores = {}
    quartile_h_means = {}
    
    for name, indices in quartiles.items():
        scores = []
        # Sample max 50 z každého kvartilu
        sample = indices[:min(50, len(indices))]
        
        for idx in sample:
            query = K_active[idx:idx+1].unsqueeze(0)
            r_V, _, _, _ = centers.read(query, top_k=32)
            
            original_V = V_active[idx]
            retrieved_V = r_V.squeeze()
            similarity = cosine_similarity(original_V, retrieved_V)
            scores.append(similarity)
        
        quartile_scores[name] = np.mean(scores)
        quartile_h_means[name] = h_active[indices].mean().item()
        
        print(f"   {name}:")
        print(f"      h_mean: {quartile_h_means[name]:.2f}, reconstruction: {quartile_scores[name]:.4f}")
    
    # =============================================
    # TEST 3: Interference Test
    # =============================================
    print("\n🧪 TEST 3: Interference (splývání)")
    print("-" * 40)
    
    # Pro každé centrum najdi nejpodobnější jiné centrum
    # a změř, jak moc se jejich hodnoty V liší
    
    # Kosinová podobnost mezi všemi klíči
    K_norm = K_active / (torch.norm(K_active, dim=1, keepdim=True) + 1e-8)
    sim_matrix = torch.matmul(K_norm, K_norm.T)
    
    # Vynuluj diagonálu
    sim_matrix.fill_diagonal_(0)
    
    # Pro každé centrum najdi nejpodobnější
    most_similar_indices = sim_matrix.argmax(dim=1)
    most_similar_scores = sim_matrix.max(dim=1).values
    
    # Změř, jak moc se liší hodnoty V u podobných center
    v_differences = []
    for i in range(min(200, n_active)):
        j = most_similar_indices[i].item()
        v_sim = cosine_similarity(V_active[i], V_active[j])
        key_sim = most_similar_scores[i].item()
        # Ideálně: podobné klíče MOHOU mít podobné hodnoty (to je OK)
        # Problém je, když ROZDÍLNÉ klíče mají stejné hodnoty (splývání)
        v_differences.append((key_sim, v_sim))
    
    key_sims = [x[0] for x in v_differences]
    v_sims = [x[1] for x in v_differences]
    
    print(f"   Průměrná podobnost nejbližšího souseda (K): {np.mean(key_sims):.4f}")
    print(f"   Průměrná podobnost hodnot (V) sousedů: {np.mean(v_sims):.4f}")
    
    if np.mean(v_sims) > 0.9 and np.mean(key_sims) < 0.7:
        print("   ⚠️ VAROVÁNÍ: Hodnoty splývají i pro vzdálené klíče!")
    else:
        print("   ✅ Hodnoty jsou dostatečně distinktní.")
    
    # =============================================
    # Vizualizace
    # =============================================
    print("\n📊 Generuji vizualizace...")
    
    out_dir = Path("terrain_visualizations")
    out_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Analýza Rekonstrukce Vzpomínek po 9000 Interakcích', fontsize=14, fontweight='bold')
    
    # 1. Histogram rekonstrukčních skóre
    ax = axes[0, 0]
    ax.hist(reconstruction_scores, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(x=avg_reconstruction, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_reconstruction:.3f}')
    ax.set_xlabel('Cosine Similarity (Original V vs Retrieved V)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribuce Self-Reconstruction Skóre')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Self-weight histogram
    ax = axes[0, 1]
    ax.hist(weight_concentrations, bins=30, color='coral', edgecolor='black', alpha=0.7)
    ax.axvline(x=avg_self_weight, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_self_weight:.3f}')
    ax.set_xlabel('Weight on Self (during retrieval)')
    ax.set_ylabel('Frequency')
    ax.set_title('Koncentrace Váhy na Vlastním Centru')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Rekonstrukce podle kvartilů
    ax = axes[1, 0]
    names = list(quartile_scores.keys())
    scores = [quartile_scores[n] for n in names]
    h_means = [quartile_h_means[n] for n in names]
    
    bars = ax.bar(range(len(names)), scores, color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4'], edgecolor='black')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(['Q1\n(staré)', 'Q2', 'Q3', 'Q4\n(čerstvé)'])
    ax.set_ylabel('Průměrná Rekonstrukce')
    ax.set_title('Kvalita Rekonstrukce podle Stáří Vzpomínky')
    ax.set_ylim([0.8, 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Přidej h_mean jako text
    for i, (bar, h) in enumerate(zip(bars, h_means)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
               f'h={h:.1f}', ha='center', fontsize=9)
    
    # 4. Key similarity vs Value similarity scatter
    ax = axes[1, 1]
    sc = ax.scatter(key_sims, v_sims, alpha=0.5, c='purple', s=30)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='y=x')
    ax.set_xlabel('Key Similarity (nejbližší soused)')
    ax.set_ylabel('Value Similarity')
    ax.set_title('Interference: Podobnost Klíčů vs Hodnot')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = out_dir / "memory_reconstruction_analysis.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   -> Uloženo: {save_path}")
    plt.close()
    
    # =============================================
    # Souhrn
    # =============================================
    print("\n" + "=" * 60)
    print("📋 SOUHRN REKONSTRUKCE")
    print("=" * 60)
    
    # Skóre
    reconstruction_grade = "A" if avg_reconstruction > 0.95 else "B" if avg_reconstruction > 0.9 else "C" if avg_reconstruction > 0.8 else "D"
    aging_diff = quartile_scores["Q4 (nejčerstvější/nejsilnější)"] - quartile_scores["Q1 (nejstarší/nejslabší)"]
    aging_grade = "A" if aging_diff < 0.02 else "B" if aging_diff < 0.05 else "C" if aging_diff < 0.1 else "D"
    
    print(f"   Self-Reconstruction: {avg_reconstruction:.4f} (Grade: {reconstruction_grade})")
    print(f"   Stárnutí (Q4-Q1): {aging_diff:+.4f} (Grade: {aging_grade})")
    print(f"   Self-Weight: {avg_self_weight:.4f}")
    
    if reconstruction_grade in ["A", "B"] and aging_grade in ["A", "B"]:
        print("\n   ✅ PAMĚŤ FUNGUJE VÝBORNĚ!")
    elif reconstruction_grade == "D" or aging_grade == "D":
        print("\n   ⚠️ VAROVÁNÍ: Paměť vykazuje problémy.")
    else:
        print("\n   ℹ️ Paměť funguje přijatelně, ale je prostor pro zlepšení.")


if __name__ == "__main__":
    test_reconstruction()

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg') # Headless mode
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage
import sys

# Nastavení estetiky
plt.style.use('dark_background')
sns.set_style("dark")

def visualize_topology():
    print('='*60)
    print('🧬 ANALÝZA TOPOLOGIE PAMĚTI (64D)')
    print('='*60)

    # 1. Načtení dat
    base_dir = Path("stress_test_results")
    if not base_dir.exists():
        print("❌ Adresář stress_test_results neexistuje.")
        sys.exit(1)
        
    # Najít nejnovější snapshot memory.pt
    # Hledáme rekurzivně ve všech podadresářích
    snapshots = list(base_dir.rglob("memory.pt"))
    if not snapshots:
        print("❌ Nenalezen žádný memory.pt snapshot.")
        sys.exit(1)
        
    # Seřadit podle času změny
    latest_snapshot = max(snapshots, key=lambda p: p.stat().st_mtime)
    print(f"📂 Načítám snapshot: {latest_snapshot}")
    
    try:
        # Load safe
        state = torch.load(str(latest_snapshot), map_location='cpu', weights_only=False)
        centers = state['ltm_centers']
        
        # Extrakce aktivních center
        active_mask = centers['active'].cpu().bool().numpy()
        K = centers['K'].cpu().numpy()
        h = centers['h'].cpu().numpy()
        
        # Filtrujeme jen aktivní
        K_active = K[active_mask]
        h_active = h[active_mask]
        
        n_centers = len(K_active)
        print(f"✅ Načteno {n_centers} aktivních sémantických vektorů (64D).")
        
        if n_centers < 10:
            print("⚠️ Příliš málo center pro topologickou analýzu (<10).")
            return

        # Output directory
        out_dir = Path("terrain_visualizations")
        out_dir.mkdir(exist_ok=True)
        
        # ==========================================
        # 1. CLUSTERMAP (Hierarchická Heatmapa)
        # ==========================================
        print("🔥 Generuji Clustermap (Similarity Matrix)...")
        
        # Spočítat kosinovou podobnost
        # Epsilon pro stabilitu, ačkoli memory keys by měly být normalizované
        norms = np.linalg.norm(K_active, axis=1, keepdims=True)
        K_norm = K_active / (norms + 1e-8)
        sim_matrix = np.dot(K_norm, K_norm.T)
        
        # Vykreslení
        # Clustermap automaticky provede hierarchické shlukování a přeuspořádá řádky/sloupce
        g = sns.clustermap(
            sim_matrix,
            cmap='magma',
            figsize=(12, 12),
            xticklabels=False,
            yticklabels=False,
            dendrogram_ratio=(.1, .1),
            cbar_pos=(0.02, 0.8, 0.03, 0.15)
        )
        g.ax_heatmap.set_title(f"Sémantická Podobnost ({n_centers} center)", fontsize=16, pad=20)
        
        save_path = out_dir / "topology_01_clustermap.png"
        g.savefig(save_path, dpi=300)
        print(f"   -> Uloženo: {save_path}")
        plt.close()

        # ==========================================
        # 2. DENDROGRAM (Strom témat)
        # ==========================================
        print("🌳 Generuji Dendrogram...")
        
        plt.figure(figsize=(14, 7))
        
        # Wardova metoda minimalizuje rozptyl ve shlucích
        Z = linkage(K_active, method='ward')
        
        dendrogram(
            Z,
            leaf_rotation=90.,
            leaf_font_size=8.,
            no_labels=True, # Pro 450 bodů jsou popisky nečitelné
            color_threshold=Z[-10, 2] if n_centers > 10 else None # Barevné odlišení hlavních větví
        )
        
        plt.title('Hierarchický Strom Paměti (Dendrogram)', fontsize=16)
        plt.xlabel(f'Jednotlivá paměťová centra ({n_centers})')
        plt.ylabel('Sémantická vzdálenost (Ward distance)')
        plt.grid(True, alpha=0.1, axis='y')
        
        save_path = out_dir / "topology_02_dendrogram.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   -> Uloženo: {save_path}")
        plt.close()

        # ==========================================
        # 3. t-SNE (Manifold Projection)
        # ==========================================
        print("🌌 Počítám t-SNE projekci (64D -> 2D)...")
        
        # Perplexity určuje, kolik sousedů bere v potaz. Default 30.
        # Pro menší datasety (450) je 30 OK.
        tsne = TSNE(n_components=2, perplexity=min(30, n_centers-1), random_state=42, init='pca', learning_rate='auto')
        X_embedded = tsne.fit_transform(K_active)
        
        plt.figure(figsize=(12, 10))
        
        # Scatter s barvou podle intenzity vzpomínky
        sc = plt.scatter(
            X_embedded[:, 0], 
            X_embedded[:, 1], 
            c=h_active, 
            cmap='spring', 
            s=50, 
            alpha=0.7, 
            edgecolors='none'
        )
        
        plt.colorbar(sc, label='Intenzita vzpomínky (Usage)')
        plt.title('t-SNE Manifold Paměti\n(Shluky reprezentují sémantická témata)', fontsize=16)
        plt.xlabel('t-SNE dimension 1')
        plt.ylabel('t-SNE dimension 2')
        plt.grid(True, alpha=0.1)
        
        # Přidat anotace pro "velké" vzpomínky (nejaktivnější)
        # Najdeme top 5 center
        top_indices = np.argsort(h_active)[-5:]
        for idx in top_indices:
            plt.annotate(
                f"#{idx}", 
                (X_embedded[idx, 0], X_embedded[idx, 1]),
                xytext=(5, 5), textcoords='offset points',
                color='white', fontsize=9, fontweight='bold'
            )

        save_path = out_dir / "topology_03_tsne.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   -> Uloženo: {save_path}")
        plt.close()
        
        print("\n✅ Vizualizace topologie dokončena.")
        
    except Exception as e:
        print(f"\n❌ CHYBA Při vizualizaci: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    visualize_topology()

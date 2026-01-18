import torch
import numpy as np
import matplotlib
matplotlib.use('Agg') # Bezpečnější pro automatizovaný běh
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sys

# Přidat cestu k projektu
sys.path.insert(0, str(Path(__file__).parent.parent))

def visualize_centers_structure():
    print('='*60)
    print('🔬 ANALÝZA STRUKTURY PAMĚŤOVÝCH CENTER')
    print('='*60)
    
    # 1. Načtení snapshotu
    snapshot_dirs = list(Path("stress_test_results").glob("RealisticMixed_step_*/memory.pt"))
    if not snapshot_dirs:
        print("❌ Žádný snapshot.")
        return
    latest = max(snapshot_dirs, key=lambda p: p.stat().st_mtime)
    print(f"📂 Načítám: {latest.name}")
    state = torch.load(str(latest), map_location='cpu', weights_only=False)
    
    if 'ltm_centers' not in state:
        print("❌ Chybí LTM centers.")
        return
        
    centers = state['ltm_centers']
    active_mask = centers['active'].bool()
    
    # K: [N_active, 64]
    K = centers['K'][active_mask].numpy()
    # h: [N_active] - intenzita
    h = centers['h'][active_mask].numpy()
    
    n_active = len(K)
    print(f"✅ Načteno {n_active} aktivních center.")
    print(f"   Intenzita (h): min={h.min():.4f}, max={h.max():.4f}, mean={h.mean():.4f}")
    
    if n_active < 3:
        print("⚠️ Příliš málo center pro PCA.")
        return

    # 2. PCA Projekce do 2D (pro X, Y souřadnice mapy)
    print("🧮 Počítám PCA (64D -> 2D)...")
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(K)
    
    # Normalizace do [0, 1] pro hezké vykreslení
    coords_2d = (coords_2d - coords_2d.min(axis=0)) / (coords_2d.max(axis=0) - coords_2d.min(axis=0))
    
    # 3. Vizualizace: SCATTER 3D (X, Y z PCA, Z z intenzity)
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot
    scatter = ax.scatter(
        coords_2d[:, 0], 
        coords_2d[:, 1], 
        h, 
        c=h, 
        cmap='plasma',
        s=50, # velikost bodů
        alpha=0.8,
        depthshade=True
    )
    
    # "Drop lines" (stonky) k zemi pro lepší orientaci v 3D
    for i in range(n_active):
        ax.plot(
            [coords_2d[i, 0], coords_2d[i, 0]],
            [coords_2d[i, 1], coords_2d[i, 1]],
            [0, h[i]],
            c='gray', alpha=0.2, linewidth=0.5
        )
    
    ax.set_xlabel('PCA Component 1 (Semantic)', fontsize=12)
    ax.set_ylabel('PCA Component 2 (Semantic)', fontsize=12)
    ax.set_zlabel('Memory Intensity (Strength)', fontsize=12)
    ax.set_title(f'Skutečná struktura paměti ({n_active} center)\nZobrazena pomocí PCA projekce', fontsize=16)
    
    cbar = fig.colorbar(scatter, shrink=0.5, aspect=10)
    cbar.set_label('Intensity')
    
    output_path = "terrain_visualizations/centers_structure_scatter.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Scatter plot uložen: {output_path}")
    
    # 4. Vizualizace: "IDEÁLNÍ TERÉN" (Interpolace z těchto bodů)
    # Zkusíme vytvořit hladký povrch interpolací mezi těmito body (ne difuzí)
    from scipy.interpolate import griddata
    
    print("🎨 Generuji ideální topologickou mapu...")
    
    # Sníženo pro stabilitu (100x100 stačí pro vizualizaci)
    grid_x, grid_y = np.mgrid[0:1:100j, 0:1:100j]
    
    # Interpolace (cubic)
    grid_z = griddata(coords_2d, h, (grid_x, grid_y), method='linear', fill_value=0)
    # Vyhlazení (Gaussian filter) pro hezčí vzhled
    from scipy.ndimage import gaussian_filter
    grid_z_smooth = gaussian_filter(grid_z, sigma=2)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(grid_z_smooth.T, extent=(0,1,0,1), origin='lower', cmap='plasma')
    plt.colorbar(label='Intensity')
    plt.scatter(coords_2d[:, 0], coords_2d[:, 1], c='white', s=10, alpha=0.5, label='Centers')
    plt.title('Rekonstruovaná sémantická mapa (PCA + Interpolace)')
    plt.xlabel('Semantic X')
    plt.ylabel('Semantic Y')
    plt.legend()
    
    output_path_map = "terrain_visualizations/centers_ideal_map.png"
    plt.savefig(output_path_map, dpi=300, bbox_inches='tight')
    print(f"✅ Ideální mapa uložena: {output_path_map}")

    # 5. Vizualizace: 3D SURFACE PLOT
    print("🏔️ Generuji finální 3D krajinu...")
    
    # Použijeme již vypočtený grid_z_smooth
    from matplotlib.colors import LightSource
    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Logaritmické škálování pro vnímání detailů (volitelné, vypadá více "sci-fi")
    # Zde použijeme lineární, ale s colormapou, která to zvýrazní
    
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(grid_z_smooth.T, cmap=plt.cm.plasma, vert_exag=0.5, blend_mode='soft')
    
    # Surface plot
    surf = ax.plot_surface(
        grid_x, grid_y, grid_z_smooth.T,
        rstride=1, cstride=1,
        facecolors=rgb,
        linewidth=0,
        antialiased=True,
        shade=False # Shading děláme ručně přes LightSource
    )
    
    ax.view_init(elev=50, azim=-45) # Pohled z výšky
    ax.set_zlim(0, h.max() * 1.5) # Trochu místa nahoře
    
    # Odstranit osy pro čistý vzhled
    ax.set_axis_off()
    
    plt.title('BioCortex Semantic Landscape (Reconstructed)', fontsize=20, y=0.95)
    
    output_path_3d = "terrain_visualizations/centers_ideal_3d_landscape.png"
    plt.savefig(output_path_3d, dpi=300, bbox_inches='tight', pad_inches=0)
    print(f"✅ 3D Krajina uložena: {output_path_3d}")
    sys.exit(0)

if __name__ == "__main__":
    visualize_centers_structure()

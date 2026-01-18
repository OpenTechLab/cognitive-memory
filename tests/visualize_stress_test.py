# visualize_stress_test.py
"""
Vizualizace výsledků zátěžového testu Cognitive Memory.

Generuje:
- Časové grafy metrik (retention, interference, kapacita)
- 3D vizualizace terénu
- Heatmapy aktivace center
- Evoluce emocí
- Konsolidační eventy
"""

import sys
from pathlib import Path

# Přidej parent directory do sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib
# Použít ne-interaktivní backend pro ukládání souborů
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json
from mpl_toolkits.mplot3d import Axes3D

# Optional Plotly for interactive 3D
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from cognitive_memory import load_memory_state


class StressTestVisualizer:
    """Vizualizér pro výsledky zátěžového testu."""
    
    def __init__(self, results_dir: str = "stress_test_results"):
        self.results_dir = Path(results_dir)
        self.metrics_df = None
        self.consolidation_events = None
        
        # Načti data
        self._load_data()
        
        # Styl grafů
        plt.style.use('default')  # Místo seaborn
    
    def _load_data(self):
        """Načte CSV a JSON data."""
        # Hledej metrics_*.csv (wildcard)
        metrics_files = list(self.results_dir.glob("metrics_*.csv"))
        
        if metrics_files:
            metrics_path = metrics_files[0]
            self.metrics_df = pd.read_csv(metrics_path)
            print(f"✓ Loaded {len(self.metrics_df)} rows from {metrics_path.name}")
        else:
            print(f"⚠ Metrics not found: {self.results_dir / 'metrics.csv'}")
        
        consol_path = self.results_dir / "consolidation_events.json"
        if consol_path.exists():
            with open(consol_path, 'r') as f:
                self.consolidation_events = json.load(f)
            print(f"✓ Loaded {len(self.consolidation_events)} consolidation events")
        else:
            print(f"⚠ Consolidation events not found")
    
    def plot_overview(self, save_path: str = None):
        """
        Přehledový dashboard s klíčovými metrikami.
        
        OPRAVENO 2026-01-14:
        - Kompatibilita s novým CSV formátem (interaction vs step)
        """
        if self.metrics_df is None:
            print("⚠ No metrics to plot")
            return
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle('Cognitive Memory - Stress Test Overview', fontsize=16, fontweight='bold')
        
        df = self.metrics_df
        
        # Kompatibilita: najdi správný sloupec pro x-osu
        x_col = 'interaction' if 'interaction' in df.columns else 'step'
        x_label = 'Interaction' if x_col == 'interaction' else 'Step'
        
        # 1. Aktivní centra LTM/STM
        ax = axes[0, 0]
        ax.plot(df[x_col], df['n_active'], label='LTM Active', linewidth=2)
        ax.plot(df[x_col], df['stm_active'], label='STM Active', linewidth=2, alpha=0.7)
        ax.set_xlabel(x_label)
        ax.set_ylabel('Active Centers')
        ax.set_title('Memory Center Activation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Síla zápisu (write strength)
        ax = axes[0, 1]
        ax.plot(df[x_col], df['omega_mean'], label='Mean Write Strength', linewidth=2, color='green')
        ax.fill_between(df[x_col], 0, df['omega_mean'], alpha=0.3, color='green')
        ax.set_xlabel(x_label)
        ax.set_ylabel('ω (Write Strength)')
        ax.set_title('Write Strength Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Intenzita terénu (terrain intensity)
        ax = axes[1, 0]
        ax.plot(df[x_col], df['H_mean'], label='H Mean', linewidth=2, color='orange')
        ax.plot(df[x_col], df['H_max'], label='H Max', linewidth=2, alpha=0.7, color='red')
        ax.set_xlabel(x_label)
        ax.set_ylabel('Terrain Intensity')
        ax.set_title('3D Terrain Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Únava (fatigue) a konsolidace
        ax = axes[1, 1]
        ax.plot(df[x_col], df['fatigue'], label='Fatigue Level', linewidth=2, color='purple')
        ax.axhline(y=1.0, color='red', linestyle='--', label='Threshold', alpha=0.7)
        
        # Označení konsolidačních eventů
        if self.consolidation_events:
            # Kompatibilita: interaction vs step
            event_col = 'interaction' if 'interaction' in self.consolidation_events[0] else 'step'
            consol_steps = [e[event_col] for e in self.consolidation_events]
            ax.scatter(consol_steps, [1.0] * len(consol_steps), 
                      color='red', marker='v', s=100, zorder=5, label='Consolidation')
        
        ax.set_xlabel(x_label)
        ax.set_ylabel('Fatigue Level')
        ax.set_title('Fatigue & Consolidation Events')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Nová centra (vytváření)
        ax = axes[2, 0]
        # Kumulativní
        df['new_ltm_centers_cumsum'] = df['new_ltm_centers'].fillna(0).cumsum()
        df['new_stm_centers_cumsum'] = df['new_stm_centers'].fillna(0).cumsum()
        
        ax.plot(df[x_col], df['new_ltm_centers_cumsum'], 
               label='LTM New Centers', linewidth=2, color='blue')
        ax.plot(df[x_col], df['new_stm_centers_cumsum'], 
               label='STM New Centers', linewidth=2, color='cyan', alpha=0.7)
        ax.set_xlabel(x_label)
        ax.set_ylabel('Cumulative New Centers')
        ax.set_title('Memory Growth')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. Distribuce surprise (bimodální by měla být viditelná)
        ax = axes[2, 1]
        ax.hist(df['surprise'], bins=50, alpha=0.7, color='teal', edgecolor='black')
        ax.set_xlabel('Surprise Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Surprise Distribution (should be bimodal!)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Overview saved to {save_path}")
        
        plt.show()
    
    def plot_retention_analysis(self, save_path: str = None):
        """Analýza retence paměti v čase."""
        if self.metrics_df is None:
            return
        
        df = self.metrics_df
        
        # Kompatibilita: najdi správný sloupec pro x-osu
        x_col = 'interaction' if 'interaction' in df.columns else 'step'
        x_label = 'Interaction' if x_col == 'interaction' else 'Step'
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Memory Retention Analysis', fontsize=16, fontweight='bold')
        
        # 1. Retention rate podle času
        # Aproximace: pokud n_active roste, retention je OK
        ax = axes[0, 0]
        retention_proxy = df['n_active'] / df['n_active'].max()
        ax.plot(df[x_col], retention_proxy, linewidth=2, color='darkgreen')
        ax.fill_between(df[x_col], 0, retention_proxy, alpha=0.3, color='green')
        ax.set_xlabel(x_label)
        ax.set_ylabel('Retention Proxy (normalized)')
        ax.set_title('Memory Retention Over Time')
        ax.grid(True, alpha=0.3)
        
        # 2. Intenzita vs stáří
        ax = axes[0, 1]
        # Nemáme přímý access k h_i per centrum, použijeme průměry
        window = 100
        # Rolling mean (pokud je dost dat)
        if len(df) > window:
            df['h_mean_rolling'] = df['H_mean'].rolling(window=window).mean()
            y_data = df['h_mean_rolling']
        else:
            y_data = df['H_mean']
            
        ax.plot(df[x_col], y_data, linewidth=2, color='coral')
        ax.set_xlabel(x_label)
        ax.set_ylabel('Mean Intensity (smoothed)')
        ax.set_title(f'Memory Intensity Decay (window={window})')
        ax.grid(True, alpha=0.3)
        
        # 3. Poločas decay (analyticky)
        ax = axes[1, 0]
        # Teoretický decay: h(t) = h_0 * (1-λ)^t
        # Poločas: t_1/2 = ln(2)/ln(1/(1-λ))
        lambda_ltm = 3.8e-5
        t_half = np.log(2) / np.log(1 / (1 - lambda_ltm))
        
        steps = np.arange(0, len(df), 100)
        theoretical_decay = np.exp(-lambda_ltm * steps)
        
        ax.plot(steps, theoretical_decay, linewidth=3, color='red', 
               label=f'Theoretical (λ={lambda_ltm:.1e})')
        ax.axvline(x=t_half, color='orange', linestyle='--', 
                  label=f'Half-life: {int(t_half)} steps', linewidth=2)
        ax.set_xlabel('Steps')
        ax.set_ylabel('Relative Intensity')
        ax.set_title('Theoretical Memory Decay')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # 4. Konsolidace efekt
        if self.consolidation_events:
            ax = axes[1, 1]
            
            consol_df = pd.DataFrame(self.consolidation_events)
            ax.bar(range(len(consol_df)), consol_df['consolidated_centers'], 
                  color='steelblue', edgecolor='black', alpha=0.7)
            ax.set_xlabel('Consolidation Event Index')
            ax.set_ylabel('Centers Consolidated')
            ax.set_title('STM → LTM Consolidation Effectiveness')
            ax.grid(True, alpha=0.3, axis='y')
        else:
            axes[1, 1].text(0.5, 0.5, 'No consolidation events', 
                           ha='center', va='center', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Retention analysis saved to {save_path}")
        
        plt.show()
    
    def plot_terrain_3d_snapshot(self, snapshot_step: int, save_path: str = None):
        """
        3D vizualizace terénu v daném kroku.
        
        Args:
            snapshot_step: Krok kde byl snapshot uložen
            save_path: Kam uložit obrázek
        """
        # Najdi snapshot
        snapshot_dirs = list(self.results_dir.glob(f"*_step_{snapshot_step}"))
        if not snapshot_dirs:
            print(f"⚠ No snapshot found for step {snapshot_step}")
            return
        
        snapshot_path = snapshot_dirs[0] / "memory.pt"
        if not snapshot_path.exists():
            print(f"⚠ Snapshot file not found: {snapshot_path}")
            return
        
        # Načti terén
        state = load_memory_state(str(snapshot_path))
        terrain = state['ltm_terrain']
        
        # 3D grid
        H = terrain.H.cpu().numpy()
        E = terrain.E.cpu().numpy()
        
        # Vytvoř průřez (2D slice pro vizualizaci)
        fig = plt.figure(figsize=(16, 5))
        
        # XY průřez (střed Z)
        ax1 = fig.add_subplot(131)
        z_mid = H.shape[2] // 2
        im1 = ax1.imshow(H[:, :, z_mid], cmap='hot', interpolation='bilinear')
        ax1.set_title(f'XY Slice (Z={z_mid})', fontsize=12, fontweight='bold')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        plt.colorbar(im1, ax=ax1, label='Intensity')
        
        # XZ průřez
        ax2 = fig.add_subplot(132)
        y_mid = H.shape[1] // 2
        im2 = ax2.imshow(H[:, y_mid, :], cmap='hot', interpolation='bilinear')
        ax2.set_title(f'XZ Slice (Y={y_mid})', fontsize=12, fontweight='bold')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Z')
        plt.colorbar(im2, ax=ax2, label='Intensity')
        
        # YZ průřez
        ax3 = fig.add_subplot(133)
        x_mid = H.shape[0] // 2
        im3 = ax3.imshow(H[x_mid, :, :], cmap='hot', interpolation='bilinear')
        ax3.set_title(f'YZ Slice (X={x_mid})', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Y')
        ax3.set_ylabel('Z')
        plt.colorbar(im3, ax=ax3, label='Intensity')
        
        fig.suptitle(f'3D Terrain Intensity - Step {snapshot_step}', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 3D terrain snapshot saved to {save_path}")
        
        plt.show()
    
    def plot_terrain_3d_interactive(self, snapshot_step: int, save_html: str = None):
        """
        Interaktivní 3D vizualizace pomocí Plotly.
        
        Args:
            snapshot_step: Krok snapshotu
            save_html: Kam uložit HTML
        """
        if not PLOTLY_AVAILABLE:
            print("⚠ Plotly not installed. Skipping interactive 3D visualization.")
            print("   Install with: pip install plotly")
            return
        
        # Najdi snapshot
        snapshot_dirs = list(self.results_dir.glob(f"*_step_{snapshot_step}"))
        if not snapshot_dirs:
            print(f"⚠ No snapshot found for step {snapshot_step}")
            return
        
        snapshot_path = snapshot_dirs[0] / "memory.pt"
        state = load_memory_state(str(snapshot_path))
        terrain = state['ltm_terrain']
        
        H = terrain.H.cpu().numpy()
        
        # Vytvoř threshold pro zobrazení jen relevantních voxelů
        threshold = H.mean() + H.std()
        
        # Získej souřadnice aktivních voxelů
        x, y, z = np.where(H > threshold)
        intensities = H[x, y, z]
        
        # Plotly scatter 3D
        fig = go.Figure(data=[go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=3,
                color=intensities,
                colorscale='Hot',
                showscale=True,
                colorbar=dict(title="Intensity"),
                opacity=0.8
            ),
            text=[f'H={h:.3f}' for h in intensities],
            hoverinfo='text'
        )])
        
        fig.update_layout(
            title=f'3D Terrain Visualization - Step {snapshot_step}',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'
            ),
            width=900,
            height=700
        )
        
        if save_html:
            fig.write_html(save_html)
            print(f"✓ Interactive 3D saved to {save_html}")
        
        fig.show()
    
    def plot_consolidation_timeline(self, save_path: str = None):
        """Timeline konsolidačních eventů."""
        if not self.consolidation_events:
            print("⚠ No consolidation events to plot")
            return
        
        df_consol = pd.DataFrame(self.consolidation_events)
        
        # Dynamická detekce sloupce
        x_col = 'interaction' if 'interaction' in df_consol.columns else 'step'
        x_label = 'Interaction' if x_col == 'interaction' else 'Step'
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        fig.suptitle('Consolidation Events Timeline', fontsize=16, fontweight='bold')
        
        # Počet konsolidovaných center
        ax = axes[0]
        ax.bar(df_consol[x_col], df_consol['consolidated_centers'], 
              width=100, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_ylabel('Centers Consolidated')
        ax.set_title('Consolidation Volume')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Únava před/po
        ax = axes[1]
        ax.plot(df_consol[x_col], df_consol['pre_fatigue'], 
               'o-', label='Pre-fatigue', linewidth=2, markersize=8, color='red')
        ax.plot(df_consol[x_col], df_consol['post_fatigue'], 
               'o-', label='Post-fatigue', linewidth=2, markersize=8, color='green')
        ax.set_xlabel(x_label)
        ax.set_ylabel('Fatigue Level')
        ax.set_title('Fatigue Before/After Consolidation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Consolidation timeline saved to {save_path}")
        
        plt.show()
    
    def generate_full_report(self, output_dir: str = None):
        """Vygeneruje kompletní report se všemi vizualizacemi."""
        if output_dir is None:
            output_dir = self.results_dir / "visualizations"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True, parents=True)
        
        print("\n" + "="*60)
        print("GENERATING FULL VISUALIZATION REPORT")
        print("="*60 + "\n")
        
        # 1. Overview
        print("📊 Generating overview...")
        self.plot_overview(save_path=str(output_dir / "01_overview.png"))
        
        # 2. Retention
        print("📈 Generating retention analysis...")
        self.plot_retention_analysis(save_path=str(output_dir / "02_retention.png"))
        
        # 3. Consolidation
        if self.consolidation_events:
            print("💤 Generating consolidation timeline...")
            self.plot_consolidation_timeline(save_path=str(output_dir / "03_consolidation.png"))
        
        # 4. 3D terén snapshots
        if self.metrics_df is not None:
            # Dynamická detekce sloupce
            x_col = 'interaction' if 'interaction' in self.metrics_df.columns else 'step'
            
            snapshot_steps = self.metrics_df[x_col].iloc[::1000].tolist()
            for i, step in enumerate(snapshot_steps[:3]):  # Prvních 3
                print(f"🗺️  Generating 3D terrain snapshot {i+1}/3 ({x_col} {step})...")
                self.plot_terrain_3d_snapshot(
                    step, 
                    save_path=str(output_dir / f"04_terrain_{x_col}_{step}.png")
                )
        
        print("\n" + "="*60)
        print(f"✓ Full report generated in: {output_dir}")
        print("="*60)


def main():
    """Entry point."""
    print("="*60)
    print("STRESS TEST VISUALIZATION")
    print("="*60 + "\n")
    
    viz = StressTestVisualizer("stress_test_results")
    
    # Generuj kompletní report
    viz.generate_full_report()


if __name__ == "__main__":
    main()

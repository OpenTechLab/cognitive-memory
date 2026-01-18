# stress_test_memory.py
"""
Zátěžový test Cognitive Memory systému.

OPRAVENO 2026-01-14:
- Realistické sekvence tokenů (50-200 tokenů/interakce)
- Homeostáza jednou za INTERAKCI (ne za token!)
- Správná kalibrace pro roční provoz

Simuluje několik měsíců zápisů a sleduje:
- Retention (jak paměť drží)
- Interference (zda se neprolínají nesouvisející vzpomínky)
- Kapacitu (merge/prune)
- Stabilitu difuze
- Konsolidaci STM→LTM

Výstupy:
- CSV s metrikami
- JSON detailní záznamy
- Vizualizace 3D terénu
- Grafické reporty
"""

import sys
from pathlib import Path

# Přidej parent directory do sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

from cognitive_memory import (
    MemoryConfig,
    Terrain3D,
    MemoryCenters,
    MemoryWriter,
    SleepConsolidator,
    AutomaticConsolidator,
    save_memory_state,
)

# Import realistických scénářů
try:
    from realistic_scenarios import RealisticMixedScenario
    REALISTIC_SCENARIOS_AVAILABLE = True
except ImportError:
    REALISTIC_SCENARIOS_AVAILABLE = False


class StressTestScenario:
    """Scénář pro zátěžový test (memory-efficient generator)."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.n_total = 0
        self.d_model = 0
        self.cluster_centers = 5
    
    def generate_embedding(self, step: int) -> Tuple[torch.Tensor, Dict]:
        """
        Generuje JEDEN embedding on-the-fly (lazy).
        
        Args:
            step: Index kroku
            
        Returns:
            (embedding, metadata)
        """
        raise NotImplementedError
    
    def prepare(self, n: int, d_model: int, cluster_centers: int = 5):
        """Připraví generátor (bez alokace embeddingů)."""
        self.n_total = n
        self.d_model = d_model
        self.cluster_centers = cluster_centers


class RandomScenario(StressTestScenario):
    """Náhodné embeddingy (worst case pro interferenci)."""
    
    def __init__(self):
        super().__init__(
            "Random",
            "Náhodné embeddingy bez struktury"
        )
    
    @torch.no_grad()
    def generate_embedding(self, step: int) -> Tuple[torch.Tensor, Dict]:
        # Čistě náhodný embedding
        emb = torch.randn(self.d_model)
        emb = emb / emb.norm()
        
        # Náhodné emoce
        emotions = torch.rand(4) + 0.5  # [0.5, 1.5]
        
        return emb, {
            "emotions": emotions,
            "surprise": np.random.uniform(0, 1),
            "cluster_id": -1,
            "timestamp": step
        }


class ClusteredScenario(StressTestScenario):
    """Tematicky seskupené embeddingy (realistické)."""
    
    def __init__(self):
        super().__init__(
            "Clustered",
            "Embeddingy seskupené do tematických clusterů"
        )
        self.centers = None
    
    def prepare(self, n: int, d_model: int, cluster_centers: int = 5):
        super().prepare(n, d_model, cluster_centers)
        # Vytvoř cluster centra jednou
        with torch.no_grad():
            self.centers = torch.randn(cluster_centers, d_model)
            self.centers = self.centers / self.centers.norm(dim=-1, keepdim=True)
    
    @torch.no_grad()
    def generate_embedding(self, step: int) -> Tuple[torch.Tensor, Dict]:
        # Vyber náhodný cluster
        cluster_id = np.random.randint(0, self.cluster_centers)
        center = self.centers[cluster_id]
        
        # Embedding blízko clusteru (Gaussian noise)
        noise = torch.randn(self.d_model) * 0.3
        emb = center + noise
        emb = emb / emb.norm()
        
        # Emoce korelují s clusterem
        base_emotions = torch.tensor([
            0.8 + cluster_id * 0.1,
            1.0 + (cluster_id % 2) * 0.2,
            1.0 - cluster_id * 0.05,
            0.9 + cluster_id * 0.08
        ])
        emotions = base_emotions + torch.randn(4) * 0.1
        emotions = torch.clamp(emotions, 0.5, 1.5)
        
        return emb, {
            "emotions": emotions,
            "surprise": np.random.uniform(0, 0.5),
            "cluster_id": cluster_id,
            "timestamp": step
        }


class TemporalScenario(StressTestScenario):
    """Časově strukturované embeddingy (narrative)."""
    
    def __init__(self):
        super().__init__(
            "Temporal",
            "Embeddingy s časovou strukturou (evoluce tématu)"
        )
        self.current = None
    
    def prepare(self, n: int, d_model: int, cluster_centers: int = 5):
        super().prepare(n, d_model, cluster_centers)
        # Počáteční embedding
        with torch.no_grad():
            self.current = torch.randn(d_model)
            self.current = self.current / self.current.norm()
    
    @torch.no_grad()
    def generate_embedding(self, step: int) -> Tuple[torch.Tensor, Dict]:
        # Malá změna od předchozího
        drift = torch.randn(self.d_model) * 0.1
        self.current = self.current + drift
        self.current = self.current / self.current.norm()
        
        # Emoce se vyvíjejí pomalu
        phase = step / self.n_total * 2 * np.pi
        emotions = torch.tensor([
            1.0 + 0.2 * np.sin(phase),
            1.0 + 0.2 * np.cos(phase),
            1.0 - 0.1 * np.sin(phase),
            1.0 + 0.15 * np.sin(phase * 2)
        ])
        
        return self.current.clone(), {
            "emotions": emotions,
            "surprise": 0.2 + 0.3 * abs(np.sin(phase)),
            "cluster_id": int(step / (self.n_total / self.cluster_centers)),
            "timestamp": step
        }


class StressTestRunner:
    """Hlavní runner pro zátěžové testy."""
    
    def __init__(
        self,
        config: MemoryConfig,
        output_dir: str = "stress_test_results",
        device: str = "cpu"
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.device = device
        
        # Vytvoř komponenty
        self.ltm_centers = MemoryCenters(
            n_centers=config.n_ltm_centers,
            d_key=config.d_memory_key,
            d_value=config.d_memory_value,
            sigma_read=config.ltm_sigma_read,
            sigma_write=config.ltm_sigma_write,
            leak=config.ltm_leak,
            leak_emotion=config.ltm_leak_emotion,
            leak_value=config.ltm_leak_value,
            alpha_value=config.ltm_alpha_value,
            alpha_emotion=config.ltm_alpha_emotion,
            device=device
        )
        
        self.stm_centers = MemoryCenters(
            n_centers=config.n_stm_centers,
            d_key=config.d_stm_key,
            d_value=config.d_memory_value,
            sigma_read=config.stm_sigma_read,
            sigma_write=config.stm_sigma_write,
            leak=config.stm_leak,
            leak_emotion=config.stm_leak_emotion,
            leak_value=config.stm_leak_value,
            alpha_value=config.stm_alpha_value,
            alpha_emotion=config.stm_alpha_emotion,
            device=device
        )
        
        self.ltm_terrain = Terrain3D(
            resolution=config.terrain_resolution,
            alpha_h=config.terrain_ltm_alpha_h,
            alpha_e=config.terrain_ltm_alpha_e,
            leak=config.terrain_ltm_lambda,
            device=device
        )
        
        self.stm_terrain = Terrain3D(
            resolution=config.terrain_resolution,
            alpha_h=config.terrain_stm_alpha_h,
            alpha_e=config.terrain_stm_alpha_e,
            leak=config.terrain_stm_lambda,
            device=device
        )
        
        self.writer = MemoryWriter(
            d_model=config.d_model,
            d_ltm_key=config.d_memory_key,
            d_stm_key=config.d_stm_key,
            d_value=config.d_memory_value,
            segment_size=config.segment_size,
            # OPRAVA: Předat write parametry z config!
            write_strength_base=config.write_strength_base,
            write_novelty_weight=config.write_novelty_weight,
            write_surprise_weight=config.write_surprise_weight,
            write_emotion_weight=config.write_emotion_weight,
            write_bias=config.write_bias,
        )
        
        sleep_consolidator = SleepConsolidator(
            d_stm_key=config.d_stm_key,
            d_ltm_key=config.d_memory_key,
            fatigue_threshold=config.fatigue_threshold,
            fatigue_leak=config.fatigue_leak,
            # OPRAVA: Předat consolidation parametry!
            consolidation_kappa=config.consolidation_kappa,
        )
        self.consolidator = AutomaticConsolidator(
            sleep_consolidator,
            min_interval=50  # Sníženo pro častější kontrolu
        )
        
        # CSV writer (streaming místo list v RAM)
        self.csv_writer = None
        self.csv_file = None
        self.consolidation_events = []
        
        import gc
        self.gc = gc  # Pro pravidelné čištění
        
    @torch.no_grad()  # Vypnout gradienty globally
    def run_scenario(
        self,
        scenario: StressTestScenario,
        n_interactions: int,
        tokens_per_interaction: int = 100,
        save_interval: int = 100,
        use_sequences: bool = True
    ):
        """
        Spustí test scénáře s REALISTICKOU simulací.
        
        OPRAVENO 2026-01-14:
        - n_interactions = počet INTERAKCÍ (ne tokenů!)
        - Každá interakce = sekvence tokenů (prompt + odpověď)
        - Homeostáza se volá JEDNOU za interakci (správná kalibrace!)
        
        Args:
            scenario: Scénář pro test
            n_interactions: Počet INTERAKCÍ (uživatel-model výměn)
            tokens_per_interaction: Průměrný počet tokenů na interakci
            save_interval: Interval pro ukládání snapshots (v interakcích)
            use_sequences: True = realistické sekvence, False = legacy single-token
        """
        print(f"\n{'='*60}")
        print(f"Running scenario: {scenario.name}")
        print(f"Description: {scenario.description}")
        print(f"Interactions: {n_interactions}")
        if use_sequences:
            print(f"Tokens/interaction: ~{tokens_per_interaction}")
            print(f"Total tokens: ~{n_interactions * tokens_per_interaction:,}")
        print(f"{'='*60}\n")
        
        # Připrav scénář
        print("Preparing scenario...")
        # Pro sekvence připrav s celkovým počtem tokenů (pro zpětnou kompatibilitu)
        total_tokens = n_interactions * tokens_per_interaction if use_sequences else n_interactions
        scenario.prepare(total_tokens, self.config.d_model)
        
        # Otevři CSV file pro streaming
        csv_path = self.output_dir / f"metrics_{scenario.name}.csv"
        self.csv_file = open(csv_path, 'w', newline='')
        
        # Header (přidány nové sloupce)
        fieldnames = ['interaction', 'n_tokens', 'scenario', 'cluster_id', 'surprise', 
                     'omega_mean', 'omega_max', 'new_ltm_centers', 'new_stm_centers',
                     'n_active', 'n_total', 'stm_active', 
                     'H_mean', 'H_max', 'H_std', 'fatigue', 'is_aha_moment']
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.csv_writer.writeheader()
        
        # === NULTÝ SNÍMEK (výchozí stav) ===
        self._save_snapshot(0, scenario.name)
        
        # Simulace - OPRAVENÁ
        for interaction in tqdm(range(n_interactions), desc="Simulating interactions"):
            
            if use_sequences and hasattr(scenario, 'generate_sequence'):
                # NOVÉ: Generuj celou sekvenci tokenů pro interakci
                sequence, meta = scenario.generate_sequence(interaction)
                n_tokens = sequence.shape[0]
                
                # Hidden states jako sekvence [1, T, D]
                hidden_states = sequence.unsqueeze(0).to(self.device)
            else:
                # Legacy: jednotlivé embeddingy
                emb, meta = scenario.generate_embedding(interaction)
                n_tokens = 1
                hidden_states = emb.unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, D]
            
            # Zápis do paměti (celá sekvence najednou)
            # Zápis do paměti (celá sekvence najednou)
            stats = self.writer.write_to_memory(
                hidden_states=hidden_states,
                emotions=meta["emotions"].to(self.device),
                ltm_centers=self.ltm_centers,
                stm_centers=self.stm_centers,
                ltm_terrain=self.ltm_terrain,
                stm_terrain=self.stm_terrain,
                surprise=torch.full((1, hidden_states.shape[1]), meta["surprise"], device=self.device),
                # PŘEDÁNÍ THRESHOLDU Z CONFIGU
                ltm_threshold=self.config.ltm_new_center_threshold,
                # STM threshold momentálně není v configu explicitně, používáme default nebo stejný
                stm_threshold=0.3 
            )
            
            # ========================================
            # KRITICKÁ OPRAVA: Homeostáza JEDNOU za INTERAKCI
            # ========================================
            # Podle dokumentace: "Volat jednou za interakci."
            # Koeficienty jsou kalibrovány pro 50 interakcí/den, 18250 interakcí/rok
            # NE pro počet tokenů!
            self.ltm_centers.homeostasis_step()
            self.stm_centers.homeostasis_step()
            self.ltm_terrain.step()
            self.stm_terrain.step()
            
            # Konsolidace (měřeno v interakcích)
            consolidation_stats = self.consolidator.step(
                stats.get('omega_mean', 0),
                self.stm_centers,
                self.ltm_centers,
                self.stm_terrain,
                self.ltm_terrain
            )
            
            if consolidation_stats is not None:
                self.consolidation_events.append({
                    'interaction': interaction,
                    **consolidation_stats
                })
            
            # Metriky
            ltm_stats = self.ltm_centers.get_stats()
            terrain_stats = self.ltm_terrain.get_stats()
            
            row = {
                'interaction': interaction,
                'n_tokens': n_tokens if use_sequences else 1,
                'scenario': scenario.name,
                'cluster_id': meta['cluster_id'],
                'surprise': meta['surprise'],
                'omega_mean': stats.get('omega_mean', 0),
                'omega_max': stats.get('omega_max', 0),
                'new_ltm_centers': stats.get('new_ltm_centers', 0),
                'new_stm_centers': stats.get('new_stm_centers', 0),
                'n_active': ltm_stats['n_active'],
                'n_total': ltm_stats['n_total'],
                'stm_active': self.stm_centers.get_n_active(),
                'H_mean': terrain_stats['H_mean'],
                'H_max': terrain_stats['H_max'],
                'H_std': terrain_stats['H_std'],
                'fatigue': self.consolidator.consolidator.get_fatigue_level(),
                'is_aha_moment': meta.get('is_aha_moment', False)
            }
            self.csv_writer.writerow(row)
            
            # Flush každých 50 interakcí + GC
            if (interaction + 1) % 50 == 0:
                self.csv_file.flush()
                self.gc.collect()
            
            # Snapshot
            if (interaction + 1) % save_interval == 0:
                self._save_snapshot(interaction, scenario.name)
        
        # Zavři CSV
        self.csv_file.close()
        
        print(f"\n✓ Scenario '{scenario.name}' completed")
        print(f"  Total interactions: {n_interactions}")
        print(f"  LTM active centers: {self.ltm_centers.get_n_active()}")
        print(f"  STM active centers: {self.stm_centers.get_n_active()}")
        print(f"  Consolidation events: {len(self.consolidation_events)}")
        print(f"  Metrics saved to: {csv_path}")
    
    def _save_snapshot(self, step: int, scenario_name: str):
        """Uloží snapshot paměti a vygeneruje vizualizace pro timelapse."""
        snapshot_dir = self.output_dir / f"{scenario_name}_step_{step}"
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        
        save_memory_state(
            str(snapshot_dir / "memory.pt"),
            self.config,
            self.ltm_centers,
            self.stm_centers,
            self.ltm_terrain,
            self.stm_terrain
        )
        
        # Generuj vizualizace pro timelapse
        self._generate_timelapse_visualizations(step, scenario_name)
    
    def _generate_timelapse_visualizations(self, step: int, scenario_name: str):
        """
        Generuje kvalitní vizualizace pro timelapse efekt.
        
        Vytváří 4 typy vizualizací:
        1. centers_ideal_map - 2D sémantická mapa
        2. centers_ideal_3d_landscape - 3D krajina
        3. topology_clustermap - Matice podobnosti
        4. topology_dendrogram - Hierarchický strom
        
        Výstupy jsou číslovány: step_0000, step_0499, ...
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.decomposition import PCA
            from scipy.interpolate import griddata
            from scipy.ndimage import gaussian_filter
            from scipy.cluster.hierarchy import dendrogram, linkage
            from matplotlib.colors import LightSource
            
            # Výstupní adresář
            timelapse_dir = self.output_dir / "timelapse"
            timelapse_dir.mkdir(exist_ok=True, parents=True)
            
            # Formátování step čísla
            step_str = f"step_{step:04d}"
            
            # Získej data aktivních LTM center
            active_mask = self.ltm_centers.active.bool()
            n_active = active_mask.sum().item()
            
            if n_active < 3:
                # Pro prázdnou paměť vytvoř placeholder
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.text(0.5, 0.5, f'Step {step}\n\nPrázdná paměť\n({n_active} center)', 
                       ha='center', va='center', fontsize=24, color='gray')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                plt.savefig(timelapse_dir / f"map_{step_str}.png", dpi=150)
                plt.close()
                return
            
            K_active = self.ltm_centers.K[active_mask].cpu().numpy()
            h_active = self.ltm_centers.h[active_mask].cpu().numpy()
            
            # === PCA projekce (společná pro všechny vizualizace) ===
            pca = PCA(n_components=2)
            coords_2d = pca.fit_transform(K_active)
            # Normalizace do [0, 1]
            coords_2d_norm = (coords_2d - coords_2d.min(axis=0)) / (coords_2d.max(axis=0) - coords_2d.min(axis=0) + 1e-8)
            
            # === 1. SÉMANTICKÁ MAPA (2D heatmapa) ===
            try:
                grid_x, grid_y = np.mgrid[0:1:100j, 0:1:100j]
                grid_z = griddata(coords_2d_norm, h_active, (grid_x, grid_y), method='linear', fill_value=0)
                grid_z_smooth = gaussian_filter(grid_z, sigma=2)
                
                plt.figure(figsize=(10, 8))
                plt.imshow(grid_z_smooth.T, extent=(0,1,0,1), origin='lower', cmap='plasma')
                plt.colorbar(label='Intenzita')
                plt.scatter(coords_2d_norm[:, 0], coords_2d_norm[:, 1], c='white', s=8, alpha=0.6)
                plt.title(f'Sémantická Mapa - Step {step}\n({n_active} center, h_max={h_active.max():.1f})', fontsize=14)
                plt.xlabel('Semantic X')
                plt.ylabel('Semantic Y')
                plt.tight_layout()
                plt.savefig(timelapse_dir / f"map_{step_str}.png", dpi=150, bbox_inches='tight')
                plt.close()
            except Exception:
                pass
            
            # === 2. 3D KRAJINA ===
            if n_active >= 10:
                try:
                    fig = plt.figure(figsize=(12, 9))
                    ax = fig.add_subplot(111, projection='3d')
                    
                    ls = LightSource(azdeg=315, altdeg=45)
                    rgb = ls.shade(grid_z_smooth.T, cmap=plt.cm.plasma, vert_exag=0.5, blend_mode='soft')
                    
                    ax.plot_surface(
                        grid_x, grid_y, grid_z_smooth.T,
                        rstride=2, cstride=2,
                        facecolors=rgb,
                        linewidth=0,
                        antialiased=True,
                        shade=False
                    )
                    
                    ax.view_init(elev=50, azim=-45)
                    ax.set_zlim(0, max(h_active.max() * 1.2, 1))
                    ax.set_axis_off()
                    plt.title(f'Memory Landscape - Step {step}', fontsize=16, y=0.95)
                    plt.savefig(timelapse_dir / f"landscape_{step_str}.png", dpi=150, bbox_inches='tight', pad_inches=0)
                    plt.close()
                except Exception:
                    pass
            
            # === 3. CLUSTERMAP (matice podobnosti) ===
            if n_active >= 10 and n_active <= 500:  # Limit pro rychlost
                try:
                    # Kosinová podobnost
                    norms = np.linalg.norm(K_active, axis=1, keepdims=True)
                    K_norm = K_active / (norms + 1e-8)
                    sim_matrix = np.dot(K_norm, K_norm.T)
                    
                    plt.style.use('dark_background')
                    g = sns.clustermap(
                        sim_matrix,
                        cmap='magma',
                        figsize=(10, 10),
                        xticklabels=False,
                        yticklabels=False,
                        dendrogram_ratio=(.1, .1),
                        cbar_pos=(0.02, 0.8, 0.03, 0.15)
                    )
                    g.ax_heatmap.set_title(f"Sémantická Podobnost - Step {step}\n({n_active} center)", fontsize=14, pad=20)
                    g.savefig(timelapse_dir / f"clustermap_{step_str}.png", dpi=150)
                    plt.close()
                    plt.style.use('default')
                except Exception:
                    plt.style.use('default')
            
            # === 4. DENDROGRAM ===
            if n_active >= 10 and n_active <= 500:
                try:
                    plt.style.use('dark_background')
                    plt.figure(figsize=(12, 6))
                    
                    Z = linkage(K_active, method='ward')
                    dendrogram(
                        Z,
                        leaf_rotation=90.,
                        no_labels=True,
                        color_threshold=Z[-10, 2] if n_active > 10 else None
                    )
                    
                    plt.title(f'Hierarchický Strom - Step {step}', fontsize=14)
                    plt.xlabel(f'{n_active} center')
                    plt.ylabel('Ward distance')
                    plt.grid(True, alpha=0.1, axis='y')
                    plt.tight_layout()
                    plt.savefig(timelapse_dir / f"dendrogram_{step_str}.png", dpi=150, bbox_inches='tight')
                    plt.close()
                    plt.style.use('default')
                except Exception:
                    plt.style.use('default')
            
        except ImportError:
            pass  # Chybí závislosti
        except Exception as e:
            print(f"  ⚠ Timelapse vizualizace selhala: {e}")
    
    def test_retention(self, scenario: StressTestScenario, n_test: int = 100):
        """
        Test retence: Jak dobře se paměť vybavuje po čase?
        
        POZNÁMKA: Tento test vyžaduje embeddingy v paměti, což není kompatibilní
        s lazy generation. Pro retention test použijte vizualizaci metrik.
        """
        print("\n" + "="*60)
        print("RETENTION TEST")
        print("="*60)
        print("⚠ Skipped (not compatible with lazy generation)")
        print("   Use visualize_stress_test.py to analyze retention from CSV")
        
        return [], 0.0
    
    def test_interference(self, scenario: StressTestScenario, n_test: int = 100):
        """
        Test interference: Vybavují se správné clustery?
        
        POZNÁMKA: Tento test vyžaduje embeddingy v paměti, což není kompatibilní
        s lazy generation. Pro interference analýzu použijte risk_report.
        """
        print("\n" + "="*60)
        print("INTERFERENCE TEST")
        print("="*60)
        print("⚠ Skipped (not compatible with lazy generation)")
        print("   Use analyze_memory_risks.py for interference analysis")
        
        return None, None
    
    def save_results(self, filename: str = "metrics.csv"):
        """Uloží metriky do CSV."""
        csv_path = self.output_dir / filename
        
        if not self.metrics:
            print("⚠ No metrics to save")
            return
        
        keys = self.metrics[0].keys()
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.metrics)
        
        print(f"✓ Metrics saved to {csv_path}")
        
        # Konsolidační eventy
        if self.consolidation_events:
            consol_path = self.output_dir / "consolidation_events.json"
            with open(consol_path, 'w') as f:
                json.dump(self.consolidation_events, f, indent=2)
            print(f"✓ Consolidation events saved to {consol_path}")


def main():
    """
    Hlavní entry point.
    
    OPRAVENO 2026-01-14:
    - Počítá INTERAKCE (ne tokeny!)
    - 50 interakcí/den = standard (kalibrace koeficientů)
    - Každá interakce = ~100 tokenů (realistické)
    """
    print("="*60)
    # Konfigurace - POUŽÍVÁME PRODUKČNÍ DEFAULTS (viz config.py)
    # Pouze mírně upravíme parametry specifické pro tento dlouhý test
    config = MemoryConfig(
        d_model=256,
        terrain_resolution=32,
        n_ltm_centers=2048,
        n_stm_centers=512,
        
        # Ostatní parametry se nyní berou z config.py (AGRESIVNÍ PRODUKČNÍ NASTAVENÍ)
    )
    
    print(f"\n⚡ FULL STRESS TEST configuration:")
    print(f"   Terrain: {config.terrain_resolution}³ grid")
    print(f"   LTM Centers: {config.n_ltm_centers}")
    print(f"   New Center Threshold: {config.ltm_new_center_threshold} (Should be 0.8)")
    print(f"   Write Sigma: {config.ltm_sigma_write} (Should be 0.15)")
    print(f"   Estimated RAM: ~500 MB - 2 GB\n")
    
    # Runner
    runner = StressTestRunner(config, output_dir="stress_test_results")
    
    # Scénáře
    if REALISTIC_SCENARIOS_AVAILABLE:
        print("✅ Using ULTRA-REALISTIC scenario:")
        scenarios = [
            RealisticMixedScenario(tokens_per_interaction=100),
        ]
    else:
        scenarios = [ClusteredScenario()]
    
    # DLOUHÝ TEST PRO FINÁLNÍ VALIDACI
    n_interactions = 9000  # = půl roku reálného provozu
    tokens_per_interaction = 100
    days = n_interactions / 50
    
    print(f"\n📅 Simulating: {n_interactions:,} interactions")
    print(f"   = {days:.0f} days")
    print(f"   = {days / 30:.1f} months")  
    print(f"   = {days / 365:.2f} years")
    print(f"   Total tokens: ~{n_interactions * tokens_per_interaction:,}")
    print(f"\n⏱️  Estimated time: ~20-30 minutes\n")
    
    for scenario in scenarios:
        # NOVÉ API: interakce místo kroků
        runner.run_scenario(
            scenario, 
            n_interactions=n_interactions,
            tokens_per_interaction=tokens_per_interaction,
            save_interval=500,  # Snapshot každých 500 interakcí (~10 dní)
            use_sequences=True
        )
        
        # Testy (info msg)
        print("\n🔍 Running retention test...")
        retention_results, retention_rate = runner.test_retention(scenario, n_test=100)
        
        print("\n🔍 Running interference test...")
        interference_results, avg_interference = runner.test_interference(scenario, n_test=100)
    
    # Konsolidační eventy
    if runner.consolidation_events:
        consol_path = runner.output_dir / "consolidation_events.json"
        with open(consol_path, 'w') as f:
            json.dump(runner.consolidation_events, f, indent=2)
        print(f"\n✓ Consolidation events saved to {consol_path}")
    
    print("\n" + "="*60)
    print("✓ Realistic stress test completed successfully")
    print(f"Results saved to: {runner.output_dir}")
    print("\nNext steps:")
    print("  python visualize_stress_test.py")
    print("  python analyze_memory_risks.py")
    print("="*60)


if __name__ == "__main__":
    main()

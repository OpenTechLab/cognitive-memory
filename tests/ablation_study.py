# ablation_study.py
"""
Ablační studie pro Cognitive Memory paper.

Spouští 3 ablace na RealisticMixed scénáři:
(A) Bez TerrainPrior (jen RBF na K)
(B) Bez difuze/homeostázy terénu (α=0, λ=0)
(C) Bez STM (jen LTM)

Plus baseline (plný systém) pro srovnání.

Každá ablace testuje jednu hypotézu:
- (A) TerrainPrior stabilizuje retrieval při driftu
- (B) Difuze + leak brání přepálení terénu
- (C) STM tlumí zápisový šum a chrání LTM

Výstupy:
- ablation_results.json - souhrnné metriky
- ablation_results.md - markdown tabulka pro paper

Autor: Michal Seidl / OpenTechLab
Datum: 2026-01-15
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import json
import csv
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from tqdm import tqdm

from cognitive_memory import (
    MemoryConfig,
    Terrain3D,
    MemoryCenters,
    MemoryWriter,
    SleepConsolidator,
    AutomaticConsolidator,
)

from realistic_scenarios import RealisticMixedScenario


@dataclass
class AblationResult:
    """Výsledky jednoho ablačního běhu."""
    name: str
    description: str
    hypothesis: str
    
    # Kapacita
    ltm_active_centers: int
    stm_active_centers: int
    new_centers_total: int
    
    # Konsolidace
    consolidation_events: int
    
    # Terén
    h_max: float
    h_mean: float
    h_std: float
    
    # Zápis
    omega_mean: float
    omega_max: float
    
    # Čas
    duration_seconds: float
    
    # Aha momenty
    aha_moments: int


class AblationRunner:
    """Runner pro jednu ablační konfiguraci."""
    
    def __init__(
        self,
        name: str,
        description: str,
        hypothesis: str,
        config: MemoryConfig,
        use_terrain_prior: bool = True,
        use_diffusion: bool = True,
        use_homeostasis: bool = True,
        use_stm: bool = True,
        device: str = "cpu"
    ):
        self.name = name
        self.description = description
        self.hypothesis = hypothesis
        self.config = config
        self.use_terrain_prior = use_terrain_prior
        self.use_diffusion = use_diffusion
        self.use_homeostasis = use_homeostasis
        self.use_stm = use_stm
        self.device = device
        
        # Komponenty
        self._setup_components()
    
    def _setup_components(self):
        """Inicializuje komponenty podle ablačních flagů."""
        config = self.config
        
        # LTM centra (vždy)
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
            device=self.device
        )
        
        # STM centra (podmíněně)
        if self.use_stm:
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
                device=self.device
            )
        else:
            # Dummy STM (nikdy se nepoužije)
            self.stm_centers = MemoryCenters(
                n_centers=16,  # Minimální
                d_key=config.d_stm_key,
                d_value=config.d_memory_value,
                sigma_read=config.stm_sigma_read,
                sigma_write=config.stm_sigma_write,
                leak=1.0,  # Okamžitý decay
                device=self.device
            )
        
        # LTM terén (s ablací difuze/homeostázy)
        ltm_alpha_h = config.terrain_ltm_alpha_h if self.use_diffusion else 0.0
        ltm_alpha_e = config.terrain_ltm_alpha_e if self.use_diffusion else 0.0
        ltm_leak = config.terrain_ltm_lambda if self.use_homeostasis else 0.0
        
        self.ltm_terrain = Terrain3D(
            resolution=config.terrain_resolution,
            alpha_h=ltm_alpha_h,
            alpha_e=ltm_alpha_e,
            leak=ltm_leak,
            device=self.device
        )
        
        # STM terén
        stm_alpha_h = config.terrain_stm_alpha_h if self.use_diffusion else 0.0
        stm_alpha_e = config.terrain_stm_alpha_e if self.use_diffusion else 0.0
        stm_leak = config.terrain_stm_lambda if self.use_homeostasis else 0.0
        
        self.stm_terrain = Terrain3D(
            resolution=config.terrain_resolution,
            alpha_h=stm_alpha_h,
            alpha_e=stm_alpha_e,
            leak=stm_leak,
            device=self.device
        )
        
        # Writer
        self.writer = MemoryWriter(
            d_model=config.d_model,
            d_ltm_key=config.d_memory_key,
            d_stm_key=config.d_stm_key,
            d_value=config.d_memory_value,
            segment_size=config.segment_size,
            write_strength_base=config.write_strength_base,
            write_novelty_weight=config.write_novelty_weight,
            write_surprise_weight=config.write_surprise_weight,
            write_emotion_weight=config.write_emotion_weight,
            write_bias=config.write_bias,
            # Terrain boost = 0 pokud bez TerrainPrior
            terrain_boost=config.write_terrain_boost if self.use_terrain_prior else 0.0,
        )
        
        # Konsolidátor (jen pokud STM)
        if self.use_stm:
            sleep_consolidator = SleepConsolidator(
                d_stm_key=config.d_stm_key,
                d_ltm_key=config.d_memory_key,
                fatigue_threshold=config.fatigue_threshold,
                fatigue_leak=config.fatigue_leak,
                consolidation_kappa=config.consolidation_kappa,
            )
            self.consolidator = AutomaticConsolidator(
                sleep_consolidator,
                min_interval=50
            )
        else:
            self.consolidator = None
        
        # Statistiky
        self.consolidation_events = 0
        self.new_centers_total = 0
        self.aha_moments = 0
        self.omega_values = []
    
    @torch.no_grad()
    def run(self, n_interactions: int = 3000) -> AblationResult:
        """Spustí ablační test."""
        import time
        start_time = time.time()
        
        # Scénář
        scenario = RealisticMixedScenario(tokens_per_interaction=100)
        scenario.prepare(n_interactions * 100, self.config.d_model)
        
        print(f"\n{'='*60}")
        print(f"ABLATION: {self.name}")
        print(f"Description: {self.description}")
        print(f"Hypothesis: {self.hypothesis}")
        print(f"Interactions: {n_interactions}")
        print(f"{'='*60}")
        
        for interaction in tqdm(range(n_interactions), desc=f"[{self.name}]"):
            # Generuj sekvenci
            if hasattr(scenario, 'generate_sequence'):
                sequence, meta = scenario.generate_sequence(interaction)
                hidden_states = sequence.unsqueeze(0).to(self.device)
            else:
                emb, meta = scenario.generate_embedding(interaction)
                hidden_states = emb.unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Aha moment
            if meta.get('is_aha_moment', False):
                self.aha_moments += 1
            
            # Zápis do paměti
            # Pro NoSTM použijeme standardní threshold pro LTM ale ignorujeme STM
            ltm_thresh = self.config.ltm_new_center_threshold
            stm_thresh = 0.3 if self.use_stm else 0.99
            
            stats = self.writer.write_to_memory(
                hidden_states=hidden_states,
                emotions=meta["emotions"].to(self.device),
                ltm_centers=self.ltm_centers,
                stm_centers=self.stm_centers,
                ltm_terrain=self.ltm_terrain,
                stm_terrain=self.stm_terrain,
                surprise=torch.full((1, hidden_states.shape[1]), meta["surprise"], device=self.device),
                ltm_threshold=ltm_thresh,
                stm_threshold=stm_thresh
            )
            
            self.new_centers_total += stats.get('new_ltm_centers', 0)
            self.omega_values.append(stats.get('omega_mean', 0))
            
            # Homeostáza (pokud povolena)
            if self.use_homeostasis:
                self.ltm_centers.homeostasis_step()
                if self.use_stm:
                    self.stm_centers.homeostasis_step()
            
            # Terén step (difuze + homeostáza)
            self.ltm_terrain.step()
            if self.use_stm:
                self.stm_terrain.step()
            
            # Konsolidace
            if self.consolidator is not None:
                consolidation_stats = self.consolidator.step(
                    stats.get('omega_mean', 0),
                    self.stm_centers,
                    self.ltm_centers,
                    self.stm_terrain,
                    self.ltm_terrain
                )
                if consolidation_stats is not None:
                    self.consolidation_events += 1
        
        duration = time.time() - start_time
        
        # Statistiky
        ltm_stats = self.ltm_centers.get_stats()
        terrain_stats = self.ltm_terrain.get_stats()
        
        result = AblationResult(
            name=self.name,
            description=self.description,
            hypothesis=self.hypothesis,
            ltm_active_centers=ltm_stats['n_active'],
            stm_active_centers=self.stm_centers.get_n_active() if self.use_stm else 0,
            new_centers_total=self.new_centers_total,
            consolidation_events=self.consolidation_events,
            h_max=terrain_stats['H_max'],
            h_mean=terrain_stats['H_mean'],
            h_std=terrain_stats['H_std'],
            omega_mean=float(np.mean(self.omega_values)) if self.omega_values else 0.0,
            omega_max=float(np.max(self.omega_values)) if self.omega_values else 0.0,
            duration_seconds=duration,
            aha_moments=self.aha_moments
        )
        
        print(f"\n✓ Ablation '{self.name}' completed")
        print(f"  LTM centers: {result.ltm_active_centers}")
        print(f"  STM centers: {result.stm_active_centers}")
        print(f"  Consolidations: {result.consolidation_events}")
        print(f"  h_max: {result.h_max:.2f}")
        print(f"  Duration: {result.duration_seconds:.1f}s")
        
        return result


def create_ablation_configs() -> List[Dict]:
    """Vytvoří konfigurace pro všechny ablace."""
    
    base_config = MemoryConfig(
        d_model=256,
        terrain_resolution=32,
        n_ltm_centers=2048,
        n_stm_centers=512,
    )
    
    return [
        # BASELINE (plný systém)
        {
            "name": "Baseline",
            "description": "Plný systém (TerrainPrior + Diffusion + STM)",
            "hypothesis": "Referenční konfigurace pro srovnání",
            "config": base_config,
            "use_terrain_prior": True,
            "use_diffusion": True,
            "use_homeostasis": True,
            "use_stm": True,
        },
        
        # (A) Bez TerrainPrior
        {
            "name": "NoTerrainPrior",
            "description": "Bez TerrainPrior (jen RBF nad K)",
            "hypothesis": "TerrainPrior stabilizuje retrieval při driftu a snižuje false recall",
            "config": base_config,
            "use_terrain_prior": False,
            "use_diffusion": True,
            "use_homeostasis": True,
            "use_stm": True,
        },
        
        # (B) Bez difuze a homeostázy terénu
        {
            "name": "NoDiffusion",
            "description": "Bez difuze/homeostázy terénu (α=0, λ=0)",
            "hypothesis": "Difuze + leak brání přepálení (lokální maxima) a drží prior hladký",
            "config": base_config,
            "use_terrain_prior": True,
            "use_diffusion": False,
            "use_homeostasis": False,
            "use_stm": True,
        },
        
        # (C) Bez STM
        {
            "name": "NoSTM",
            "description": "Bez STM (přímý zápis do LTM)",
            "hypothesis": "STM tlumí zápisový šum a chrání LTM; konsolidace vybírá stabilní stopy",
            "config": base_config,
            "use_terrain_prior": True,
            "use_diffusion": True,
            "use_homeostasis": True,
            "use_stm": False,
        },
    ]


def generate_markdown_table(results: List[AblationResult]) -> str:
    """Generuje markdown tabulku pro paper."""
    
    # Najdi baseline pro delta výpočty
    baseline = next((r for r in results if r.name == "Baseline"), results[0])
    
    lines = [
        "# Ablační studie - Výsledky",
        "",
        f"**Datum:** {datetime.now().strftime('%Y-%m-%d')}",
        f"**Scénář:** RealisticMixed (3000 interakcí = ~60 dní)",
        "",
        "## Souhrn",
        "",
        "| Konfigurace | LTM Centers | h_max | Consolidations | Δ od Baseline |",
        "|-------------|-------------|-------|----------------|---------------|",
    ]
    
    for r in results:
        delta_ltm = r.ltm_active_centers - baseline.ltm_active_centers
        delta_sign = "+" if delta_ltm >= 0 else ""
        delta_str = f"{delta_sign}{delta_ltm}" if r.name != "Baseline" else "—"
        
        lines.append(
            f"| **{r.name}** | {r.ltm_active_centers} | {r.h_max:.1f} | "
            f"{r.consolidation_events} | {delta_str} |"
        )
    
    lines.extend([
        "",
        "## Detailní výsledky",
        "",
    ])
    
    for r in results:
        lines.extend([
            f"### {r.name}",
            "",
            f"**Popis:** {r.description}",
            "",
            f"**Hypotéza:** {r.hypothesis}",
            "",
            "| Metrika | Hodnota |",
            "|---------|---------|",
            f"| LTM Active Centers | {r.ltm_active_centers} |",
            f"| STM Active Centers | {r.stm_active_centers} |",
            f"| New Centers Total | {r.new_centers_total} |",
            f"| Consolidation Events | {r.consolidation_events} |",
            f"| h_max | {r.h_max:.2f} |",
            f"| h_mean | {r.h_mean:.2f} |",
            f"| h_std | {r.h_std:.2f} |",
            f"| ω_mean | {r.omega_mean:.4f} |",
            f"| ω_max | {r.omega_max:.4f} |",
            f"| Aha Moments | {r.aha_moments} |",
            f"| Duration | {r.duration_seconds:.1f}s |",
            "",
        ])
    
    lines.extend([
        "## Interpretace",
        "",
        "### (A) NoTerrainPrior",
        "",
        "Očekávaný dopad: podobný počet center, ale horší stabilita retrieval při driftu.",
        "",
        "### (B) NoDiffusion", 
        "",
        "Očekávaný dopad: terén se fragmentuje do ostrých špiček (vyšší h_max, h_std).",
        "",
        "### (C) NoSTM",
        "",
        "Očekávaný dopad: rychlejší růst LTM center, horší kapacitní efektivita.",
        "",
        "---",
        "",
        "*Ablace nejsou kandidátní deploy konfigurace, ale kauzální analýza přínosu modulů.*",
    ])
    
    return "\n".join(lines)


def main():
    """Hlavní entry point."""
    import gc
    
    print("="*60)
    print("🔬 ABLATION STUDY - Cognitive Memory")
    print("="*60)
    print()
    print("Konfigurace:")
    print("  - 4 běhy (Baseline + 3 ablace)")
    print("  - 3000 interakcí/běh (~60 dní simulace)")
    print("  - RealisticMixed scénář (500 témat)")
    print()
    
    # Výstupní adresář
    output_dir = Path("ablation_results")
    output_dir.mkdir(exist_ok=True)
    
    # Konfigurace
    configs = create_ablation_configs()
    results: List[AblationResult] = []
    
    # Spusť ablace
    for i, cfg in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Starting: {cfg['name']}")
        
        runner = AblationRunner(
            name=cfg["name"],
            description=cfg["description"],
            hypothesis=cfg["hypothesis"],
            config=cfg["config"],
            use_terrain_prior=cfg["use_terrain_prior"],
            use_diffusion=cfg["use_diffusion"],
            use_homeostasis=cfg["use_homeostasis"],
            use_stm=cfg["use_stm"],
        )
        
        result = runner.run(n_interactions=3000)
        results.append(result)
        
        # Cleanup
        del runner
        gc.collect()
    
    # Ulož výsledky
    print("\n" + "="*60)
    print("📊 SAVING RESULTS")
    print("="*60)
    
    # JSON
    json_path = output_dir / "ablation_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)
    print(f"✓ JSON saved to: {json_path}")
    
    # Markdown
    md_content = generate_markdown_table(results)
    md_path = output_dir / "ablation_results.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"✓ Markdown saved to: {md_path}")
    
    # CSV
    csv_path = output_dir / "ablation_results.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    print(f"✓ CSV saved to: {csv_path}")
    
    # Souhrn
    print("\n" + "="*60)
    print("📋 SUMMARY")
    print("="*60)
    print()
    print(f"{'Config':<18} {'LTM':>8} {'STM':>8} {'h_max':>8} {'Consol':>8}")
    print("-" * 54)
    for r in results:
        print(f"{r.name:<18} {r.ltm_active_centers:>8} {r.stm_active_centers:>8} "
              f"{r.h_max:>8.1f} {r.consolidation_events:>8}")
    
    print()
    print("="*60)
    print("✓ Ablation study completed successfully!")
    print(f"  Results saved to: {output_dir.absolute()}")
    print("="*60)


if __name__ == "__main__":
    main()

# ablation_retrieval_test.py
"""
Retrieval Benchmark pro TerrainPrior ablaci.

Testuje hypotézu: TerrainPrior stabilizuje retrieval při driftu a snižuje false recall.

Metodika:
1. Fáze zápisu: Vložíme vzpomínky z N různých témat
2. Fáze driftu: Simulujeme průchod času (homeostáza bez zápisů)
3. Fáze retrieval: Testujeme vybavování vzpomínek pomocí probe queries

Metriky:
- Retrieval Accuracy: % správně identifikovaných témat
- False Recall Rate: % kdy se vybavilo špatné téma s vysokou confidencí
- Top-1 Accuracy: Zda nejsilnější centrum odpovídá správnému tématu
- Weight Distribution: Jak jsou váhy rozloženy mezi správné/špatné centra

Srovnání: Baseline vs NoTerrainPrior

Autor: BioCortexAI
Datum: 2026-01-15
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
from tqdm import tqdm

from cognitive_memory import (
    MemoryConfig,
    Terrain3D,
    MemoryCenters,
    MemoryWriter,
)
from cognitive_memory.terrain_prior import TerrainPrior
from cognitive_memory.projections import ProjectionBundle


@dataclass
class RetrievalResult:
    """Výsledky retrieval testu."""
    name: str
    description: str
    
    # Přesnost
    top1_accuracy: float          # Nejsilnější centrum = správné téma
    top3_accuracy: float          # Správné téma v top 3
    top5_accuracy: float          # Správné téma v top 5
    
    # False recall
    false_recall_rate: float      # % špatných témat s confidencí > 0.3
    max_false_confidence: float   # Nejvyšší confidence pro špatné téma
    
    # Distribuce vah
    correct_weight_mean: float    # Průměrná váha na správném tématu
    correct_weight_std: float     # Std váhy na správném tématu
    
    # Po driftu
    post_drift_top1: float        # Top-1 accuracy po driftu
    post_drift_correct_weight: float  # Správná váha po driftu
    
    # Čas
    duration_seconds: float


class RetrievalBenchmark:
    """Benchmark pro srovnání retrieval s/bez TerrainPrior."""
    
    def __init__(
        self,
        config: MemoryConfig,
        use_terrain_prior: bool = True,
        n_topics: int = 20,
        samples_per_topic: int = 10,
        device: str = "cpu"
    ):
        self.config = config
        self.use_terrain_prior = use_terrain_prior
        self.n_topics = n_topics
        self.samples_per_topic = samples_per_topic
        self.device = device
        
        self._setup_components()
    
    def _setup_components(self):
        """Inicializace komponent."""
        config = self.config
        
        # LTM centra
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
        
        # LTM terén
        self.ltm_terrain = Terrain3D(
            resolution=config.terrain_resolution,
            alpha_h=config.terrain_ltm_alpha_h,
            alpha_e=config.terrain_ltm_alpha_e,
            leak=config.terrain_ltm_lambda,
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
            terrain_boost=config.write_terrain_boost if self.use_terrain_prior else 0.0,
        )
        
        # TerrainPrior (podmíněně)
        if self.use_terrain_prior:
            self.terrain_prior = TerrainPrior(
                d_key=config.d_memory_key,
                d_emotion=config.d_emotion,
                beta=config.terrain_prior_beta,
                gate_bias=config.terrain_prior_gate_bias
            )
        else:
            self.terrain_prior = None
        
        # Projekce pro dotazy
        self.proj = ProjectionBundle(
            d_model=config.d_model,
            d_ltm_key=config.d_memory_key,
            d_stm_key=config.d_stm_key,
            d_value=config.d_memory_value
        )
        
        # Dummy STM (unused but required by writer)
        self.stm_centers = MemoryCenters(
            n_centers=16,
            d_key=config.d_stm_key,
            d_value=config.d_memory_value,
            leak=1.0,  # Fast decay
            device=self.device
        )
        self.stm_terrain = Terrain3D(
            resolution=16,
            alpha_h=0.02,
            alpha_e=0.01,
            leak=0.01,
            device=self.device
        )
    
    def _generate_topic_data(self) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Generuje data pro N témat.
        
        Returns:
            topic_centers: [N, d_model] - centra témat
            samples: [N * samples_per_topic, d_model] - vzorky
            labels: [N * samples_per_topic] - ID témat
        """
        d_model = self.config.d_model
        
        # Vytvoř distinktní centra témat
        topic_centers = torch.randn(self.n_topics, d_model)
        topic_centers = F.normalize(topic_centers, dim=-1)
        
        samples = []
        labels = []
        
        for topic_id in range(self.n_topics):
            center = topic_centers[topic_id]
            
            for _ in range(self.samples_per_topic):
                # Vzorek blízko centra
                noise = torch.randn(d_model) * 0.2
                sample = center + noise
                sample = F.normalize(sample, dim=-1)
                samples.append(sample)
                labels.append(topic_id)
        
        samples = torch.stack(samples)
        return topic_centers, samples, labels
    
    def _write_samples(self, samples: torch.Tensor, labels: List[int]):
        """Zapíše vzorky do paměti."""
        for i, (sample, label) in enumerate(zip(samples, labels)):
            # Emotions based on topic
            emotions = torch.ones(4)
            emotions[label % 4] += 0.5
            
            # Surprise based on novelty
            surprise = 0.5 if i < self.n_topics else 0.2
            
            hidden_states = sample.unsqueeze(0).unsqueeze(0)  # [1, 1, D]
            
            self.writer.write_to_memory(
                hidden_states=hidden_states,
                emotions=emotions,
                ltm_centers=self.ltm_centers,
                stm_centers=self.stm_centers,
                ltm_terrain=self.ltm_terrain,
                stm_terrain=self.stm_terrain,
                surprise=torch.full((1, 1), surprise),
                ltm_threshold=self.config.ltm_new_center_threshold,
                stm_threshold=0.99
            )
            
            # Homeostáza
            self.ltm_centers.homeostasis_step()
            self.ltm_terrain.step()
    
    def _retrieve_with_query(
        self, 
        query: torch.Tensor,
        correct_topic: int
    ) -> Dict:
        """
        Provede retrieval a vrátí metriky.
        
        Args:
            query: [d_model] dotaz
            correct_topic: ID správného tématu
        """
        # Projekce do LTM space
        q_ltm = self.proj.project_to_ltm(query.unsqueeze(0).unsqueeze(0))  # [1, 1, 64]
        
        # Aplikuj TerrainPrior pokud je povolený
        if self.terrain_prior is not None:
            q_tilde, g_prior, _ = self.terrain_prior(q_ltm, self.ltm_terrain)
        else:
            q_tilde = q_ltm
            g_prior = torch.zeros(1, 1, 1)
        
        # Čtení z paměti
        r_V, r_E, weights, indices = self.ltm_centers.read(
            q_tilde, 
            top_k=min(32, self.ltm_centers.get_n_active())
        )
        
        # Analýza výsledků
        weights = weights.squeeze()  # [top_k]
        indices = indices.squeeze()  # [top_k]
        
        # Získej klíče vybraných center
        K = self.ltm_centers.K[indices]  # [top_k, 64]
        
        # Pro každé centrum urči téma (podle podobnosti s topic centers)
        # Potřebujeme promapovat zpět - použijeme projection bundle
        
        return {
            "weights": weights.detach(),
            "indices": indices.detach(),
            "g_prior": g_prior.squeeze().item() if g_prior.numel() == 1 else 0.0,
            "total_weight": weights.sum().item(),
            "top1_weight": weights[0].item() if len(weights) > 0 else 0.0,
        }
    
    @torch.no_grad()
    def run(self, drift_steps: int = 500) -> RetrievalResult:
        """
        Spustí retrieval benchmark.
        
        Args:
            drift_steps: Počet kroků homeostázy mezi zápisem a retrieval
        """
        import time
        start_time = time.time()
        
        name = "Baseline" if self.use_terrain_prior else "NoTerrainPrior"
        print(f"\n{'='*60}")
        print(f"RETRIEVAL BENCHMARK: {name}")
        print(f"Topics: {self.n_topics}, Samples/topic: {self.samples_per_topic}")
        print(f"Drift steps: {drift_steps}")
        print(f"{'='*60}")
        
        # 1. Generuj data
        print("\n📊 Generating topic data...")
        topic_centers, samples, labels = self._generate_topic_data()
        
        # 2. Zapiš vzorky do paměti
        print("📝 Writing samples to memory...")
        self._write_samples(samples, labels)
        print(f"   LTM centers: {self.ltm_centers.get_n_active()}")
        
        # 3. Retrieval PŘED driftem
        print("\n🔍 Testing retrieval BEFORE drift...")
        pre_drift_results = self._test_retrieval(topic_centers, labels)
        
        # 4. Simuluj drift (homeostáza bez zápisů)
        print(f"\n⏳ Simulating {drift_steps} drift steps...")
        for _ in tqdm(range(drift_steps), desc="Drift"):
            self.ltm_centers.homeostasis_step()
            self.ltm_terrain.step()
        
        # 5. Retrieval PO driftu
        print("\n🔍 Testing retrieval AFTER drift...")
        post_drift_results = self._test_retrieval(topic_centers, labels)
        
        duration = time.time() - start_time
        
        result = RetrievalResult(
            name=name,
            description=f"{'With' if self.use_terrain_prior else 'Without'} TerrainPrior",
            top1_accuracy=pre_drift_results["top1_accuracy"],
            top3_accuracy=pre_drift_results["top3_accuracy"],
            top5_accuracy=pre_drift_results["top5_accuracy"],
            false_recall_rate=pre_drift_results["false_recall_rate"],
            max_false_confidence=pre_drift_results["max_false_confidence"],
            correct_weight_mean=pre_drift_results["correct_weight_mean"],
            correct_weight_std=pre_drift_results["correct_weight_std"],
            post_drift_top1=post_drift_results["top1_accuracy"],
            post_drift_correct_weight=post_drift_results["correct_weight_mean"],
            duration_seconds=duration
        )
        
        print(f"\n✓ Benchmark '{name}' completed")
        print(f"  Top-1 Accuracy: {result.top1_accuracy:.1%}")
        print(f"  Post-drift Top-1: {result.post_drift_top1:.1%}")
        print(f"  False Recall Rate: {result.false_recall_rate:.1%}")
        
        return result
    
    def _test_retrieval(
        self, 
        topic_centers: torch.Tensor,
        labels: List[int]
    ) -> Dict:
        """
        Testuje retrieval pro všechna témata.
        """
        top1_correct = 0
        top3_correct = 0
        top5_correct = 0
        false_recalls = 0
        max_false_confidence = 0.0
        correct_weights = []
        
        n_queries = self.n_topics
        
        for topic_id in range(self.n_topics):
            # Probe query = centrum tématu
            query = topic_centers[topic_id]
            
            # Projekce do LTM space
            q_ltm = self.proj.project_to_ltm(query.unsqueeze(0).unsqueeze(0))
            
            # TerrainPrior
            if self.terrain_prior is not None:
                q_tilde, g_prior, _ = self.terrain_prior(q_ltm, self.ltm_terrain)
            else:
                q_tilde = q_ltm
            
            # Čtení z paměti
            n_active = self.ltm_centers.get_n_active()
            if n_active == 0:
                continue
            
            top_k = min(32, n_active)
            r_V, r_E, weights, indices = self.ltm_centers.read(q_tilde, top_k=top_k)
            
            weights = weights.squeeze()
            indices = indices.squeeze()
            
            if weights.dim() == 0:
                weights = weights.unsqueeze(0)
                indices = indices.unsqueeze(0)
            
            # Pro každé vrácené centrum urči téma
            # Mapování: porovnáme klíče center s projektovanými topic centry
            center_topics = []
            for idx in indices:
                center_key = self.ltm_centers.K[idx]  # [64]
                
                # Najdi nejbližší téma
                topic_queries = self.proj.project_to_ltm(topic_centers.unsqueeze(0)).squeeze(0)  # [N, 64]
                similarities = F.cosine_similarity(
                    center_key.unsqueeze(0), 
                    topic_queries, 
                    dim=-1
                )
                best_topic = similarities.argmax().item()
                center_topics.append(best_topic)
            
            # Top-1 accuracy
            if len(center_topics) > 0 and center_topics[0] == topic_id:
                top1_correct += 1
            
            # Top-3 accuracy
            if topic_id in center_topics[:3]:
                top3_correct += 1
            
            # Top-5 accuracy
            if topic_id in center_topics[:5]:
                top5_correct += 1
            
            # Váha na správném tématu
            correct_weight = 0.0
            for i, (t, w) in enumerate(zip(center_topics, weights)):
                if t == topic_id:
                    correct_weight += w.item()
            correct_weights.append(correct_weight)
            
            # False recall: vysoká confidence na špatném tématu
            for i, (t, w) in enumerate(zip(center_topics, weights)):
                if t != topic_id and w.item() > 0.3:
                    false_recalls += 1
                    max_false_confidence = max(max_false_confidence, w.item())
                    break  # Počítáme jen jednou za query
        
        return {
            "top1_accuracy": top1_correct / n_queries if n_queries > 0 else 0,
            "top3_accuracy": top3_correct / n_queries if n_queries > 0 else 0,
            "top5_accuracy": top5_correct / n_queries if n_queries > 0 else 0,
            "false_recall_rate": false_recalls / n_queries if n_queries > 0 else 0,
            "max_false_confidence": max_false_confidence,
            "correct_weight_mean": np.mean(correct_weights) if correct_weights else 0,
            "correct_weight_std": np.std(correct_weights) if correct_weights else 0,
        }


def main():
    """Hlavní entry point."""
    print("="*60)
    print("🔬 RETRIEVAL BENCHMARK - TerrainPrior Ablation")
    print("="*60)
    
    config = MemoryConfig(
        d_model=256,
        terrain_resolution=32,
        n_ltm_centers=1024,  # Větší kapacita
    )
    
    output_dir = Path("ablation_results")
    output_dir.mkdir(exist_ok=True)
    
    # Spustíme 3 opakování pro statistickou významnost
    n_runs = 3
    all_results = {"Baseline": [], "NoTerrainPrior": []}
    
    for run in range(n_runs):
        print(f"\n{'='*60}")
        print(f"RUN {run+1}/{n_runs}")
        print(f"{'='*60}")
        
        # Test s TerrainPrior (Baseline)
        benchmark_baseline = RetrievalBenchmark(
            config=config,
            use_terrain_prior=True,
            n_topics=30,           # Více témat
            samples_per_topic=15   # Více vzorků
        )
        result_baseline = benchmark_baseline.run(drift_steps=500)
        all_results["Baseline"].append(result_baseline)
        
        # Test bez TerrainPrior
        benchmark_no_prior = RetrievalBenchmark(
            config=config,
            use_terrain_prior=False,
            n_topics=30,
            samples_per_topic=15
        )
        result_no_prior = benchmark_no_prior.run(drift_steps=500)
        all_results["NoTerrainPrior"].append(result_no_prior)
    
    # Průměrné výsledky
    def avg_results(results_list):
        return {
            "top1_accuracy": np.mean([r.top1_accuracy for r in results_list]),
            "top1_std": np.std([r.top1_accuracy for r in results_list]),
            "top3_accuracy": np.mean([r.top3_accuracy for r in results_list]),
            "false_recall_rate": np.mean([r.false_recall_rate for r in results_list]),
            "correct_weight_mean": np.mean([r.correct_weight_mean for r in results_list]),
            "post_drift_top1": np.mean([r.post_drift_top1 for r in results_list]),
            "post_drift_std": np.std([r.post_drift_top1 for r in results_list]),
        }
    
    avg_baseline = avg_results(all_results["Baseline"])
    avg_no_prior = avg_results(all_results["NoTerrainPrior"])
    
    results = [all_results["Baseline"][-1], all_results["NoTerrainPrior"][-1]]
    result_baseline = results[0]
    result_no_prior = results[1]
    
    # Výsledky
    print("\n" + "="*60)
    print("📊 RESULTS COMPARISON")
    print("="*60)
    print()
    print(f"{'Metric':<30} {'Baseline':>15} {'NoTerrainPrior':>15} {'Δ':>10}")
    print("-" * 70)
    
    metrics = [
        ("Top-1 Accuracy", "top1_accuracy", "%"),
        ("Top-3 Accuracy", "top3_accuracy", "%"),
        ("Top-5 Accuracy", "top5_accuracy", "%"),
        ("False Recall Rate", "false_recall_rate", "%"),
        ("Correct Weight Mean", "correct_weight_mean", ""),
        ("Post-drift Top-1", "post_drift_top1", "%"),
        ("Post-drift Weight", "post_drift_correct_weight", ""),
    ]
    
    for label, key, fmt in metrics:
        v1 = getattr(result_baseline, key)
        v2 = getattr(result_no_prior, key)
        delta = v2 - v1
        
        if fmt == "%":
            print(f"{label:<30} {v1:>14.1%} {v2:>14.1%} {delta:>+9.1%}")
        else:
            print(f"{label:<30} {v1:>14.3f} {v2:>14.3f} {delta:>+9.3f}")
    
    # Uložení
    json_path = output_dir / "retrieval_benchmark.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)
    print(f"\n✓ Results saved to: {json_path}")
    
    # Markdown pro paper
    md_lines = [
        "# Retrieval Benchmark - TerrainPrior Ablation",
        "",
        f"**Datum:** {datetime.now().strftime('%Y-%m-%d')}",
        f"**Konfigurace:** {config.n_ltm_centers} LTM centers, {20} topics, {10} samples/topic",
        "",
        "## Výsledky",
        "",
        "| Metrika | Baseline | NoTerrainPrior | Δ |",
        "|---------|----------|----------------|---|",
    ]
    
    for label, key, fmt in metrics:
        v1 = getattr(result_baseline, key)
        v2 = getattr(result_no_prior, key)
        delta = v2 - v1
        
        if fmt == "%":
            md_lines.append(f"| {label} | {v1:.1%} | {v2:.1%} | {delta:+.1%} |")
        else:
            md_lines.append(f"| {label} | {v1:.3f} | {v2:.3f} | {delta:+.3f} |")
    
    md_lines.extend([
        "",
        "## Interpretace",
        "",
        f"- **Top-1 Accuracy:** {'Baseline lepší' if result_baseline.top1_accuracy > result_no_prior.top1_accuracy else 'NoTerrainPrior lepší' if result_no_prior.top1_accuracy > result_baseline.top1_accuracy else 'Shodné'}",
        f"- **Post-drift degradace:** Baseline {result_baseline.top1_accuracy - result_baseline.post_drift_top1:.1%}, NoTerrainPrior {result_no_prior.top1_accuracy - result_no_prior.post_drift_top1:.1%}",
        f"- **False Recall:** {'Baseline nižší' if result_baseline.false_recall_rate < result_no_prior.false_recall_rate else 'NoTerrainPrior nižší' if result_no_prior.false_recall_rate < result_baseline.false_recall_rate else 'Shodné'}",
        "",
        "## Závěr",
        "",
    ])
    
    # Automatický závěr
    baseline_wins = 0
    no_prior_wins = 0
    
    if result_baseline.top1_accuracy > result_no_prior.top1_accuracy:
        baseline_wins += 1
    elif result_no_prior.top1_accuracy > result_baseline.top1_accuracy:
        no_prior_wins += 1
    
    if result_baseline.false_recall_rate < result_no_prior.false_recall_rate:
        baseline_wins += 1
    elif result_no_prior.false_recall_rate < result_baseline.false_recall_rate:
        no_prior_wins += 1
    
    drift_degrad_baseline = result_baseline.top1_accuracy - result_baseline.post_drift_top1
    drift_degrad_no_prior = result_no_prior.top1_accuracy - result_no_prior.post_drift_top1
    
    if drift_degrad_baseline < drift_degrad_no_prior:
        baseline_wins += 1
        md_lines.append("**TerrainPrior pomáhá:** Menší degradace po driftu.")
    elif drift_degrad_no_prior < drift_degrad_baseline:
        no_prior_wins += 1
        md_lines.append("**TerrainPrior nepomáhá:** Větší degradace po driftu s TerrainPrior.")
    else:
        md_lines.append("**Drift:** Shodná degradace.")
    
    if baseline_wins > no_prior_wins:
        md_lines.append(f"\n**Celkový závěr:** TerrainPrior přináší zlepšení ({baseline_wins}:{no_prior_wins}).")
    elif no_prior_wins > baseline_wins:
        md_lines.append(f"\n**Celkový závěr:** TerrainPrior nepřináší zlepšení ({baseline_wins}:{no_prior_wins}).")
    else:
        md_lines.append(f"\n**Celkový závěr:** Nerozhodný výsledek ({baseline_wins}:{no_prior_wins}).")
    
    md_path = output_dir / "retrieval_benchmark.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    print(f"✓ Markdown saved to: {md_path}")
    
    print("\n" + "="*60)
    print("✓ Retrieval benchmark completed!")
    print("="*60)


if __name__ == "__main__":
    main()

# memory_quality_test.py
"""
KVALITATIVNÍ TESTY COGNITIVE MEMORY

Měří skutečnou UŽITEČNOST paměti:
1. RETENTION TEST - Ukládám vzpomínku, čtu ji zpět po čase
2. RECONSTRUCTION TEST - Kvalita rekonstrukce z paměti
3. INTERFERENCE TEST - Prolínání nesouvisejících vzpomínek
4. CAPACITY TEST - Kolik DISTINKTNÍCH vzpomínek lze uložit
5. CONSOLIDATION TEST - Přežívají vzpomínky konsolidaci STM→LTM?

Výstupy:
- Kvantitativní skóre pro každý test
- Vizualizace paměťového prostoru
- Doporučení pro ladění
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

from cognitive_memory import (
    MemoryConfig,
    Terrain3D,
    MemoryCenters,
    MemoryWriter,
    SleepConsolidator,
    AutomaticConsolidator,
)


@dataclass
class MemoryItem:
    """Jedna vzpomínka pro testování."""
    id: int
    key: torch.Tensor       # Unikátní klíč [d_model]
    value: torch.Tensor     # Očekávaná hodnota [d_value]
    emotion: torch.Tensor   # Emoční kontext [4]
    topic_id: int           # Téma/cluster
    timestamp: int          # Kdy byla uložena


@dataclass
class RetrievalResult:
    """Výsledek vybavení vzpomínky."""
    query_id: int
    retrieved_value: torch.Tensor
    expected_value: torch.Tensor
    similarity: float       # Cosine similarity
    exact_match: bool       # Nad prahem
    interference_score: float  # Jak moc se míchají jiná témata


class MemoryQualityTester:
    """
    Komplexní tester kvality paměti.
    """
    
    def __init__(
        self,
        config: MemoryConfig = None,
        device: str = "cpu"
    ):
        self.config = config or MemoryConfig()
        self.device = device
        
        # Inicializace paměťových komponent
        self._init_memory()
        
        # Úložiště vzpomínek pro testování
        self.stored_memories: List[MemoryItem] = []
        self.retrieval_results: List[RetrievalResult] = []
        
    def _init_memory(self):
        """Inicializuje paměťové komponenty."""
        config = self.config
        
        self.ltm_centers = MemoryCenters(
            n_centers=config.n_ltm_centers,
            d_key=config.d_memory_key,
            d_value=config.d_memory_value,
            sigma_read=config.ltm_sigma_read,
            sigma_write=config.ltm_sigma_write,
            leak=config.ltm_leak,
            device=self.device
        )
        
        self.stm_centers = MemoryCenters(
            n_centers=config.n_stm_centers,
            d_key=config.d_stm_key,
            d_value=config.d_memory_value,
            sigma_read=config.stm_sigma_read,
            sigma_write=config.stm_sigma_write,
            leak=config.stm_leak,
            device=self.device
        )
        
        self.ltm_terrain = Terrain3D(config.terrain_resolution, device=self.device)
        self.stm_terrain = Terrain3D(config.terrain_resolution, device=self.device)
        
        self.writer = MemoryWriter(
            d_model=config.d_model,
            d_ltm_key=config.d_memory_key,
            d_stm_key=config.d_stm_key,
            d_value=config.d_memory_value,
            write_strength_base=config.write_strength_base,
            write_bias=config.write_bias,
        )
        
        self.consolidator = AutomaticConsolidator(
            SleepConsolidator(
                d_stm_key=config.d_stm_key,
                d_ltm_key=config.d_memory_key,
                fatigue_threshold=config.fatigue_threshold,
                consolidation_kappa=config.consolidation_kappa,
            ),
            min_interval=50
        )
    
    def create_distinct_memory(
        self,
        topic_id: int,
        memory_id: int,
        d_model: int = 256
    ) -> MemoryItem:
        """
        Vytvoří DISTINKTNÍ vzpomínku s unikátním vzorem.
        
        Klíč a hodnota jsou korelovány, aby bylo možné testovat rekonstrukci.
        """
        # Unikátní seed pro reprodukovatelnost
        torch.manual_seed(topic_id * 1000 + memory_id)
        
        # Klíč má strukturu: topic_base + memory_specific
        topic_base = torch.randn(d_model) 
        topic_base = topic_base / topic_base.norm()
        
        memory_specific = torch.randn(d_model) * 0.3
        key = F.normalize(topic_base + memory_specific, dim=-1)
        
        # Hodnota je ODVODITELNÁ z klíče (pro testování rekonstrukce)
        # Použijeme deterministickou transformaci
        value_seed = (key[:self.config.d_memory_value] + key[-self.config.d_memory_value:]) / 2
        value = F.normalize(value_seed, dim=-1)
        
        # Emoce korelují s tématem
        base_emotion = torch.ones(4)
        base_emotion[topic_id % 4] += 0.5  # Jedno dominantní
        emotion = torch.clamp(base_emotion + torch.randn(4) * 0.1, 0.5, 2.0)
        
        return MemoryItem(
            id=memory_id,
            key=key,
            value=value,
            emotion=emotion,
            topic_id=topic_id,
            timestamp=0
        )
    
    def store_memory(self, memory: MemoryItem, timestamp: int):
        """
        Uloží vzpomínku do paměti.
        """
        memory.timestamp = timestamp
        
        # Převod na batch formát
        hidden_states = memory.key.unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, d_model]
        
        # !!! OPRAVA LOGIKY TESTU !!!
        # MemoryWriter ignoruje původní memory.value a místo toho generuje 
        # hodnotu k uložení projekcí z hidden_states.
        # Musíme aktualizovat naše očekávání (memory.value), aby odpovídalo tomu,
        # co MemoryWriter skutečně uloží.
        
        with torch.no_grad():
            # Získáme projekci, kterou provádí writer interně
            projected_value = self.writer.proj.project_to_value(hidden_states) # [1, 1, d_value]
            
            # Aktualizujeme ground truth pro test
            memory.value = projected_value.squeeze(0).squeeze(0).detach()
            
            # Poznámka: Pokud by MemoryWriter dělal další normalizaci, měli bychom ji zde také provést.
            # Pro cosine similarity (v retrieve_memory) na škále nezáleží.
        
        # Síla zápisu - vysoká pro testování
        surprise = torch.tensor([[0.8]], device=self.device)
        
        stats = self.writer.write_to_memory(
            hidden_states=hidden_states,
            emotions=memory.emotion.to(self.device),
            ltm_centers=self.ltm_centers,
            stm_centers=self.stm_centers,
            ltm_terrain=self.ltm_terrain,
            stm_terrain=self.stm_terrain,
            surprise=surprise
        )
        
        self.stored_memories.append(memory)
        
        return stats
    
    def retrieve_memory(self, query_key: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Vybavení vzpomínky z paměti.
        
        Returns:
            (retrieved_value, retrieved_emotion, confidence)
        """
        # Projekce do LTM key space - OPRAVA: správný formát
        query_batch = query_key.unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, d_model]
        q_ltm = self.writer.proj.project_to_ltm(query_batch)  # [1, 1, d_ltm_key]
        
        # Čtení z LTM
        r_V, r_E, weights, indices = self.ltm_centers.read(
            q_ltm,  # [1, 1, d_key]
            top_k=8
        )
        
        # Confidence = suma vah (čím vyšší, tím jistější vybavení)
        confidence = weights.max().item() if weights.shape[-1] > 0 else 0.0
        
        return r_V.squeeze(0).squeeze(0), r_E.squeeze(0).squeeze(0), confidence
    
    def test_retention(
        self,
        n_memories: int = 50,
        n_topics: int = 10,
        delay_steps: int = 100
    ) -> Dict:
        """
        TEST 1: RETENTION
        
        Uloží vzpomínky, počká, pak je zkusí vybavit.
        Měří kolik z nich je stále dostupných.
        """
        print("\n" + "="*60)
        print("TEST 1: RETENTION (Uchování vzpomínek)")
        print("="*60)
        
        # Reset paměti
        self._init_memory()
        self.stored_memories = []
        
        # 1. Ulož vzpomínky
        print(f"\n📝 Ukládám {n_memories} vzpomínek ({n_topics} témat)...")
        
        for i in tqdm(range(n_memories), desc="Storing"):
            topic_id = i % n_topics
            memory = self.create_distinct_memory(topic_id, i, self.config.d_model)
            self.store_memory(memory, timestamp=i)
            
            # Homeostáza po každém zápisu
            self.ltm_centers.homeostasis_step()
            self.stm_centers.homeostasis_step()
        
        print(f"   LTM centers: {self.ltm_centers.get_n_active()}")
        print(f"   STM centers: {self.stm_centers.get_n_active()}")
        
        # 2. Simuluj čas (homeostáza bez zápisů)
        print(f"\n⏳ Simuluji {delay_steps} kroků bez zápisů...")
        for _ in tqdm(range(delay_steps), desc="Time passing"):
            self.ltm_centers.homeostasis_step()
            self.stm_centers.homeostasis_step()
        
        # 3. Zkus vybavit každou vzpomínku
        print(f"\n🔍 Vybavuji vzpomínky...")
        
        retrievals = []
        for memory in tqdm(self.stored_memories, desc="Retrieving"):
            r_V, r_E, confidence = self.retrieve_memory(memory.key)
            
            # Měř podobnost s očekávanou hodnotou
            if r_V.norm() > 1e-6 and memory.value.norm() > 1e-6:
                similarity = F.cosine_similarity(
                    r_V.unsqueeze(0), 
                    memory.value.to(self.device).unsqueeze(0)
                ).item()
            else:
                similarity = 0.0
            
            retrievals.append({
                "memory_id": memory.id,
                "topic_id": memory.topic_id,
                "similarity": similarity,
                "confidence": confidence,
                "age": delay_steps + (n_memories - memory.id)
            })
        
        # 4. Analýza výsledků
        similarities = [r["similarity"] for r in retrievals]
        confidences = [r["confidence"] for r in retrievals]
        
        # Práh pro "úspěšné vybavení"
        threshold = 0.5
        successful = sum(1 for s in similarities if s > threshold)
        
        results = {
            "n_memories": n_memories,
            "n_topics": n_topics,
            "delay_steps": delay_steps,
            "retention_rate": successful / n_memories,
            "avg_similarity": np.mean(similarities),
            "std_similarity": np.std(similarities),
            "avg_confidence": np.mean(confidences),
            "ltm_centers_final": self.ltm_centers.get_n_active(),
            "stm_centers_final": self.stm_centers.get_n_active(),
            "retrievals": retrievals
        }
        
        print(f"\n📊 VÝSLEDKY RETENTION TESTU:")
        print(f"   Retention rate: {results['retention_rate']:.1%}")
        print(f"   Průměrná podobnost: {results['avg_similarity']:.3f}")
        print(f"   Průměrná confidence: {results['avg_confidence']:.3f}")
        print(f"   LTM centers: {results['ltm_centers_final']}")
        
        return results
    
    def test_interference(
        self,
        n_topics: int = 10,
        memories_per_topic: int = 10
    ) -> Dict:
        """
        TEST 2: INTERFERENCE
        
        Měří, jak moc se vzpomínky z různých témat prolínají.
        """
        print("\n" + "="*60)
        print("TEST 2: INTERFERENCE (Prolínání vzpomínek)")
        print("="*60)
        
        # Reset paměti
        self._init_memory()
        self.stored_memories = []
        
        # Ulož vzpomínky ze všech témat
        print(f"\n📝 Ukládám {n_topics * memories_per_topic} vzpomínek...")
        
        memory_id = 0
        for topic in range(n_topics):
            for _ in range(memories_per_topic):
                memory = self.create_distinct_memory(topic, memory_id, self.config.d_model)
                self.store_memory(memory, timestamp=memory_id)
                memory_id += 1
        
        # Měř interference: dotaz na téma A, kolik z výsledku je z jiných témat
        print(f"\n🔍 Měřím interference...")
        
        interference_scores = []
        
        for topic in range(n_topics):
            # Vyber reprezentativní vzpomínku z tématu
            topic_memories = [m for m in self.stored_memories if m.topic_id == topic]
            query_memory = topic_memories[0]
            
            # Projekce dotazu
            q_ltm = self.writer.proj.project_to_ltm(
                query_memory.key.unsqueeze(0).unsqueeze(0)
            ).squeeze(0)
            
            # Najdi nejbližší centra
            weights, indices = self.ltm_centers.compute_rbf_weights(
                q_ltm.unsqueeze(0),
                top_k=16,
                normalize=True
            )
            
            if weights.shape[-1] == 0:
                interference_scores.append(0.0)
                continue
            
            # Zjisti, kolik center "nepatří" k tomuto tématu
            # (Toto je aproximace - nemáme přímé mapování center na témata)
            total_weight = weights.sum().item()
            interference_scores.append(1.0 - total_weight)  # Vyšší = více interference
        
        results = {
            "n_topics": n_topics,
            "memories_per_topic": memories_per_topic,
            "avg_interference": np.mean(interference_scores),
            "max_interference": np.max(interference_scores),
            "ltm_centers": self.ltm_centers.get_n_active(),
            "centers_per_topic": self.ltm_centers.get_n_active() / n_topics
        }
        
        print(f"\n📊 VÝSLEDKY INTERFERENCE TESTU:")
        print(f"   Průměrná interference: {results['avg_interference']:.3f}")
        print(f"   Maximální interference: {results['max_interference']:.3f}")
        print(f"   LTM centra: {results['ltm_centers']}")
        print(f"   Centra/téma: {results['centers_per_topic']:.1f}")
        
        return results
    
    def test_capacity(
        self,
        max_memories: int = 200,
        similarity_threshold: float = 0.3
    ) -> Dict:
        """
        TEST 3: CAPACITY
        
        Kolik DISTINKTNÍCH vzpomínek lze uložit před degradací?
        """
        print("\n" + "="*60)
        print("TEST 3: CAPACITY (Kapacita paměti)")
        print("="*60)
        
        # Reset paměti
        self._init_memory()
        self.stored_memories = []
        
        print(f"\n📝 Postupně ukládám až {max_memories} vzpomínek...")
        
        capacity_curve = []
        degradation_point = None
        
        for i in tqdm(range(max_memories), desc="Testing capacity"):
            # Vytvoř a ulož vzpomínku (každá z jiného "mini-tématu")
            memory = self.create_distinct_memory(topic_id=i, memory_id=i, d_model=self.config.d_model)
            self.store_memory(memory, timestamp=i)
            
            # Homeostáza
            self.ltm_centers.homeostasis_step()
            
            # Každých 10 vzpomínek zkontroluj retention
            if (i + 1) % 10 == 0:
                # Zkus vybavit náhodný vzorek
                sample_size = min(10, len(self.stored_memories))
                sample_indices = np.random.choice(len(self.stored_memories), sample_size, replace=False)
                
                similarities = []
                for idx in sample_indices:
                    mem = self.stored_memories[idx]
                    r_V, _, _ = self.retrieve_memory(mem.key)
                    
                    if r_V.norm() > 1e-6:
                        sim = F.cosine_similarity(
                            r_V.unsqueeze(0),
                            mem.value.to(self.device).unsqueeze(0)
                        ).item()
                        similarities.append(sim)
                
                avg_sim = np.mean(similarities) if similarities else 0.0
                
                capacity_curve.append({
                    "n_memories": i + 1,
                    "avg_similarity": avg_sim,
                    "ltm_centers": self.ltm_centers.get_n_active(),
                    "stm_centers": self.stm_centers.get_n_active()
                })
                
                # Detekce degradace
                if degradation_point is None and avg_sim < similarity_threshold:
                    degradation_point = i + 1
        
        results = {
            "max_tested": max_memories,
            "degradation_point": degradation_point or max_memories,
            "final_ltm_centers": self.ltm_centers.get_n_active(),
            "final_stm_centers": self.stm_centers.get_n_active(),
            "capacity_curve": capacity_curve
        }
        
        print(f"\n📊 VÝSLEDKY CAPACITY TESTU:")
        print(f"   Bod degradace: {results['degradation_point']} vzpomínek")
        print(f"   Finální LTM centra: {results['final_ltm_centers']}")
        print(f"   Finální STM centra: {results['final_stm_centers']}")
        
        return results
    
    def test_consolidation_survival(
        self,
        n_memories: int = 30,
        n_consolidations: int = 5
    ) -> Dict:
        """
        TEST 4: CONSOLIDATION SURVIVAL
        
        Přežívají vzpomínky konsolidaci STM→LTM?
        """
        print("\n" + "="*60)
        print("TEST 4: CONSOLIDATION SURVIVAL")
        print("="*60)
        
        # Reset paměti
        self._init_memory()
        self.stored_memories = []
        
        # Ulož vzpomínky
        print(f"\n📝 Ukládám {n_memories} vzpomínek...")
        
        for i in range(n_memories):
            memory = self.create_distinct_memory(i % 5, i, self.config.d_model)
            self.store_memory(memory, timestamp=i)
        
        # Měření před konsolidací
        pre_results = self._measure_retrieval_quality()
        print(f"   PŘED konsolidací: avg_sim={pre_results['avg_similarity']:.3f}")
        
        # Proveď konsolidace
        print(f"\n💤 Provádím {n_consolidations} konsolidací...")
        
        for _ in range(n_consolidations):
            # Vynutí konsolidaci (nastavit fatigue nad threshold)
            self.consolidator.consolidator.fatigue = torch.tensor(
                self.config.fatigue_threshold + 1.0
            )
            
            self.consolidator.consolidator.consolidate(
                self.stm_centers,
                self.ltm_centers,
                self.stm_terrain,
                self.ltm_terrain
            )
        
        # Měření po konsolidaci
        post_results = self._measure_retrieval_quality()
        print(f"   PO konsolidaci: avg_sim={post_results['avg_similarity']:.3f}")
        
        results = {
            "n_memories": n_memories,
            "n_consolidations": n_consolidations,
            "pre_consolidation": pre_results,
            "post_consolidation": post_results,
            "survival_rate": post_results['avg_similarity'] / max(pre_results['avg_similarity'], 0.01),
            "ltm_centers_gained": post_results['ltm_centers'] - pre_results['ltm_centers']
        }
        
        print(f"\n📊 VÝSLEDKY CONSOLIDATION TESTU:")
        print(f"   Survival rate: {results['survival_rate']:.1%}")
        print(f"   LTM centra: {pre_results['ltm_centers']} → {post_results['ltm_centers']}")
        
        return results
    
    def _measure_retrieval_quality(self) -> Dict:
        """Pomocná metoda pro měření kvality vybavení."""
        similarities = []
        
        for memory in self.stored_memories:
            r_V, _, _ = self.retrieve_memory(memory.key)
            
            if r_V.norm() > 1e-6:
                sim = F.cosine_similarity(
                    r_V.unsqueeze(0),
                    memory.value.to(self.device).unsqueeze(0)
                ).item()
                similarities.append(sim)
        
        return {
            "avg_similarity": np.mean(similarities) if similarities else 0.0,
            "std_similarity": np.std(similarities) if similarities else 0.0,
            "ltm_centers": self.ltm_centers.get_n_active(),
            "stm_centers": self.stm_centers.get_n_active()
        }
    
    def run_full_suite(self) -> Dict:
        """
        Spustí všechny testy a vytvoří souhrnný report.
        """
        print("\n" + "="*70)
        print("🧠 COGNITIVE MEMORY - QUALITY TEST SUITE")
        print("="*70)
        
        results = {}
        
        # Test 1: Retention
        results["retention"] = self.test_retention(
            n_memories=50,
            n_topics=10,
            delay_steps=100
        )
        
        # Test 2: Interference
        results["interference"] = self.test_interference(
            n_topics=10,
            memories_per_topic=10
        )
        
        # Test 3: Capacity
        results["capacity"] = self.test_capacity(
            max_memories=100,
            similarity_threshold=0.3
        )
        
        # Test 4: Consolidation
        results["consolidation"] = self.test_consolidation_survival(
            n_memories=30,
            n_consolidations=3
        )
        
        # Souhrnné hodnocení
        print("\n" + "="*70)
        print("📋 SOUHRNNÉ HODNOCENÍ")
        print("="*70)
        
        scores = {
            "retention": results["retention"]["retention_rate"],
            "interference": 1.0 - results["interference"]["avg_interference"],
            "capacity": min(1.0, results["capacity"]["degradation_point"] / 100),
            "consolidation": results["consolidation"]["survival_rate"]
        }
        
        overall = np.mean(list(scores.values()))
        
        print(f"\n   Retention Score:     {scores['retention']:.1%}")
        print(f"   Anti-Interference:   {scores['interference']:.1%}")
        print(f"   Capacity Score:      {scores['capacity']:.1%}")
        print(f"   Consolidation Score: {scores['consolidation']:.1%}")
        print(f"\n   ═══════════════════════════════")
        print(f"   OVERALL SCORE:       {overall:.1%}")
        
        if overall > 0.7:
            print(f"\n   ✅ Paměť funguje DOBŘE")
        elif overall > 0.4:
            print(f"\n   ⚠️  Paměť potřebuje LADĚNÍ")
        else:
            print(f"\n   ❌ Paměť má VÁŽNÉ PROBLÉMY")
        
        results["summary"] = {
            "scores": scores,
            "overall": overall
        }
        
        return results
    
    def save_results(self, results: Dict, path: str = "memory_quality_results.json"):
        """Uloží výsledky do JSON."""
        # Konverze non-serializovatelných typů
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(x) for x in obj]
            return obj
        
        with open(path, 'w') as f:
            json.dump(convert(results), f, indent=2)
        
        print(f"\n✓ Výsledky uloženy: {path}")


def main():
    """Entry point."""
    # Konfigurace optimalizovaná pro testování
    config = MemoryConfig(
        d_model=256,
        n_ltm_centers=512,
        n_stm_centers=128,
        terrain_resolution=24,
        
        # Vyšší plasticita pro lepší zápis
        write_strength_base=0.4,
        write_bias=-0.2,
        
        # Menší sigma = více distinktních center
        ltm_sigma_write=0.2,
        ltm_sigma_read=0.3,
        stm_sigma_write=0.15,
        stm_sigma_read=0.25,
        
        # Pomalejší decay pro lepší retention
        ltm_leak=1e-5,
        stm_leak=1e-4,
        
        # Nižší threshold pro consolidation testing
        fatigue_threshold=2.0,
        consolidation_kappa=0.8,
    )
    
    tester = MemoryQualityTester(config)
    results = tester.run_full_suite()
    tester.save_results(results)


if __name__ == "__main__":
    main()

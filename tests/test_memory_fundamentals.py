# test_memory_fundamentals.py
"""
ZÁKLADNÍ TESTY PAMĚŤOVÝCH OPERACÍ

Testuje fundamentální operace:
1. Direct write/read to/from centers
2. Retention over time (decay)
3. Capacity limits
4. Similarity-based retrieval

Tento test obchází MemoryWriter a testuje přímo MemoryCenters.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
from cognitive_memory import MemoryConfig, MemoryCenters


def test_direct_write_read():
    """Test 1: Základní zápis a čtení."""
    print("\n" + "="*60)
    print("TEST 1: Direct Write/Read")
    print("="*60)
    
    centers = MemoryCenters(
        n_centers=100,
        d_key=64,
        d_value=128,
        sigma_read=0.3,
        sigma_write=0.2,
        leak=1e-5
    )
    
    print(f"\n📝 Zapisuji 10 distinktních vzpomínek...")
    
    memories = []
    for i in range(10):
        # Vytvořím unikátní klíč a hodnotu
        torch.manual_seed(i * 100)
        key = F.normalize(torch.randn(64), dim=-1)
        value = F.normalize(torch.randn(128), dim=-1)
        emotion = torch.ones(4)
        intensity = 1.0
        
        memories.append({
            "key": key.clone(),
            "value": value.clone()
        })
        
        # Přímý zápis do center
        n_new = centers.write(
            keys=key.unsqueeze(0),
            values=value.unsqueeze(0),
            emotions=emotion.unsqueeze(0),
            intensities=torch.tensor([intensity]),
            new_center_threshold=0.2  # Nižší threshold pro více center
        )
        
        print(f"   Memory {i}: new_centers={n_new}, active={centers.get_n_active()}")
    
    print(f"\n🔍 Čtu zpět všechny vzpomínky...")
    
    similarities = []
    for i, mem in enumerate(memories):
        # Přímé čtení z center
        r_V, r_E, weights, indices = centers.read(
            mem["key"].unsqueeze(0).unsqueeze(0),  # [1, 1, 64]
            top_k=8
        )
        
        r_V = r_V.squeeze()  # [128]
        
        if r_V.norm() > 1e-6:
            sim = F.cosine_similarity(
                r_V.unsqueeze(0),
                mem["value"].unsqueeze(0)
            ).item()
        else:
            sim = 0.0
        
        similarities.append(sim)
        conf = weights.max().item() if weights.shape[-1] > 0 else 0.0
        print(f"   Memory {i}: similarity={sim:.3f}, confidence={conf:.3f}")
    
    avg_sim = np.mean(similarities)
    success_rate = sum(1 for s in similarities if s > 0.5) / len(similarities)
    
    print(f"\n📊 VÝSLEDKY:")
    print(f"   Průměrná podobnost: {avg_sim:.3f}")
    print(f"   Success rate (>0.5): {success_rate:.0%}")
    print(f"   Aktivní centra: {centers.get_n_active()}")
    
    return avg_sim, success_rate


def test_retention_over_time():
    """Test 2: Retention po simulovaném čase."""
    print("\n" + "="*60)
    print("TEST 2: Retention Over Time")
    print("="*60)
    
    centers = MemoryCenters(
        n_centers=100,
        d_key=64,
        d_value=128,
        sigma_read=0.3,
        sigma_write=0.2,
        leak=1e-4  # Pomalejší decay pro test
    )
    
    print(f"\n📝 Zapisuji 5 vzpomínek...")
    
    memories = []
    for i in range(5):
        torch.manual_seed(i * 100)
        key = F.normalize(torch.randn(64), dim=-1)
        value = F.normalize(torch.randn(128), dim=-1)
        
        memories.append({"key": key.clone(), "value": value.clone()})
        
        centers.write(
            keys=key.unsqueeze(0),
            values=value.unsqueeze(0),
            emotions=torch.ones(1, 4),
            intensities=torch.tensor([1.0]),
            new_center_threshold=0.1
        )
    
    # Referenční měření
    def measure_quality():
        sims = []
        for mem in memories:
            r_V, _, _, _ = centers.read(mem["key"].unsqueeze(0).unsqueeze(0), top_k=4)
            if r_V.norm() > 1e-6:
                sims.append(F.cosine_similarity(
                    r_V.squeeze().unsqueeze(0),
                    mem["value"].unsqueeze(0)
                ).item())
        return np.mean(sims) if sims else 0.0
    
    quality_before = measure_quality()
    print(f"   Kvalita PŘED decay: {quality_before:.3f}")
    
    # Simuluj 1000 kroků decay
    print(f"\n⏳ Simuluji 1000 kroků decay...")
    for _ in range(1000):
        centers.homeostasis_step()
    
    quality_after = measure_quality()
    print(f"   Kvalita PO decay: {quality_after:.3f}")
    
    retention = quality_after / max(quality_before, 0.01)
    print(f"\n📊 RETENTION: {retention:.1%}")
    
    return retention


def test_capacity():
    """Test 3: Kapacita paměti."""
    print("\n" + "="*60)
    print("TEST 3: Capacity Test")
    print("="*60)
    
    centers = MemoryCenters(
        n_centers=50,  # Malá kapacita pro rychlý test
        d_key=64,
        d_value=128,
        sigma_read=0.2,
        sigma_write=0.15,
        leak=0
    )
    
    print(f"\n📝 Postupně zapisuji vzpomínky až do saturace...")
    
    memories = []
    for i in range(100):
        torch.manual_seed(i * 100)
        key = F.normalize(torch.randn(64), dim=-1)
        value = F.normalize(torch.randn(128), dim=-1)
        
        memories.append({"key": key.clone(), "value": value.clone()})
        
        n_new = centers.write(
            keys=key.unsqueeze(0),
            values=value.unsqueeze(0),
            emotions=torch.ones(1, 4),
            intensities=torch.tensor([1.0]),
            new_center_threshold=0.15
        )
        
        n_active = centers.get_n_active()
        
        if i % 10 == 0:
            # Měř kvalitu
            sample = memories[-min(5, len(memories)):]
            sims = []
            for mem in sample:
                r_V, _, _, _ = centers.read(mem["key"].unsqueeze(0).unsqueeze(0), top_k=4)
                if r_V.norm() > 1e-6:
                    sims.append(F.cosine_similarity(
                        r_V.squeeze().unsqueeze(0),
                        mem["value"].unsqueeze(0)
                    ).item())
            
            avg_sim = np.mean(sims) if sims else 0.0
            print(f"   i={i}: active={n_active}, new={n_new}, avg_sim={avg_sim:.3f}")
            
            if n_active >= 50:
                print(f"   ⚠️  Kapacita saturována!")
                break
    
    print(f"\n📊 Finální kapacita: {centers.get_n_active()} center")
    return centers.get_n_active()


def test_similarity_retrieval():
    """Test 4: Vybavení podobných vzpomínek."""
    print("\n" + "="*60)
    print("TEST 4: Similarity-Based Retrieval")
    print("="*60)
    
    centers = MemoryCenters(
        n_centers=100,
        d_key=64,
        d_value=128,
        sigma_read=0.25,
        sigma_write=0.2,
        leak=0
    )
    
    # Vytvoř 5 "témat" - každé má charakteristický base vektor
    topics = []
    for t in range(5):
        torch.manual_seed(t * 1000)
        base = F.normalize(torch.randn(64), dim=-1)
        topics.append(base)
        
        # Ulož 3 variace každého tématu
        for v in range(3):
            torch.manual_seed(t * 1000 + v + 1)
            variation = F.normalize(base + 0.3 * torch.randn(64), dim=-1)
            
            # Hodnota je kódovaná jako identifikátor tématu
            value = torch.zeros(128)
            value[t] = 1.0  # One-hot encoding tématu
            
            centers.write(
                keys=variation.unsqueeze(0),
                values=value.unsqueeze(0),
                emotions=torch.ones(1, 4),
                intensities=torch.tensor([1.0]),
                new_center_threshold=0.3
            )
    
    print(f"\n   Uloženo: {centers.get_n_active()} center pro 5 témat")
    
    # Testuj vybavení pro každé téma
    print(f"\n🔍 Testuji vybavení podle tématu...")
    
    correct = 0
    for t in range(5):
        # Dotaz: variace tématu
        torch.manual_seed(t * 1000 + 999)  # Nová variace
        query = F.normalize(topics[t] + 0.2 * torch.randn(64), dim=-1)
        
        r_V, _, weights, _ = centers.read(query.unsqueeze(0).unsqueeze(0), top_k=4)
        r_V = r_V.squeeze()
        
        # Zjisti které téma bylo vybaveno
        retrieved_topic = r_V.argmax().item()
        is_correct = retrieved_topic == t
        
        conf = weights.max().item() if weights.shape[-1] > 0 else 0.0
        
        print(f"   Topic {t}: retrieved={retrieved_topic}, correct={is_correct}, conf={conf:.3f}")
        
        if is_correct:
            correct += 1
    
    accuracy = correct / 5
    print(f"\n📊 ACCURACY: {accuracy:.0%} ({correct}/5)")
    
    return accuracy


def main():
    print("\n" + "="*70)
    print("🧠 COGNITIVE MEMORY - FUNDAMENTAL TESTS")
    print("="*70)
    
    results = {}
    
    # Test 1: Direct read/write
    avg_sim, success_rate = test_direct_write_read()
    results["direct_rw"] = {"avg_sim": avg_sim, "success_rate": success_rate}
    
    # Test 2: Retention
    retention = test_retention_over_time()
    results["retention"] = retention
    
    # Test 3: Capacity
    capacity = test_capacity()
    results["capacity"] = capacity
    
    # Test 4: Similarity retrieval
    accuracy = test_similarity_retrieval()
    results["retrieval_accuracy"] = accuracy
    
    # Souhrn
    print("\n" + "="*70)
    print("📋 SOUHRN VÝSLEDKŮ")
    print("="*70)
    
    print(f"\n   Direct R/W:    {results['direct_rw']['avg_sim']:.3f} avg similarity")
    print(f"   Retention:     {results['retention']:.1%} after 1000 steps")
    print(f"   Capacity:      {results['capacity']} centers used")
    print(f"   Retrieval:     {results['retrieval_accuracy']:.0%} accuracy")
    
    # Celkové hodnocení
    overall = (
        results['direct_rw']['success_rate'] +
        results['retention'] +
        min(1.0, results['capacity'] / 50) +
        results['retrieval_accuracy']
    ) / 4
    
    print(f"\n   ═══════════════════════════════")
    print(f"   OVERALL SCORE: {overall:.1%}")
    
    if overall > 0.7:
        print(f"\n   ✅ Základní operace fungují DOBŘE")
    elif overall > 0.4:
        print(f"\n   ⚠️  Některé operace potřebují LADĚNÍ")
    else:
        print(f"\n   ❌ Základní operace mají PROBLÉMY")
    
    return results


if __name__ == "__main__":
    main()

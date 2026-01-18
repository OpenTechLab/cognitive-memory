import subprocess
import sys
import time
from pathlib import Path
import os

# Seznam testů v pořadí spouštění
TEST_SUITE = [
    {
        "name": "Fundamentals Check",
        "script": "test_memory_fundamentals.py",
        "desc": "Ověření základních operací (Read/Write/Capacity)",
        "critical": True
    },
    {
        "name": "Quality Assurance",
        "script": "memory_quality_test.py",
        "desc": "Detailní testy retence, interference a konsolidace",
        "critical": True
    },
    {
        "name": "Audit Memory Content",
        "script": "audit_memory_content.py",
        "desc": "Ověření diskriminace a integrity obsahu (Retrieval Audit)",
        "critical": False
    },
    {
        "name": "Visualizations (3D Landscape)",
        "script": "visualize_centers_structure.py",
        "desc": "Generování 3D sémantické mapy a Scatter plotu",
        "critical": False
    },
    {
        "name": "Visualizations (Metrics)",
        "script": "visualize_stress_test.py",
        "desc": "Generování grafů metrik ze stress testu",
        "critical": False
    }
]

# Stress test je speciální
STRESS_TEST = {
    "name": "Full Stress Test",
    "script": "stress_test_memory.py",
    "desc": "Dlouhodobá simulace (9000 kroků)",
}

def run_command(command, description):
    print(f"\n{'='*70}")
    print(f"🚀 RUNNING: {description}")
    print(f"   Script: {command}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    # Spuštění procesu
    # Používáme sys.executable pro jistotu stejného prostředí
    process = subprocess.Popen(
        [sys.executable, command],
        stderr=subprocess.STDOUT, # Sloučit stderr do stdout
        stdout=sys.stdout,        # Tisknout rovnou na konzoli
        bufsize=1,
        universal_newlines=True,
        cwd=str(Path(__file__).parent) # Spouštět v adresáři memory-tests
    )
    
    process.wait()
    end_time = time.time()
    duration = end_time - start_time
    
    if process.returncode == 0:
        print(f"\n✅ SUCCESS ({duration:.1f}s)")
        return True
    else:
        print(f"\n❌ FAILED (Exit Code: {process.returncode})")
        return False

def main():
    print(f"""
    ############################################################
    🧠  BIOCORTEX MEMORY SUITE - ORCHESTRATION SCRIPT
    ############################################################
    """)
    
    overall_start = time.time()
    results = {}
    
    # 1. Spustit UNIT TESTY
    print("\n--- PHASE 1: UNIT & QUALITY TESTS ---")
    for test in TEST_SUITE[:3]: # První 3 (Fundamentals, Quality, Audit)
        success = run_command(test['script'], test['name'])
        results[test['name']] = success
        if not success and test['critical']:
            print("⛔ Critical test failed. Aborting suite.")
            sys.exit(1)

    # 2. Spustit STRESS TEST?
    print("\n--- PHASE 2: LONG-RUNNING STRESS TEST ---")
    # Zkontrolujeme argumenty
    force_stress = "--force" in sys.argv
    
    # Zkontrolujeme, zda už nemáme hotové výsledky
    results_dir = Path("stress_test_results")
    has_results = results_dir.exists() and (results_dir / "metrics_RealisticMixed.csv").exists()
    
    if has_results and not force_stress:
        print(f"⚠️  Found existing stress test results in {results_dir}.")
        print("   Skipping 9000-step simulation to save time.")
        print("   Use 'python run_full_memory_suite.py --force' to rerun everything.")
        run_stress = False
    else:
        run_stress = True
        if force_stress:
            print("⚠️  Force flag detected. Rerunning stress test...")
        
    if run_stress:
        success = run_command(STRESS_TEST['script'], STRESS_TEST['name'])
        results[STRESS_TEST['name']] = success
        if not success:
            print("⛔ Stress test failed. Visualizations might be incomplete.")
    else:
        results[STRESS_TEST['name']] = "SKIPPED"

    # 3. Spustit VIZUALIZACE
    print("\n--- PHASE 3: VISUALIZATIONS ---")
    for test in TEST_SUITE[3:]: # Zbytek (Vizualizace)
        success = run_command(test['script'], test['name'])
        results[test['name']] = success

    # FINAL REPORT
    overall_time = time.time() - overall_start
    print("\n" * 2)
    print("=" * 60)
    print(f"📊 SUITE COMPLETION REPORT ({overall_time/60:.1f} min)")
    print("=" * 60)
    
    all_passed = True
    for name, result in results.items():
        if result is True:
            status = "✅ PASS"
        elif result == "SKIPPED":
            status = "⏭️  SKIP"
        else:
            status = "❌ FAIL"
            all_passed = False
        print(f"{status:10} | {name}")
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 ALL SYSTEMS NOMINAL. MEMORY IS READY FOR PRODUCTION. 🧠")
    else:
        print("\n⚠️  SOME TESTS FAILED. CHECK LOGS.")
        sys.exit(1)

if __name__ == "__main__":
    main()

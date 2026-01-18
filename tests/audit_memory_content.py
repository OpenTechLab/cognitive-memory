import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

# Importy z projektu
sys.path.insert(0, str(Path(__file__).parent.parent))
from cognitive_memory.memory_centers import MemoryCenters

def audit_memory():
    print('='*60)
    print('🕵️ AUDIT OBSAHU PAMĚTI (RETRIEVAL TEST)')
    print('='*60)
    
    # 1. Načíst snapshot
    snapshot_dirs = list(Path("stress_test_results").glob("RealisticMixed_step_*/memory.pt"))
    if not snapshot_dirs:
        print("❌ Žádný snapshot.")
        return
    latest = max(snapshot_dirs, key=lambda p: p.stat().st_mtime)
    print(f"📂 Načítám snapshot: {latest.name}")
    
    state = torch.load(str(latest), map_location='cpu', weights_only=False)
    
    if 'ltm_centers' not in state:
        print("❌ Chybí LTM centers.")
        return
        
    # Rekonstrukce center
    centers_state = state['ltm_centers']
    # Potřebujeme vytvořit instanci pro použití metody read()
    # Získáme parametry ze state dict (odhadem)
    n_centers = centers_state['K'].shape[0]
    d_key = centers_state['K'].shape[1]
    d_value = centers_state['V'].shape[1]
    
    mc = MemoryCenters(
        n_centers=n_centers,
        d_key=d_key,
        d_value=d_value,
        sigma_read=0.3, # Použijeme standardní hodnotu
        device='cpu'
    )
    
    # Load state manually
    mc.K.copy_(centers_state['K'])
    mc.V.copy_(centers_state['V'])
    
    # Emoce: v MemoryCenters je to 'e', v Terrain3D 'E'
    # Ve snapshotu center by to mělo být 'e'
    if 'e' in centers_state:
        mc.e.copy_(centers_state['e'])
    elif 'E' in centers_state: # Backward compatibility
        mc.e.copy_(centers_state['E'])
        
    if 'h' in centers_state:
        mc.h.copy_(centers_state['h'])
    elif 'H' in centers_state:
        mc.h.copy_(centers_state['H'])
        
    mc.active.copy_(centers_state['active'])
    mc.usage.copy_(centers_state['usage'])
    
    n_active = mc.get_n_active()
    print(f"✅ Paměť načtena: {n_active} aktivních center")
    
    if n_active < 2:
        print("⚠️ Příliš málo center pro test diskriminace.")
        return

    # 2. Test diskriminace (Rozlišitelnost)
    # Vezmeme existující klíče z aktivních center a zkusíme je vybavit
    active_indices = torch.where(mc.active)[0]
    
    # Vybereme 5 náhodných center jako "Queries"
    sample_indices = active_indices[torch.randperm(len(active_indices))[:5]]
    
    print("\n🧪 TEST DISKRIMINACE (Query = Key existujícího centra):")
    print(f"{'Query ID':<10} | {'Found Match':<10} | {'Confidence':<10} | {'Similarity':<10} | {'Status'}")
    print("-" * 65)
    
    distinct_values = []
    
    for idx in sample_indices:
        # Dotaz je přímo klíč centra (ideální případ)
        query_key = mc.K[idx].view(1, 1, d_key)
        target_value = mc.V[idx]
        
        # Read
        # read vrací: values, emotions, weights, indices
        r_V, r_E, weights, _ = mc.read(query_key, top_k=4)
        
        retrieved_val = r_V.squeeze()
        confidence = weights.sum().item()
        
        # Spočítat podobnost s očekávanou hodnotou (self)
        sim = F.cosine_similarity(retrieved_val.unsqueeze(0), target_value.unsqueeze(0)).item()
        
        distinct_values.append(retrieved_val)
        
        status = "✅ OK" if sim > 0.9 else "⚠️ Weak" if sim > 0.5 else "❌ Fail"
        
        print(f"{idx.item():<10} | {weights[0,0,0].item():.4f}     | {confidence:.4f}     | {sim:.4f}     | {status}")

    # 3. Křížová podobnost (Cross-Talk)
    # Zkontrolujeme, zda jsou "vybavené hodnoty" odlišné
    print("\n🔍 KŘÍŽOVÁ KONTROLA (Jsou vybavené vzpomínky různé?):")
    import itertools
    
    tensor_stack = torch.stack(distinct_values)
    # Matice podobnosti [5, 5]
    cross_sim = torch.mm(tensor_stack, tensor_stack.t())
    
    # Průměrná podobnost mimo diagonálu
    mask = ~torch.eye(5, dtype=bool)
    avg_cross_sim = cross_sim[mask].mean().item()
    
    print(f"Průměrná podobnost mezi RŮZNÝMI vzpomínkami: {avg_cross_sim:.4f}")
    
    if avg_cross_sim > 0.8:
        print("❌ PROBLÉM: Paměť vrací velmi podobné hodnoty pro různé dotazy (Mode Collapse).")
    elif avg_cross_sim < 0.5:
        print("✅ ÚSPĚCH: Paměť pro různé klíče vrací RŮZNÉ hodnoty.")
    else:
        print("⚠️ VAROVÁNÍ: Vzpomínky jsou si částečně podobné (možná sdílené téma).")

    # Uložit heatmapu
    plt.figure()
    plt.imshow(cross_sim.detach().numpy(), cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(label='Cosine Similarity')
    plt.title('Cross-Similarity of Retrieved Memories')
    plt.savefig('memory_discrimination_audit.png')
    print("Graph saved: memory_discrimination_audit.png")

if __name__ == "__main__":
    audit_memory()

# realistic_scenarios.py
"""
ULTRA-REALISTICKÉ scénáře pro Cognitive Memory stress test.

OPRAVENO 2026-01-14:
- Fix IndexError při prázdném topics_today
- Správná implementace Poisson-like intervalů
- Bimodální surprise distribuce
- Podpora pro generování SEKVENCÍ tokenů (realistické prompty)

Kalibrace podle reálného použití:
- 500 témat (vysoká variabilita)
- 1 interakce = ~50-200 tokenů (realistický prompt + odpověď)
- 50 interakcí/den (900 tokenů/den = 50 interakcí × ~18 tokenů průměr prompt)
- Aha moment průměrně každých 18 INTERAKCÍ (ne tokenů!)
- Random emoční valence (pozitivní/negativní)
- Půl roku = ~9000 interakcí = ~180 000+ tokenů
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from typing import Tuple, Dict, List
from stress_test_memory import StressTestScenario


class RealisticMixedScenario(StressTestScenario):
    """
    Simuluje ULTRA-REALISTICKÝ provoz LLM.
    
    OPRAVENO:
    - Generuje SEKVENCE tokenů místo jednotlivých embeddingů
    - Správný Poisson model pro aha momenty
    - Fix pro prázdný topics_today
    
    Kalibrace:
    - 500 různých témat (vysoká variabilita)
    - 50 interakcí/den (každá = 50-200 tokenů)
    - Aha moment průměrně každých 18 interakcí
    - Random emoční valence (pozitivní/negativní)
    """
    
    def __init__(self, tokens_per_interaction: int = 100):
        super().__init__(
            "RealisticMixed",
            f"Ultra-realistic: 500 topics, {tokens_per_interaction} tokens/interaction, aha moments every ~18 interactions"
        )
        self.topic_centers = None
        self.topic_history = []
        self.current_day = 0
        self.topics_today = []
        self.current_topic = None  # FIX: Track current topic separately
        
        # Poisson model pro aha momenty - generuje se JEDNOU při startu interakce
        self.next_aha_at = 0  # FIX: Správný Poisson model
        self.interaction_count = 0
        
        # Konfigurace sekvencí
        self.tokens_per_interaction = tokens_per_interaction
        
    def prepare(self, n: int, d_model: int, cluster_centers: int = 500):
        """500 různých témat s vysokou variabilitou."""
        super().prepare(n, d_model, max(500, cluster_centers))
        
        with torch.no_grad():
            # 500 topic centers
            self.topic_centers = torch.randn(self.cluster_centers, d_model)
            self.topic_centers = self.topic_centers / self.topic_centers.norm(dim=-1, keepdim=True)
            
            # Topic popularity (Zipf distribution - některá častější)
            ranks = np.arange(1, self.cluster_centers + 1)
            self.topic_popularity = 1.0 / ranks
            self.topic_popularity = self.topic_popularity / self.topic_popularity.sum()
            
            # FIX: Inicializuj první téma
            self.current_topic = np.random.choice(self.cluster_centers, p=self.topic_popularity)
            self.topics_today = [self.current_topic]
            
            # FIX: Generuj první aha moment podle Poisson
            self.next_aha_at = int(np.random.exponential(18))
            self.interaction_count = 0
    
    def generate_sequence(self, interaction_idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Generuje CELOU SEKVENCI tokenů pro jednu interakci.
        
        Args:
            interaction_idx: Index interakce (ne tokenu!)
            
        Returns:
            (sequence [T, D], metadata)
        """
        with torch.no_grad():
            # Počet tokenů pro tuto interakci (variabilní)
            n_tokens = int(np.random.uniform(
                self.tokens_per_interaction * 0.5,
                self.tokens_per_interaction * 1.5
            ))
            
            # === DEN TRACKING (50 interakcí/den) ===
            current_day = interaction_idx // 50
            
            # Reset topics_today při novém dni
            if current_day != self.current_day:
                self.current_day = current_day
                # FIX: Vždy zachovat alespoň aktuální téma
                self.topics_today = [self.current_topic] if self.current_topic is not None else []
            
            # === TOPIC SELECTION ===
            # Průměrně 5 témat/den = 50/5 = 10 interakcí/téma
            
            if len(self.topics_today) == 0 or np.random.random() < 0.1:  # 10% šance na nové téma
                # Nové téma (weighted by popularity)
                cluster_id = np.random.choice(self.cluster_centers, p=self.topic_popularity)
                self.current_topic = cluster_id
                
                if cluster_id not in self.topics_today:
                    self.topics_today.append(cluster_id)
            else:
                # FIX: Bezpečný přístup k current_topic
                if self.current_topic is not None:
                    cluster_id = self.current_topic
                elif len(self.topics_today) > 0:
                    cluster_id = self.topics_today[-1]
                    self.current_topic = cluster_id
                else:
                    # Fallback: vyber náhodné téma
                    cluster_id = np.random.choice(self.cluster_centers, p=self.topic_popularity)
                    self.current_topic = cluster_id
                    self.topics_today.append(cluster_id)
                
                # Občas switch mezi dnešními tématy (udržení kontextu dne)
                if len(self.topics_today) > 1 and np.random.random() < 0.1:
                    cluster_id = np.random.choice(self.topics_today)
                    self.current_topic = cluster_id
            
            self.topic_history.append(cluster_id)
            if len(self.topic_history) > 1000:
                self.topic_history.pop(0)
            
            # === GENEROVÁNÍ SEKVENCE ===
            center = self.topic_centers[cluster_id]
            
            # Sekvence tokenů s korelací v čase (random walk kolem centra)
            sequence = torch.zeros(n_tokens, self.d_model)
            current_pos = center.clone()
            
            for t in range(n_tokens):
                # Malý drift od předchozí pozice (koherence v rámci promptu)
                drift = torch.randn(self.d_model) * 0.15
                current_pos = current_pos + drift
                current_pos = current_pos / current_pos.norm()
                
                # Větší variabilita kolem tématu
                noise = torch.randn(self.d_model) * 0.3
                token_emb = current_pos + noise
                token_emb = token_emb / token_emb.norm()
                
                sequence[t] = token_emb
            
            # === SURPRISE (bimodální - pro CELOU interakci) ===
            # 80% low (routine), 20% high (novelty)
            if np.random.random() < 0.80:
                surprise = float(np.random.beta(2, 5))  # Low (mean ~0.2)
            else:
                surprise = float(np.random.beta(5, 2))  # High (mean ~0.7)
            
            # === AHA MOMENT (Poisson model - fixní interval) ===
            self.interaction_count += 1
            is_aha_moment = self.interaction_count >= self.next_aha_at
            
            if is_aha_moment:
                # FIX: Generuj NOVÝ interval až po dosažení
                self.next_aha_at = self.interaction_count + max(1, int(np.random.exponential(18)))
                surprise = max(surprise, float(np.random.beta(6, 2)))  # Force high surprise
            
            # === EMOTIONS (random valence) ===
            base = torch.ones(4)
            
            # 45% pozitivní, 45% negativní, 10% neutrální
            valence = np.random.choice(['positive', 'negative', 'neutral'], p=[0.45, 0.45, 0.10])
            
            if is_aha_moment:
                # Aha moment: silné emoce
                intensity = np.random.uniform(1.2, 2.5)
                
                if valence == 'positive':
                    base[0] += intensity * np.random.uniform(0.8, 1.2)  # Dopamin
                    base[1] += intensity * np.random.uniform(0.5, 0.9)  # Serotonin
                    base[3] += intensity * np.random.uniform(0.6, 1.0)  # Oxytocin
                elif valence == 'negative':
                    base[2] += intensity * np.random.uniform(0.8, 1.5)  # Kortizol
                    base[0] -= intensity * 0.2
                else:
                    base[0] += intensity * 0.5
                    base[2] += intensity * 0.3
            else:
                # Běžná interakce: mírnější emoce
                intensity = np.random.uniform(0.1, 0.6)
                
                if valence == 'positive':
                    base[0] += intensity * np.random.uniform(0.3, 0.7)
                    base[1] += intensity * np.random.uniform(0.2, 0.5)
                elif valence == 'negative':
                    base[2] += intensity * np.random.uniform(0.3, 0.8)
                    base[0] -= intensity * 0.1
            
            emotions = torch.clamp(base, 0.3, 3.0)
            
            return sequence, {
                "emotions": emotions,
                "surprise": surprise,
                "cluster_id": int(cluster_id),
                "interaction_idx": interaction_idx,
                "n_tokens": n_tokens,
                "is_aha_moment": is_aha_moment,
                "valence": valence,
                "day": current_day,
                "topics_today": len(self.topics_today)
            }
    
    @torch.no_grad()
    def generate_embedding(self, step: int) -> Tuple[torch.Tensor, Dict]:
        """
        Zpětná kompatibilita - generuje jeden embedding.
        Pro realistické testy použijte generate_sequence().
        """
        # Mapuj step na interakci (každých tokens_per_interaction kroků = 1 interakce)
        interaction_idx = step // self.tokens_per_interaction
        token_in_interaction = step % self.tokens_per_interaction
        
        # === DEN TRACKING ===
        current_day = interaction_idx // 50
        
        if current_day != self.current_day:
            self.current_day = current_day
            self.topics_today = [self.current_topic] if self.current_topic is not None else []
        
        # === TOPIC SELECTION (jen na začátku interakce) ===
        if token_in_interaction == 0:
            if len(self.topics_today) == 0 or np.random.random() < 0.1:
                cluster_id = np.random.choice(self.cluster_centers, p=self.topic_popularity)
                self.current_topic = cluster_id
                if cluster_id not in self.topics_today:
                    self.topics_today.append(cluster_id)
        
        # FIX: Vždy použij current_topic (bezpečně)
        if self.current_topic is not None:
            cluster_id = self.current_topic
        elif len(self.topics_today) > 0:
            cluster_id = self.topics_today[-1]
            self.current_topic = cluster_id
        else:
            cluster_id = np.random.choice(self.cluster_centers, p=self.topic_popularity)
            self.current_topic = cluster_id
            self.topics_today.append(cluster_id)
        
        # === EMBEDDING ===
        center = self.topic_centers[cluster_id]
        noise = torch.randn(self.d_model) * 0.4
        emb = center + noise
        emb = emb / emb.norm()
        
        # === SURPRISE (bimodální) ===
        if np.random.random() < 0.80:
            surprise = float(np.random.beta(2, 5))
        else:
            surprise = float(np.random.beta(5, 2))
        
        # === AHA MOMENT (jen na začátku interakce) ===
        is_aha_moment = False
        if token_in_interaction == 0:
            self.interaction_count += 1
            is_aha_moment = self.interaction_count >= self.next_aha_at
            
            if is_aha_moment:
                self.next_aha_at = self.interaction_count + max(1, int(np.random.exponential(18)))
                surprise = max(surprise, float(np.random.beta(6, 2)))
        
        # === EMOTIONS ===
        base = torch.ones(4)
        valence = np.random.choice(['positive', 'negative', 'neutral'], p=[0.45, 0.45, 0.10])
        
        if is_aha_moment:
            intensity = np.random.uniform(1.2, 2.5)
            if valence == 'positive':
                base[0] += intensity * np.random.uniform(0.8, 1.2)
                base[1] += intensity * np.random.uniform(0.5, 0.9)
                base[3] += intensity * np.random.uniform(0.6, 1.0)
            elif valence == 'negative':
                base[2] += intensity * np.random.uniform(0.8, 1.5)
                base[0] -= intensity * 0.2
            else:
                base[0] += intensity * 0.5
                base[2] += intensity * 0.3
        else:
            intensity = np.random.uniform(0.1, 0.6)
            if valence == 'positive':
                base[0] += intensity * np.random.uniform(0.3, 0.7)
                base[1] += intensity * np.random.uniform(0.2, 0.5)
            elif valence == 'negative':
                base[2] += intensity * np.random.uniform(0.3, 0.8)
                base[0] -= intensity * 0.1
        
        emotions = torch.clamp(base, 0.3, 3.0)
        
        return emb, {
            "emotions": emotions,
            "surprise": surprise,
            "cluster_id": int(cluster_id),
            "timestamp": step,
            "interaction_idx": interaction_idx,
            "token_in_interaction": token_in_interaction,
            "is_aha_moment": is_aha_moment,
            "valence": valence,
            "day": current_day,
            "topics_today": len(self.topics_today)
        }


# Export
__all__ = ['RealisticMixedScenario']

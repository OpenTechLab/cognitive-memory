# analyze_memory_risks.py
"""
Analýza známých rizik v Cognitive Memory systému.

Detekuje a kvantifikuje:
1. Catastrophic interference (prolínání nesouvisejících vzpomínek)
2. Capacity saturation (vyčerpání kapacity)
3. Diffusion instability (nestabilita difuze)
4. Consolidation failure (selhání STM→LTM)
5. Memory leak (nežádoucí růst paměti)
6. Temporal bias (zapomnění starých vzpomínek)
"""

import sys
from pathlib import Path

# Přidej parent directory do sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import warnings

from cognitive_memory import load_memory_state


@dataclass
class RiskAssessment:
    """Struktura pro hodnocení rizika."""
    risk_name: str
    severity: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    score: float  # 0-1
    description: str
    details: Dict
    recommendations: List[str]


class MemoryRiskAnalyzer:
    """Analyzér rizik pro Cognitive Memory."""
    
    def __init__(self, results_dir: str = "stress_test_results"):
        self.results_dir = Path(results_dir)
        self.metrics_df = None
        self.consolidation_events = None
        self.risks: List[RiskAssessment] = []
        
        # Načti data
        self._load_data()
    
    def _load_data(self):
        """Načte metriky a eventy."""
        # Hledej metrics_*.csv (podporuje různé názvy scénářů)
        metrics_files = list(self.results_dir.glob("metrics_*.csv"))
        
        if metrics_files:
            # Načti první nalezený soubor
            metrics_path = metrics_files[0]
            self.metrics_df = pd.read_csv(metrics_path)
            print(f"✓ Loaded metrics: {metrics_path.name}")
        else:
            print(f"⚠ No metrics found in {self.results_dir}")
        
        consol_path = self.results_dir / "consolidation_events.json"
        if consol_path.exists():
            with open(consol_path, 'r') as f:
                self.consolidation_events = json.load(f)
            print(f"✓ Loaded {len(self.consolidation_events)} consolidation events")
        else:
            print(f"⚠ No consolidation events found")
    
    def analyze_catastrophic_interference(self) -> RiskAssessment:
        """
        Riziko 1: Catastrophic Interference
        
        Detekce: Sleduje, zda nové vzpomínky přepisují staré.
        Metriky:
        - Pokles retention rate
        - Překrývání clusterů (pokud jsou k dispozici)
        - Fluktuace v intenzitě center
        """
        if self.metrics_df is None:
            return self._no_data_risk("Catastrophic Interference")
        
        df = self.metrics_df
        
        # Proxy: pokud počet aktivních center klesá, může docházet k přepisování
        n_active = df['n_active'].values
        
        # Pokles v posledních 20% dat
        split_point = int(len(n_active) * 0.8)
        early_mean = n_active[:split_point].mean()
        late_mean = n_active[split_point:].mean()
        
        retention_loss = max(0, 1 - (late_mean / early_mean)) if early_mean > 0 else 0
        
        # Fluktuace write strength (vysoká variance = nepředvídatelné přepisování)
        omega_variance = df['omega_mean'].var()
        
        # Kombinovaný score
        interference_score = 0.5 * retention_loss + 0.5 * min(1.0, omega_variance / 0.01)
        
        # Severity
        if interference_score < 0.2:
            severity = "LOW"
            color = "🟢"
        elif interference_score < 0.5:
            severity = "MEDIUM"
            color = "🟡"
        elif interference_score < 0.8:
            severity = "HIGH"
            color = "🟠"
        else:
            severity = "CRITICAL"
            color = "🔴"
        
        recommendations = []
        if interference_score > 0.3:
            recommendations.append("Zvětšit kapacitu LTM (n_ltm_centers)")
            recommendations.append("Snížit write_strength_base (pomalejší zápis)")
            recommendations.append("Zvýšit merge_similarity_threshold (méně mergování)")
        
        return RiskAssessment(
            risk_name=f"{color} Catastrophic Interference",
            severity=severity,
            score=interference_score,
            description="Riziko přepisování starých vzpomínek novými",
            details={
                "retention_loss": f"{retention_loss:.2%}",
                "omega_variance": f"{omega_variance:.4f}",
                "early_active_mean": int(early_mean),
                "late_active_mean": int(late_mean)
            },
            recommendations=recommendations
        )
    
    def analyze_capacity_saturation(self) -> RiskAssessment:
        """
        Riziko 2: Capacity Saturation
        
        Detekce: Systém vyčerpává dostupnou kapacitu.
        Metriky:
        - Podíl aktivních center k maximu
        - Frekvence mergování
        - Frekvence prune operací
        """
        if self.metrics_df is None:
            return self._no_data_risk("Capacity Saturation")
        
        df = self.metrics_df
        
        # Finální využití kapacity
        final_active = df['n_active'].iloc[-1]
        max_capacity = df['n_total'].iloc[-1]
        capacity_usage = final_active / max_capacity
        
        # Trend růstu (lineární fit)
        steps = df['step'].values
        active = df['n_active'].values
        
        if len(steps) > 100:
            # Poslední 20%
            tail_steps = steps[-len(steps)//5:]
            tail_active = active[-len(active)//5:]
            
            # Slope
            slope = np.polyfit(tail_steps, tail_active, 1)[0]
            
            # Extrapolace kdy se naplní
            if slope > 0:
                steps_to_full = (max_capacity - final_active) / slope
            else:
                steps_to_full = float('inf')
        else:
            steps_to_full = float('inf')
        
        # Score
        saturation_score = capacity_usage
        if steps_to_full < 5000:  # Méně než 5000 kroků do naplnění
            saturation_score = max(saturation_score, 0.7)
        
        # Severity
        if saturation_score < 0.5:
            severity = "LOW"
            color = "🟢"
        elif saturation_score < 0.75:
            severity = "MEDIUM"
            color = "🟡"
        elif saturation_score < 0.9:
            severity = "HIGH"
            color = "🟠"
        else:
            severity = "CRITICAL"
            color = "🔴"
        
        recommendations = []
        if saturation_score > 0.6:
            recommendations.append("Zvýšit max_centers_ltm (větší kapacita)")
            recommendations.append("Snížit new_center_threshold (méně nových center)")
            recommendations.append("Zvýšit prune_intensity_threshold (agresivnější prune)")
        
        return RiskAssessment(
            risk_name=f"{color} Capacity Saturation",
            severity=severity,
            score=saturation_score,
            description="Riziko vyčerpání dostupné kapacity paměti",
            details={
                "capacity_usage": f"{capacity_usage:.2%}",
                "active_centers": int(final_active),
                "max_capacity": int(max_capacity),
                "steps_to_full": int(steps_to_full) if steps_to_full != float('inf') else "∞",
                "growth_slope": f"{slope:.2f}" if 'slope' in locals() else "N/A"
            },
            recommendations=recommendations
        )
    
    def analyze_diffusion_stability(self) -> RiskAssessment:
        """
        Riziko 3: Diffusion Instability
        
        Detekce: Nestabilní difuze může vést k explozi hodnot.
        Metriky:
        - H_max > threshold (exploding values)
        - E_max > 2.0 (emoce mimo rozsah)
        - Fluktuace v H_mean
        """
        if self.metrics_df is None:
            return self._no_data_risk("Diffusion Instability")
        
        df = self.metrics_df
        
        # Maximální hodnoty
        h_max_peak = df['H_max'].max()
        h_mean_std = df['H_mean'].std()
        
        # E_max (pokud je v datech)
        if 'E_max' in df.columns:
            e_max_peak = df['E_max'].max()
        else:
            e_max_peak = 0
        
        # CFL stability condition: α ≤ 1/6
        # Pokud α_H = 0.002, je stabilní
        # Ale můžeme kontrolovat růst hodnot
        
        # Score based on values
        instability_score = 0
        
        # 1. H_max by neměl růst do nekonečna
        if h_max_peak > 5.0:  # Arbitrary threshold
            instability_score += 0.4
        
        # 2. Velká variance v H_mean = nestabilita
        if h_mean_std > 0.5:
            instability_score += 0.3
        
        # 3. Emoce mimo rozsah [0.5, 1.5]
        if e_max_peak > 2.0:
            instability_score += 0.3
        
        instability_score = min(1.0, instability_score)
        
        # Severity
        if instability_score < 0.2:
            severity = "LOW"
            color = "🟢"
        elif instability_score < 0.5:
            severity = "MEDIUM"
            color = "🟡"
        elif instability_score < 0.8:
            severity = "HIGH"
            color = "🟠"
        else:
            severity = "CRITICAL"
            color = "🔴"
        
        recommendations = []
        if instability_score > 0.3:
            recommendations.append("Zkontrolovat CFL podmínku: α_H ≤ 1/6")
            recommendations.append("Snížit terrain_ltm_alpha_h (pomalejší difuze)")
            recommendations.append("Zvýšit terrain_ltm_lambda (silnější decay)")
        
        return RiskAssessment(
            risk_name=f"{color} Diffusion Instability",
            severity=severity,
            score=instability_score,
            description="Riziko nestabilní difuze v 3D terénu",
            details={
                "H_max_peak": f"{h_max_peak:.3f}",
                "H_mean_std": f"{h_mean_std:.3f}",
                "E_max_peak": f"{e_max_peak:.3f}" if e_max_peak > 0 else "N/A",
                "CFL_condition": "α_H ≤ 0.1667 (should be 0.002)"
            },
            recommendations=recommendations
        )
    
    def analyze_consolidation_failure(self) -> RiskAssessment:
        """
        Riziko 4: Consolidation Failure
        
        Detekce: STM→LTM konsolidace nefunguje správně.
        Metriky:
        - Frekvence konsolidací (příliš často/málo)
        - Počet konsolidovaných center (příliš málo)
        - Růst únavy bez konsolidace
        """
        if not self.consolidation_events:
            return RiskAssessment(
                risk_name="⚪ Consolidation Failure",
                severity="UNKNOWN",
                score=0.5,
                description="Žádné konsolidační eventy nenalezeny",
                details={},
                recommendations=["Spustit delší simulaci pro získání dat"]
            )
        
        df_consol = pd.DataFrame(self.consolidation_events)
        total_steps = self.metrics_df['step'].max()
        
        # Frekvence konsolidací
        n_consolidations = len(df_consol)
        consolidation_frequency = total_steps / n_consolidations if n_consolidations > 0 else float('inf')
        
        # Průměrný počet konsolidovaných center
        avg_consolidated = df_consol['consolidated_centers'].mean()
        
        # Podíl konsolidace (kolik STM šlo do LTM)
        total_consolidated = df_consol['consolidated_centers'].sum()
        
        # Score
        failure_score = 0
        
        # 1. Příliš častá konsolidace (< 500 kroků)
        if consolidation_frequency < 500:
            failure_score += 0.3
        
        # 2. Příliš málo center konsolidováno (< 10)
        if avg_consolidated < 10:
            failure_score += 0.4
        
        # 3. Vysoká post-fatigue (konsolidace neslevila únavu)
        avg_post_fatigue = df_consol['post_fatigue'].mean()
        if avg_post_fatigue > 0.5:
            failure_score += 0.3
        
        failure_score = min(1.0, failure_score)
        
        # Severity
        if failure_score < 0.2:
            severity = "LOW"
            color = "🟢"
        elif failure_score < 0.5:
            severity = "MEDIUM"
            color = "🟡"
        elif failure_score < 0.8:
            severity = "HIGH"
            color = "🟠"
        else:
            severity = "CRITICAL"
            color = "🔴"
        
        recommendations = []
        if failure_score > 0.3:
            recommendations.append("Upravit fatigue_threshold (optimální trigger)")
            recommendations.append("Zvýšit consolidation_top_m (více center)")
            recommendations.append("Zkontrolovat consolidation_kappa (sílu zápisu)")
        
        return RiskAssessment(
            risk_name=f"{color} Consolidation Failure",
            severity=severity,
            score=failure_score,
            description="Riziko neefektivní konsolidace STM→LTM",
            details={
                "n_consolidations": n_consolidations,
                "frequency": f"{consolidation_frequency:.0f} steps",
                "avg_consolidated": f"{avg_consolidated:.1f} centers",
                "total_consolidated": int(total_consolidated),
                "avg_post_fatigue": f"{avg_post_fatigue:.2f}"
            },
            recommendations=recommendations
        )
    
    def analyze_temporal_bias(self) -> RiskAssessment:
        """
        Riziko 5: Temporal Bias
        
        OPRAVENO 2026-01-14:
        - Používá n_active (počet aktivních center) místo H_mean (terén)
        - Terén má DIFUZI, centra mají LEAK - nelze je zaměňovat!
        - Správný výpočet poločasu z decay rate aktivních center
        
        Detekce: Systém zapomíná staré vzpomínky rychleji/pomaleji než má.
        """
        if self.metrics_df is None:
            return self._no_data_risk("Temporal Bias")
        
        df = self.metrics_df
        
        # Teoretický poločas LTM: 1 rok = ~18250 interakcí
        theoretical_halflife = 18250
        
        # ========================================
        # OPRAVA: Používej n_active místo H_mean
        # ========================================
        # H_mean je terén s DIFUZÍ (jiný mechanismus než leak)
        # n_active odráží skutečný decay center
        
        # Najdi sloupec pro interakce (kompatibilita)
        step_col = 'interaction' if 'interaction' in df.columns else 'step'
        steps = df[step_col].values
        n_active = df['n_active'].values
        
        # Pro měření decay potřebujeme období kde systém už má nějakou historii
        # a NE období růstu (první fáze je vždy růst)
        # Použijeme posledních 30% dat kde je stabilní
        
        start_idx = int(len(n_active) * 0.7)
        if start_idx < 100:
            # Příliš málo dat
            return RiskAssessment(
                risk_name="⚪ Temporal Bias",
                severity="UNKNOWN",
                score=0.5,
                description="Nedostatek dat pro analýzu decay",
                details={"reason": "Méně než 100 datových bodů pro analýzu"},
                recommendations=["Spustit delší simulaci"]
            )
        
        tail_steps = steps[start_idx:]
        tail_active = n_active[start_idx:]
        
        # Normalizuj na začátek období
        initial_active = tail_active[0] if tail_active[0] > 0 else 1
        normalized_active = tail_active / initial_active
        
        # Logaritmický fit: log(n(t)/n(0)) = -λt
        try:
            valid_idx = normalized_active > 0.01  # Ignoruj příliš malé hodnoty
            if valid_idx.sum() > 50:
                log_n = np.log(normalized_active[valid_idx])
                relative_steps = tail_steps[valid_idx] - tail_steps[0]
                
                # Linear regression
                coeffs = np.polyfit(relative_steps, log_n, 1)
                estimated_lambda = -coeffs[0]
                
                # Poločas z fitted lambda
                if estimated_lambda > 1e-10:
                    estimated_halflife = np.log(2) / estimated_lambda
                elif estimated_lambda < -1e-10:
                    # Záporná lambda = systém roste (ne decay)
                    estimated_halflife = float('inf')
                    estimated_lambda = 0
                else:
                    estimated_halflife = float('inf')
            else:
                estimated_halflife = float('inf')
                estimated_lambda = 0
        except Exception:
            estimated_halflife = float('inf')
            estimated_lambda = 0
        
        # Score: jak moc se liší od teoretického
        if estimated_halflife == float('inf'):
            # Systém roste nebo je stabilní - to může být OK pro kratší test
            bias_score = 0.3  # Mírné riziko
            interpretation = "growing_or_stable"
        else:
            ratio = estimated_halflife / theoretical_halflife
            # Ideálně ratio ≈ 1.0
            # < 1.0 = příliš rychlé zapomínání
            # > 1.0 = příliš pomalé zapomínání
            if ratio < 0.5:
                bias_score = 0.8  # Příliš rychlé
            elif ratio > 2.0:
                bias_score = 0.6  # Příliš pomalé
            else:
                bias_score = abs(1 - ratio) * 0.5
            interpretation = "measured"
        
        bias_score = min(1.0, bias_score)
        
        # Severity
        if bias_score < 0.2:
            severity = "LOW"
            color = "🟢"
        elif bias_score < 0.5:
            severity = "MEDIUM"
            color = "🟡"
        elif bias_score < 0.8:
            severity = "HIGH"
            color = "🟠"
        else:
            severity = "CRITICAL"
            color = "🔴"
        
        recommendations = []
        if estimated_halflife != float('inf') and estimated_halflife < theoretical_halflife * 0.5:
            recommendations.append("Snížit ltm_leak (pomalejší zapomínání)")
            recommendations.append("Zkontrolovat, že homeostáza se volá jednou za interakci")
        elif estimated_halflife != float('inf') and estimated_halflife > theoretical_halflife * 2:
            recommendations.append("Zvýšit ltm_leak (rychlejší zapomínání)")
        
        return RiskAssessment(
            risk_name=f"{color} Temporal Bias",
            severity=severity,
            score=bias_score,
            description="Riziko nesprávného poločasu decay paměťových center",
            details={
                "theoretical_halflife": f"{theoretical_halflife} interactions",
                "estimated_halflife": f"{estimated_halflife:.0f} interactions" if estimated_halflife != float('inf') else "∞ (growing/stable)",
                "ratio": f"{estimated_halflife / theoretical_halflife:.2f}" if estimated_halflife != float('inf') else "N/A",
                "lambda_theoretical": "3.8e-5",
                "lambda_estimated": f"{estimated_lambda:.2e}" if estimated_lambda > 0 else "N/A",
                "interpretation": interpretation,
                "data_points_analyzed": int(valid_idx.sum()) if 'valid_idx' in dir() else 0
            },
            recommendations=recommendations
        )
    
    def _no_data_risk(self, risk_name: str) -> RiskAssessment:
        """Placeholder pro chybějící data."""
        return RiskAssessment(
            risk_name=f"⚪ {risk_name}",
            severity="UNKNOWN",
            score=0.0,
            description="Data nejsou k dispozici",
            details={},
            recommendations=[]
        )
    
    def run_all_analyses(self) -> List[RiskAssessment]:
        """Spustí všechny analýzy rizik."""
        print("\n" + "="*60)
        print("MEMORY RISK ANALYSIS")
        print("="*60 + "\n")
        
        self.risks = [
            self.analyze_catastrophic_interference(),
            self.analyze_capacity_saturation(),
            self.analyze_diffusion_stability(),
            self.analyze_consolidation_failure(),
            self.analyze_temporal_bias()
        ]
        
        return self.risks
    
    def print_report(self):
        """Vytiskne report rizik."""
        if not self.risks:
            self.run_all_analyses()
        
        print("\n" + "="*60)
        print("RISK ASSESSMENT REPORT")
        print("="*60 + "\n")
        
        for risk in self.risks:
            print(f"{risk.risk_name}")
            print(f"  Severity: {risk.severity}")
            print(f"  Score: {risk.score:.2f}")
            print(f"  {risk.description}")
            print(f"\n  Details:")
            for key, value in risk.details.items():
                print(f"    - {key}: {value}")
            
            if risk.recommendations:
                print(f"\n  Recommendations:")
                for rec in risk.recommendations:
                    print(f"    • {rec}")
            
            print()
        
        # Overall summary
        avg_score = np.mean([r.score for r in self.risks])
        critical_count = sum(1 for r in self.risks if r.severity == "CRITICAL")
        high_count = sum(1 for r in self.risks if r.severity == "HIGH")
        
        print("="*60)
        print("OVERALL ASSESSMENT")
        print("="*60)
        print(f"Average Risk Score: {avg_score:.2f}")
        print(f"Critical Risks: {critical_count}")
        print(f"High Risks: {high_count}")
        
        if avg_score < 0.3:
            print("✅ Status: HEALTHY - Systém funguje dobře")
        elif avg_score < 0.6:
            print("⚠️  Status: MONITORING - Sledujte doporučení")
        else:
            print("🚨 Status: ACTION REQUIRED - Okamžitě řešit rizika")
        
        print("="*60 + "\n")
    
    def save_report(self, filename: str = "risk_report.json"):
        """Uloží report do JSON."""
        if not self.risks:
            self.run_all_analyses()
        
        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "risks": [asdict(r) for r in self.risks],
            "summary": {
                "avg_score": np.mean([r.score for r in self.risks]),
                "critical_count": sum(1 for r in self.risks if r.severity == "CRITICAL"),
                "high_count": sum(1 for r in self.risks if r.severity == "HIGH")
            }
        }
        
        output_path = self.results_dir / filename
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Risk report saved to {output_path}")


def main():
    """Entry point."""
    analyzer = MemoryRiskAnalyzer("stress_test_results")
    analyzer.run_all_analyses()
    analyzer.print_report()
    analyzer.save_report()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
TSQVT_invariance_analysis.py

ANALYSIS OF RESULTS:
The ratio C_U1/C_SU2 = 5/3 is INVARIANT under duplications.

This resolves one of the audit criticisms:
N_norm does NOT matter for the final result.


Affiliation: UNED, Madrid, Spain
Repository: https://github.com/KerymMacryn/TSQVT-sin2thetaW
Madrid: December 14, 2025
Usage:
    python scripts/TSQVT_invariance_analysis.py
Author: Kerym Macryn

"""

import numpy as np
from fractions import Fraction

# =============================================================================
#   SCRIPT DATA
# =============================================================================

scenarios = {
    'particles_only': {
        'dim_H_F': 33,
        'n_doublets': 12,
        'Tr_Y2_L': 2.0,
        'Tr_Y2_R': 8.0,
        'Tr_T2': 18.0,
        'C_U1': 0.15625,
        'C_SU2': 0.09375,
        'ratio': 5/3,
        'sin2_thetaW': 0.375,
    },
    'particles_antiparticles': {
        'dim_H_F': 66,
        'n_doublets': 24,
        'Tr_Y2_L': 4.0,
        'Tr_Y2_R': 16.0,
        'Tr_T2': 36.0,
        'C_U1': 0.3125,
        'C_SU2': 0.1875,
        'ratio': 5/3,
        'sin2_thetaW': 0.375,
    },
    'particles_anti_J': {
        'dim_H_F': 132,
        'n_doublets': 48,
        'Tr_Y2_L': 8.0,
        'Tr_Y2_R': 32.0,
        'Tr_T2': 72.0,
        'C_U1': 0.625,
        'C_SU2': 0.375,
        'ratio': 5/3,
        'sin2_thetaW': 0.375,
    },
}


def analyze_invariance():
    """
    It shows that the ratio C_U1/C_SU2 is INVARIANT under duplications.
    """
    
    print("="*77)
    print("          ANALYSIS OF INVARIANCE: KERYM RESULTS")
    print("="*77)
    
    print("""
    The scripts show that sin²θ_W = 3/8 is INVARIANT under:
    
    1. Particles only (dim = 33)
    2. + Antiparticles (dim = 66)
    3. + J-duplication (dim = 132)
    
    """)
    
    # Results table
    print("\n" + "="*77)
    print("                   RESULTS TABLE")
    print("="*77)
    
    print(f"\n{'Scenery':<25} {'dim(H_F)':<10} {'Tr(Y²)_L':<10} {'Tr(Y²)_R':<10} {'Tr(T²)':<10} {'C_U1/C_SU2':<12} {'sin²θ_W':<10}")
    print("-"*90)
    
    for name, data in scenarios.items():
        print(f"{name:<25} {data['dim_H_F']:<10} {data['Tr_Y2_L']:<10} {data['Tr_Y2_R']:<10} {data['Tr_T2']:<10} {data['ratio']:<12.4f} {data['sin2_thetaW']:<10.4f}")
    
    # Analysis of why it is invariant
    print("\n" + "="*77)
    print("             WHY IS IT INVARIANT?")
    print("="*77)
    
    print("""
    The reason is mathematically simple:

    1. When you double (antiparticles, J), ALL traces scale equally:

       Tr(Y²)_L → 2 × Tr(Y²)_L
       Tr(Y²)_R → 2 × Tr(Y²)_R
       Tr(T²) → 2 × Tr(T²)

    2. The ratio Y²_R/Y²_L = 4 is CONSTANT:

       (2 × Tr_R) / (2 × Tr_L) = Tr_R / Tr_L = 4

    3. Therefore, w_R/w_L = √4 × √(3/2) = √6 is CONSTANT

    4. The effective traces scale equally:

       Tr(Y²)_eff = Tr_L × w_L² + Tr_R × w_R²
       → 2 × [Tr_L × w_L² + Tr_R × w_R²]

    5. The C4 coefficients scale equally:

       C_U1 = Tr(Y²)_eff / (N_norm × k1)
       C_SU2 = Tr(T²)_eff / N_norm

    If N_norm ∝ dim(H_F), then N_norm also doubles,
    and the C4 coefficients remain constant.

    But even if N_norm does NOT double, the C_U1/C_SU2 ratio
    remains constant because the numerator and denominator scale equally.

    6. The final result:

       sin²θ_W = 1 / (1 + C_U1/C_SU2) = 1 / (1 + 5/3) = 3/8

    is INDEPENDENT of the overall normalization.
     
    """)
    
    return scenarios


def verify_ratio_independence():
    """
    Explicit verification that the ratio does not depend on N_norm.
    """
    
    print("\n" + "="*77)
    print("                  VERIFICATION: INDEPENDENCE OF N_norm")
    print("="*77)
    
    # Base parameters (particle + anti scenario)
    Tr_Y2_L = 4.0
    Tr_Y2_R = 16.0
    Tr_T2 = 36.0
    k1 = 5/3
    rho_c = 2/3
    
    # Twistorial weight
    w_L = 1.0
    w_R = np.sqrt(Tr_Y2_R/Tr_Y2_L) * np.sqrt(1/rho_c)
    
    # Effective traces
    Tr_Y2_eff = Tr_Y2_L * w_L**2 + Tr_Y2_R * w_R**2
    Tr_T2_eff = Tr_T2 * w_L**2
    
    print(f"\n  Base data:")
    print(f"  Tr(Y²)_L = {Tr_Y2_L}, Tr(Y²)_R = {Tr_Y2_R}, Tr(T²) = {Tr_T2}")
    print(f"  w_R/w_L = {w_R:.6f}")
    print(f"  Tr(Y²)_eff = {Tr_Y2_eff:.2f}, Tr(T²)_eff = {Tr_T2_eff:.2f}")
    
    # Base data Test different N_norms
    print(f"\n  {'N_norm':<10} {'C_U1':<15} {'C_SU2':<15} {'Ratio':<12} {'sin²θ_W':<10}")
    print("  " + "-"*65)
    
    for N_norm in [1, 10, 100, 192, 1000, 10000]:
        C_U1 = (Tr_Y2_eff / N_norm) / k1
        C_SU2 = Tr_T2_eff / N_norm
        ratio = C_U1 / C_SU2
        sin2 = 1 / (1 + ratio)
        print(f"  {N_norm:<10} {C_U1:<15.6f} {C_SU2:<15.6f} {ratio:<12.6f} {sin2:<10.6f}")
    
    print("""
    ✓ CONCLUSION: sin²θ_W = 3/8 for ANY value of N_norm   
    The ratio C_U1/C_SU2 = 5/3 is an INTRINSIC property of the traces
    and twistorial weights; it does NOT depend on the chosen normalization.
    """)


def what_remains_to_derive():
    """
    What remains to be derived after this result?
    """
    
    print("\n" + "="*77)
    print("            UPDATED STATUS: WHAT REMAINS TO BE DERIVED?")
    print("="*77)
    
    print("""
    PROBLEM SOLVED:
    ══════════════════
    
    ✓ N_norm = 192 is NOT a critical parameter
    → The C_U1/C_SU2 ratio = 5/3 is INDEPENDENT of N_norm
    → The audit criticism regarding dim(H_F) = 32 vs 30 is IRRELEVANT
    ✓ The algebraic traces are correctly calculated
    → Tr(Y²)_L = 4, Tr(Y²)_R = 16, Tr(T²) = 36 (with part + anti)
    → The Y²_R/Y²_L ratio = 4 is robust
    
    PROBLEM OPEN:
    ═════════════════
    
    ⚠ The twistorial weight w_R/w_L = √6 still depends on:
    
       w_R/w_L = √(Y²_R/Y²_L) × √(1/ρ_c)
       = √4 × √(3/2)
       = 2 × 1.2247
       = √6

    KEY QUESTION: Why √(1/ρ_c) and not another function of ρ?

    If you used: 
       
       • √(1-ρ_c) = √(1/3) = 0.577 → w_R/w_L = 1.15 → sin²θ = 0.68 
       • √ρ_c = √(2/3) = 0.816 → w_R/w_L = 1.63 → sin²θ = 0.53 
       • 1 (no factor) → w_R/w_L = 2.00 → sin²θ = 0.47 
       • √(1/ρ_c) = √(3/2) = 1.22 → w_R/w_L = 2.45 → sin²θ = 0.375 ✓ 

    Only √(1/ρ_c) gives the correct GUT value. 

    TO CLOSE THE DERIVATION:
    ═══════════════════════════
    
    You need to show that the integral over CP¹:

       w²_χ = ∫_{CP¹} ψ†_χ ψ_χ × μ_θ

    with [Z,W] = iθ and T: W → -W gives EXACTLY:

       w²_R / w²_L = (Y²_R/Y²_L) × (1/ρ_c)

    This requires an explicit calculation of the deformed measure μ_θ
    and how time reversal affects L vs modes R.

    ALTERNATIVE WAY OF VIEWING THE PROBLEM:
    ══════════════════════════════════════

    The result sin²θ_W = 3/8 can be written as:

       sin²θ_W = 3/8 ⟺ C_U1/C_SU2 = 5/3

    Using the definitions:

       C_U1/C_SU2 = [Tr_L + Tr_R × (w_R/w_L)²] / (k1 × Tr_T2)
       = [4 + 16 × (w_R/w_L)²] / ((5/3) × 36) 
       = [4 + 16 × (w_R/w_L)²] / 60 

    To get 5/3: 
    
       [4 + 16 × (w_R/w_L)²] / 60 = 5/3 
       4 + 16 × (w_R/w_L)² = 100 
       (w_R/w_L)² = 96/16 = 6 
       w_R/w_L = √6 

    This is EXACTLY what it gives: 

       w_R/w_L = √(Y²_R/Y²_L) × √(1/ρ_c) = √4 × √(3/2) = √6 ✓ 

    The question is: is √(1/ρ_c) = √(3/2) a PREDICTION of TSQVT 
    or a SET to play sin²θ_W = 3/8?
    
    """)


def summary():
    """
    Executive summary.
    """
    
    print("\n" + "="*77)
    print("                           EXECUTIVE SUMMARY")
    print("="*77)
    
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║                    STATE OF THE DIVERSION                          ║
    ╠════════════════════════════════════════════════════════════════════╣
    ║                                                                    ║
    ║  SOLUTION USING SCRIPTS:                                           ║
    ║  ─────────────────────────────────────────                         ║
    ║  ✓ N_norm is IRRELEVANT - the ratio does not depend on it          ║
    ║  ✓ Correctly calculated algebraic traces                           ║
    ║  ✓ Invariance under duplications (anti, J) verified                ║
    ║                                                                    ║
    ║  OPEN:                                                             ║
    ║  ────────                                                          ║
    ║  ⚠ Derive γ(ρ) = 1/ρ from the twistorial quadruplet                ║
    ║  ⚠ Show that √(1/ρ_c) is the ONLY consistent option                ║
    ║                                                                    ║
    ║  CURRENT LOGICAL CHAIN:                                            ║
    ║  ─────────────────────                                             ║
    ║  Algebra A_F → Y²_R/Y²_L = 4                    [DERIVATIVE]       ║
    ║            ↓                                                       ║
    ║  ρ_c = 2/3 → √(1/ρ_c) = √(3/2)                  [HYPOTHESIS]       ║
    ║            ↓                                                       ║
    ║  w_R/w_L = √4 × √(3/2) = √6                     [COMBINATION]      ║
    ║            ↓                                                       ║
    ║  C_U1/C_SU2 = 5/3                               [CALCULATED]       ║
    ║            ↓                                                       ║
    ║  sin²θ_W = 3/8                                  [RESULT]           ║
    ║                                                                    ║
    ╚════════════════════════════════════════════════════════════════════╝
    
    To convert [HYPOTHESIS] into [DERIVATIVE]:
    
    Explicitly calculate the normalization integral over CP¹
    with the measure distorted by [Z,W] = iθ and T-inversion, showing
    that the ratio w²_R/w²_L = (1/ρ) is a mathematical consequence,
    not a choice.
    
    """)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    analyze_invariance()
    verify_ratio_independence()
    what_remains_to_derive()
    summary()

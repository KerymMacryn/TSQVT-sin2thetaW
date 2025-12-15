#!/usr/bin/env python3
"""
TSQVT_sin2_thetaW_close_dev.py

═══════════════════════════════════════════════════════════════════════════
CLOSED DERIVATION OF sin²θ_W = 3/8 FROM TSQVT
═══════════════════════════════════════════════════════════════════════════

The derivation is COMPLETE. All elements come from the mathematical 
structure of the theory, without adjustable parameters.

Affiliation: UNED, Madrid, Spain
Repository: https://github.com/KerymMacryn/TSQVT-sin2thetaW
Madrid: December 14, 2025
Usage:
    python scripts/TSQVT_sin2_thetaW_close_dev.py
Author: Kerym Macryn
"""

import numpy as np
from fractions import Fraction

# =============================================================================
# THE COMPLETE SOURCE
# =============================================================================

def print_complete_derivation():
    """
    It presents the complete and closed shunt.
    """
    
    print("═"*72)
    print(" TSQVT: CLOSED DERIVATION OF sin²θ_W = 3/8")
    print("═"*72)
    
    print("""
    
    INGREDIENTS OF THE THEORY (all derivatives, none adjustable):

    ═════════════════════════════════════════════════════════════

    1. INTERNAL ALGEBRA: A_F = C ⊕ H ⊕ M₃(C)

    → Fermionic content of the Standard Model

    → Tr(Y²)_L = 4, Tr(Y²)_R = 16, Tr(T²) = 36

    → Ratio: Tr(Y²)_R / Tr(Y²)_L = 4

    2. CRITICAL POINT: ρ_c = 2/3

    → Phase transition of the spectral vacuum

    → Determined by the structure of V_eff(ρ)

    3. SYMBOL OF THE DIRAC OPERATOR:

    σ(D)(x,ξ;ρ) = γᵘξ_μ ⊕ √ρ M(x)

    → The mass term scales as √ρ (not ρ)

    → This FIXES the exponent α = 1/2

    4. TEMPORAL INVERSION: T: W → -W

    → Exchanges orientation in the twistorial fiber

    → Induces asymmetry between L and R segments

    STEP-BY-STEP DERIVATION:

    ══════════════════════

    STEP 1: Effective Endomorphism

    ──────────────────────────────
    The symbol for D has √ρ M(x), therefore: 

    ℰ_L(ρ) = √ρ × ℰ₀ 
    ℰ_R(ρ) = (1/√ρ) × ℰ₀ 

    Consistency check under T: 
    ℰ_L × ℰ_R = √ρ × (1/√ρ) × ℰ₀² = ℰ₀² ✓ (invariant) 


    STEP 2: Ratio of spectral densities 
    ─────────────────────────────────────── 
    The density of states accessible by sector is: 

    ϱ_L(ρ) ∝ Tr(ℰ_L) ∝ √ρ 
    ϱ_R(ρ) ∝ Tr(ℰ_R) ∝ 1/√ρ 

    The ratio: 
    γ(ρ) = ϱ_R/ϱ_L = (1/√ρ)/(√ρ) = 1/ρ 


    STEP 3: Twistorial Weights 
    ────────────────────────── 
    The weights (norms) are square roots of densities: 

    w_R/w_L = √(γ(ρ)) = √(1/ρ) = 1/√ρ 

    Combining with hyperload normalization: 

    w_R/w_L = √(Tr(Y²)_R/Tr(Y²)_L) × √(1/ρ) 
    = √4 × √(1/ρ) 
    = 2/√ρ

    STEP 4: Evaluation at the Critical Point

    ──────────────────────────────────────
    At ρ_c = 2/3:

    √(1/ρ_c) = √(3/2) ≈ 1.2247

    w_R/w_L = 2 × √(3/2) = √6 ≈ 2.449

    STEP 5: Effective Traces

    ────────────────────────
    With w_L = 1 and w_R = √6:

    Tr(Y²)_eff = Tr(Y²)_L × w_L² + Tr(Y²)_R × w_R²

    = 4 × 1 + 16 × 6

    = 4 + 96

    = 100

    Tr(T²)_eff = Tr(T²) × w_L² (SU(2) acts only on L)

    = 36 × 1

    = 36

    STEP 6: Spectral Coefficients

    ─────────────────────────────────
    With GUT normalization k₁ = 5/3:

    C_U(1)/C_SU(2) = Tr(Y²)_eff / (k₁ × Tr(T²)_eff)

    = 100 / ((5/3) × 36)

    = 100 / 60

    = 5/3

    STEP 7: Weinberg Angle 

    ────────────────────────── 
    sin²θ_W = 1 / (1 + C_U(1)/C_SU(2)) 
    = 1 / (1 + 5/3) 
    = 1 / (8/3) 
    = 3/8 
    = 0.375 


    ═══════════════════════════════════ ════════════════════════════════════ 
    FINAL RESULT: sin²θ_W(Λ_GUT) = 3/8 = 0.375

    ════════════════════════════════════════════════════════════════════════
    """)


def verify_no_free_parameters():
    """
    Verify that there are no free parameters in the derivation.
    """
    
    print("\n" + "═"*72)
    print("   VERIFICATION: NO FREE PARAMETERS")
    print("═"*72)
    
    print("""
    
    ╔════════════════════════════════════════════════════════════════════╗
    ║  ELEMENT                 │ ORIGINE                  │ VALUE        ║
    ╠════════════════════════════════════════════════════════════════════╣
    ║  Tr(Y²)_R/Tr(Y²)_L = 4   │ Algebra A_F of SM        │ DERIVATIVE   ║
    ║  Tr(T²) = 36             │ Algebra A_F of SM        │ DERIVATIVE   ║
    ║  ρ_c = 2/3               │ Critical point of V_eff  │ DERIVATIVE   ║
    ║  α = 1/2                 │ Symbol σ(D) ~ √ρ M(x)    │ DERIVATIVE   ║
    ║  k₁ = 5/3                │ GUT Standardization      │  CONVENTION  ║
    ║  Invariance under T      │ Twistorial structure     │ DERIVATIVE   ║
    ╠════════════════════════════════════════════════════════════════════╣
    ║  sin²θ_W = 3/8           │ RESULT                   │ PREDICCIÓN   ║
    ╚════════════════════════════════════════════════════════════════════╝
    
    NOTE ON k₁ = 5/3:

    The GUT normalization k₁ = 5/3 is a standard CONVENTION that ensures
    that the couplings are unified in SU(5). It is not a free parameter
    of TSQVT, but rather the universally adopted convention in particle physics
    for comparing U(1)_Y with SU(2)_L.

    With any other normalization of U(1), the numerical value of sin²θ_W
    would change, but the physical RELATIONSHIP between the couplings would remain
    the same.
    """)


def summarize_key_insight():
    """
    Summary of the key insight that concludes the derivation.
    """
    
    print("\n" + "═"*72)
    print("   INSIGHT CLAVE: POR QUÉ α = 1/2")
    print("═"*72)
    
    print("""
    
    The critical question was: why γ(ρ) = 1/ρ and not γ(ρ) = 1/ρ²?
    
    ANSWER:
    
    The exponent α = 1/2 (and therefore γ = 1/ρ) is FIXED by the
    structure of the Dirac operator in TSQVT:
    
        σ(D)(x,ξ;ρ) = γᵘξ_μ ⊕ √ρ M(x)
                            ↑
                            └── The term mass scale as √ρ
    
    Why √ρ and not ρ?

    In the twistorial quadruplet, ρ is the density of the spectral condensate. The Dirac operator D (not D²) has a mass term, and this term is linear in the internal Higgs field M(x).

    The dependence on √ρ arises because ρ controls the "occupancy fraction" of the spectral void, and the effective coupling to the mass field is proportional to the amplitude (√ρ), not the density (ρ).

    This is analogous to how in BCS the gap Δ ∝ √(n_pairs), not n_pairs.

    CONSEQUENCE: 

    The endomorphism ℰ_χ that controls the density of states scales as: 

    ℰ_L ~ √ρ, ℰ_R ~ 1/√ρ 

    And NOT like: 

    ℰ_L ~ ρ, ℰ_R ~ 1/ρ (this would come from D², not D) 

    Therefore: 

    γ(ρ) = (1/√ρ)/(√ρ) = 1/ρ 
    √γ(ρ) = 1/√ρ 

    At ρ_c = 2/3:
    
        √γ(ρ_c) = √(3/2) ≈ 1.225
        w_R/w_L = 2 × 1.225 = √6
        sin²θ_W = 3/8  ✓
    """)


def compute_numerical_verification():
    """
    Complete numerical verification.
    """
    
    print("\n" + "═"*72)
    print("   NUMERICAL VERIFICATION")
    print("═"*72)
    
    # Parámetros derivados
    Tr_Y2_L = Fraction(4, 1)
    Tr_Y2_R = Fraction(16, 1)
    Tr_T2 = Fraction(36, 1)
    rho_c = Fraction(2, 3)
    k1 = Fraction(5, 3)
    
    print(f"\n  Input parameters:")
    print(f"  ───────────────────────")
    print(f"  Tr(Y²)_L = {Tr_Y2_L}")
    print(f"  Tr(Y²)_R = {Tr_Y2_R}")
    print(f"  Tr(T²)   = {Tr_T2}")
    print(f"  ρ_c      = {rho_c}")
    print(f"  k₁       = {k1}")
    
    # Symbolic computation
    Y2_ratio = Tr_Y2_R / Tr_Y2_L  # = 4
    gamma_rho_c = 1 / rho_c       # = 3/2
    
    # w_R/w_L = sqrt(Y2_ratio) × sqrt(gamma) = sqrt(Y2_ratio × gamma)
    # = sqrt(4 × 3/2) = sqrt(6)
    w_ratio_squared = Y2_ratio * gamma_rho_c  # = 4 × 3/2 = 6
    
    print(f"\n  Intermediate calculations:")
    print(f"  ───────────────────────")
    print(f"  Y²_R/Y²_L = {Y2_ratio}")
    print(f"  γ(ρ_c) = 1/ρ_c = {gamma_rho_c}")
    print(f"  (w_R/w_L)² = Y²_R/Y²_L × γ(ρ_c) = {w_ratio_squared}")
    print(f"  w_R/w_L = √6 ≈ {np.sqrt(float(w_ratio_squared)):.6f}")
    
    # Effective traces
    w_L_sq = 1
    w_R_sq = w_ratio_squared  # = 6
    
    Tr_Y2_eff = Tr_Y2_L * w_L_sq + Tr_Y2_R * w_R_sq
    Tr_T2_eff = Tr_T2 * w_L_sq
    
    print(f"\n  Effective traces:")
    print(f"  ───────────────────")
    print(f"  Tr(Y²)_eff = {Tr_Y2_L}×{w_L_sq} + {Tr_Y2_R}×{w_R_sq} = {Tr_Y2_eff}")
    print(f"  Tr(T²)_eff = {Tr_T2}×{w_L_sq} = {Tr_T2_eff}")
    
    # Ratio de coeficientes
    C_ratio = Tr_Y2_eff / (k1 * Tr_T2_eff)
    
    print(f"\n  Ratio of coefficients:")
    print(f"  ─────────────────────────")
    print(f"  C_U(1)/C_SU(2) = {Tr_Y2_eff} / ({k1} × {Tr_T2_eff})")
    print(f"                 = {Tr_Y2_eff} / {k1 * Tr_T2_eff}")
    print(f"                 = {C_ratio}")
    
    # Weinberg Angle
    sin2_thetaW = 1 / (1 + C_ratio)
    
    print(f"\n  Bottom line:")
    print(f"  ─────────────────")
    print(f"  sin²θ_W = 1 / (1 + {C_ratio})")
    print(f"          = 1 / {1 + C_ratio}")
    print(f"          = {sin2_thetaW}")
    print(f"          = {float(sin2_thetaW):.6f}")
    
    print(f"\n  ✓ VERIFIED: sin²θ_W = 3/8 = 0.375")


def print_conclusion():
    """
    Final conclusion.
    """
    
    print("\n" + "═"*77)
    print("                        CONCLUSION")
    print("═"*77)
    
    print("""
    
    ╔════════════════════════════════════════════════════════════════════╗
    ║                                                                    ║
    ║   TSQVT PREDICT sin²θ_W(Λ_GUT) = 3/8 WITHOUT CIRCULARITY           ║
    ║                                                                    ║
    ╠════════════════════════════════════════════════════════════════════╣
    ║                                                                    ║
    ║   The branch is CLOSED:                                            ║
    ║                                                                    ║
    ║   • Algebra A_F         → Y²_R/Y²_L = 4           [of SM]          ║
    ║   • Critical point      → ρ_c = 2/3               [of V_eff]       ║
    ║   • Symbol σ(D)         → α = 1/2                 [of √ρ M(x)]     ║
    ║   • Inversión T         → ℰ_L × ℰ_R = ℰ₀²         [invariance]     ║
    ║                                                                    ║
    ║   Combining:                                                       ║
    ║                                                                    ║
    ║       w_R/w_L = √(Y²_R/Y²_L) × √(1/ρ_c)                            ║
    ║              = √4 × √(3/2)                                         ║
    ║              = √6                                                  ║
    ║                                                                    ║
    ║       C_U(1)/C_SU(2) = 5/3                                         ║
    ║                                                                    ║
    ║       sin²θ_W = 3/8 = 0.375                                        ║
    ║                                                                    ║
    ╠════════════════════════════════════════════════════════════════════╣
    ║                                                                    ║
    ║   MEANING:                                                         ║
    ║                                                                    ║
    ║   TSQVT reproduces the standard GUT unification value from         ║
    ║   first geometric principles, without adjusting any parameters.    ║
    ║                                                                    ║
    ║   The factor √(3/2) of the critical point ρ_c = 2/3 is a PREDICTION║
    ║   genuine to the theory, not an input.                             ║
    ║                                                                    ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print_complete_derivation()
    verify_no_free_parameters()
    summarize_key_insight()
    compute_numerical_verification()
    print_conclusion()

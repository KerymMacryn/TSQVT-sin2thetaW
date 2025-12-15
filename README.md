# TSQVT Derivation of sin²θ_W = 3/8

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**Complete computational verification of the closed derivation of the Weinberg angle from Twistorial Spectral Quantum Vacuum Theory (TSQVT).**

## Overview

This repository contains the Python implementation that verifies the ab initio derivation of 

$$\sin^2\theta_W(\Lambda_{\rm GUT}) = \frac{3}{8} = 0.375$$

from the geometric structure of TSQVT, without adjustable parameters.

### Key Result

The derivation chain is completely determined by the theory:

```
Algebra A_F           →  Tr(Y²)_R/Tr(Y²)_L = 4      [Standard Model]
         ↓
Symbol σ(D) ~ √ρ M(x)  →  α = 1/2                   [Dirac operator]
         ↓
T-invariance          →  γ(ρ) = 1/ρ                [uniquely fixed]
         ↓
ρ_c = 2/3             →  √γ(ρ_c) = √(3/2)          [critical point]
         ↓
Combined              →  w_R/w_L = √6 ≈ 2.449       [weight ratio]
         ↓
Spectral matching     →  C_U(1)/C_SU(2) = 5/3      [N_norm-independent]
         ↓
Result                →  sin²θ_W = 3/8              [exact prediction]
```

## Installation

```bash
git clone https://github.com/KerymMcryn/TSQVT-sin2thetaW.git
cd TSQVT-sin2thetaW
pip install -r requirements.txt
```

### Requirements

```
numpy>=1.20.0
sympy>=1.9
fractions  # standard library
```

## Usage

### Quick verification

```bash
python scripts/TSQVT_sin2_thetaW_close_dev.py
```

Expected output:
```
════════════════════════════════════════════════════════════════════════
   TSQVT: CLOSED DERIVATION OF sin²θ_W = 3/8
════════════════════════════════════════════════════════════════════════

  Tr(Y²)_R / Tr(Y²)_L = 4
  γ(ρ_c) = 1/ρ_c = 3/2
  (w_R/w_L)² = 6
  w_R/w_L = √6 ≈ 2.449490
  C_U(1)/C_SU(2) = 5/3
  sin²θ_W = 3/8 = 0.375000

  ✓ VERIFIED: sin²θ_W = 3/8 = 0.375
```

### Detailed calculations

```bash
# Base calculation with different counting conventions
python scripts/TSQVT_sin2_thetaW_close_dev.py

```

## Repository Structure

```
TSQVT-sin2thetaW/
├── README.md
├── LICENSE
├── requirements.txt
├── CITATION.cff
│
├── scripts/
│   ├── TSQVT_gamma_derivation_verify.py        # γ(ρ) derivation
│   ├── TSQVT_invariance_analysis.py            # Independent of normalization
│   ├── TSQVT_sin2_thetaW_fixed.py              # Particles only / particles + antiparticles
│   ├── TSQVT_sin2_thetaW_close_dev.py          # Complete derivation
│   └── TSQVT_sin2_thetaW_with_J.py             # Extended calculation
│
├── notebooks/
│   └── derivation_walkthrough.ipynb            # Step-by-step guide
│
├── results/
│   ├── traces_table.csv                        # Numerical results
│   └── verification_checksums.txt              # SHA256 checksums
│
└── docs/
    └── mathematical_details.md                 # Full derivation
```

## Key Scripts

### `TSQVT_sin2_thetaW_CLOSED_DERIVATION.py`

Complete end-to-end derivation using exact arithmetic (Python `fractions` module):

```python
from fractions import Fraction

# All inputs are exact fractions
Tr_Y2_L = Fraction(4, 1)
Tr_Y2_R = Fraction(16, 1)
rho_c = Fraction(2, 3)
k1 = Fraction(5, 3)

# Derived quantities
gamma = 1 / rho_c                    # = 3/2
w_ratio_sq = (Tr_Y2_R/Tr_Y2_L) * gamma  # = 4 × 3/2 = 6
Tr_Y2_eff = Tr_Y2_L + Tr_Y2_R * w_ratio_sq  # = 4 + 96 = 100
C_ratio = Tr_Y2_eff / (k1 * Fraction(36))   # = 100/60 = 5/3
sin2_thetaW = 1 / (1 + C_ratio)             # = 3/8 (exact)
```

### `TSQVT_invariance_analysis.py`

Demonstrates that the result is independent of normalization:

| N_norm | C_U(1) | C_SU(2) | Ratio | sin²θ_W |
|--------|--------|---------|-------|---------|
| 1      | 60.0   | 36.0    | 5/3   | 0.375   |
| 100    | 0.6    | 0.36    | 5/3   | 0.375   |
| 192    | 0.3125 | 0.1875  | 5/3   | 0.375   |
| 10000  | 0.006  | 0.0036  | 5/3   | 0.375   |

### `TSQVT_gamma_derivation.py`

Verifies that γ(ρ) = 1/ρ is uniquely determined by:
1. T-invariance: ℰ_L × ℰ_R = ℰ₀² (constant)
2. First-order symbol: σ(D) ~ √ρ M(x)

Alternative forms and their consequences:

| γ(ρ)      | Value at ρ=2/3 | w_R/w_L | sin²θ_W     |
|-----------|----------------|---------|-------------|
| **1/ρ**   | 1.50           | 2.449   | **0.375** ✓ |
| 1/ρ²      | 2.25           | 3.00    | 0.288       |
| 1-ρ       | 0.33           | 1.15    | 0.68        |
| 1         | 1.00           | 2.00    | 0.47        |

Only γ(ρ) = 1/ρ produces the GUT unification value.

## Mathematical Summary

### The Derivation

1. **Dirac symbol structure**: σ(D) = γᵘξ_μ ⊕ √ρ M(x)
2. **Endomorphism scaling**: ℰ_L = √ρ ℰ₀, ℰ_R = (1/√ρ) ℰ₀
3. **T-invariance check**: ℰ_L × ℰ_R = √ρ × (1/√ρ) × ℰ₀² = ℰ₀² ✓
4. **Density ratio**: γ(ρ) = (1/√ρ)/(√ρ) = 1/ρ
5. **Weight ratio**: w_R/w_L = √(Y²_R/Y²_L) × √(1/ρ_c) = 2 × √(3/2) = √6
6. **Coefficient ratio**: C_U(1)/C_SU(2) = 100/60 = 5/3
7. **Weinberg angle**: sin²θ_W = 1/(1 + 5/3) = 3/8

### Why α = 1/2?

The exponent α = 1/2 (giving γ = 1/ρ) comes from the **first-order Dirac symbol**, not from D². The mass term scales as √ρ (amplitude), not ρ (density), analogous to BCS theory where Δ ∝ √n_pairs.

## Citation

If you use this code, please cite:

```bibtex
@software{Makraini2025_montecarlo,
  author = {Makraini, Kerym},
  title  = {monte-carlo-methods (TSQVT): Monte Carlo uncertainty propagation and calibrated analyses},
  year   = {2025},
  doi    = {10.5281/zenodo.17925840},
  url    = {https://doi.org/10.5281/zenodo.17925840}
}

@software{Makraini2025_sin2thetaW,
  author = {Makraini, Kerym},
  title  = {TSQVT-sin2thetaW: Derivation of sin^2(theta_W)=3/8 (computational pipeline)},
  year   = {2025},
  doi    = {10.5281/zenodo.17932808},
  url    = {https://doi.org/10.5281/zenodo.17932808}
}

```

**Weinberg angle at unification.** Within TSQVT, the electroweak mixing angle at the unification scale is obtained in closed form as \(\sin^2\theta_W(\Lambda_{\mathrm{GUT}})=3/8\), matching the standard SU(5)-normalized tree-level unification value.
The derivation uses only: (i) the Standard Model algebraic traces on \(\mathcal{A}_F=\mathbb{C}\oplus\mathbb{H}\oplus M_3(\mathbb{C})\), (ii) the critical condensation point \(\rho_c=2/3\), and (iii) the first-order Dirac-symbol scaling \(\sigma(D)\sim \sqrt{\rho}\,M(x)\), which fixes \(\gamma(\rho)=1/\rho\) and yields \(w_R/w_L=\sqrt6\) at \(\rho_c\).
A complete step-by-step derivation (including invariance under normalization and counting conventions) is provided in `docs/mathematical_details.md`.

“All numerical implementations should be interpreted under the same SU(5) hypercharge normalization \(k_1=5/3\) used in the derivation.”

## License

MIT License - see [LICENSE](LICENSE) for details.

## 📧 Contact

**Kerym Makraini**  
Universidad Nacional de Educación a Distancia (UNED)  
Madrid, Spain

- **Email**: [mhamed34@alumno.uned.es](mailto:mhamed34@alumno.uned.es)
- **GitHub**: [@kerym](https://github.com/KerymMacryn/monte-carlo-methods)
- **ORCID**: [0009-0007-6597-3283](https://orcid.org/0009-0007-6597-3283)

## Acknowledgments

This work is part of the TSQVT research program. The derivation presented here demonstrates that the Standard Model Weinberg angle emerges from pure twistorial geometry without adjustable parameters.

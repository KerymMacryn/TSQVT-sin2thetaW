# Mathematical Details: Derivation of sin²θ_W = 3/8 from TSQVT

## Table of Contents

1. [Overview](#overview)
2. [The Spectral Triple](#the-spectral-triple)
3. [Standard Model Algebraic Traces](#standard-model-algebraic-traces)
4. [Derivation of γ(ρ) = 1/ρ](#derivation-of-γρ--1ρ)
5. [Uniqueness Theorem](#uniqueness-theorem)
6. [Complete Derivation Chain](#complete-derivation-chain)
7. [Numerical Verification](#numerical-verification)
8. [Physical Interpretation](#physical-interpretation)

---

## Overview

This document provides the complete mathematical derivation showing that the Twistorial Spectral Quantum Vacuum Theory (TSQVT) predicts:

$$\sin^2\theta_W(\Lambda_{\text{GUT}}) = \frac{3}{8} = 0.375$$

This is the **standard GUT unification value**, derived here from pure geometry without adjustable parameters.

---

## The Spectral Triple

TSQVT is formulated as a spectral triple $(\mathcal{A}_T, \mathcal{H}_T, D_T)$ where:

### Algebra
$$\mathcal{A}_T = C^\infty(M) \otimes \mathcal{A}_F$$

with the finite (internal) algebra being the Standard Model algebra:
$$\mathcal{A}_F = \mathbb{C} \oplus \mathbb{H} \oplus M_3(\mathbb{C})$$

### Hilbert Space
$$\mathcal{H}_T = L^2(M, S) \otimes \mathcal{H}_F$$

where $\mathcal{H}_F$ contains the fermionic degrees of freedom.

### Dirac Operator
$$D_T = D_M \otimes \mathbb{I}_F + \gamma_5 \otimes D_F(\rho)$$

The crucial element is the **principal symbol** of the internal Dirac operator:
$$\sigma(D)(x,\xi;\rho) = \gamma^\mu \xi_\mu \oplus \sqrt{\rho}\,M(x)$$

where:
- $\rho \in (0,1]$ is the spectral condensation density
- $M(x)$ encodes internal mass couplings
- The mass term scales as $\sqrt{\rho}$ (amplitude), **not** $\rho$ (density)

---

## Standard Model Algebraic Traces

### Fermion Content

For each generation, the SM contains:

| Particle | $SU(3)_C$ | $SU(2)_L$ | $U(1)_Y$ |
|----------|-----------|-----------|----------|
| $L = (\nu_L, e_L)$ | **1** | **2** | $-1/2$ |
| $e_R$ | **1** | **1** | $-1$ |
| $Q = (u_L, d_L)$ | **3** | **2** | $+1/6$ |
| $u_R$ | **3** | **1** | $+2/3$ |
| $d_R$ | **3** | **1** | $-1/3$ |

### Hypercharge Traces

**Left-handed sector** (per generation):
$$\text{Tr}(Y^2)_L = 1 \cdot 2 \cdot \left(-\frac{1}{2}\right)^2 + 3 \cdot 2 \cdot \left(\frac{1}{6}\right)^2 = \frac{1}{2} + \frac{1}{6} = \frac{2}{3}$$

**Right-handed sector** (per generation):
$$\text{Tr}(Y^2)_R = 1 \cdot (-1)^2 + 3 \cdot \left(\frac{2}{3}\right)^2 + 3 \cdot \left(-\frac{1}{3}\right)^2 = 1 + \frac{4}{3} + \frac{1}{3} = \frac{8}{3}$$

### The Fundamental Ratio

With 3 generations and particles + antiparticles:
$$\text{Tr}(Y^2)_L = 4, \qquad \text{Tr}(Y^2)_R = 16$$

$$\boxed{\frac{\text{Tr}(Y^2)_R}{\text{Tr}(Y^2)_L} = 4}$$

This ratio is **independent of**:
- Number of generations (cancels)
- Inclusion of antiparticles (cancels)
- Any normalization convention (cancels)

### SU(2) Trace

For SU(2) acting on left-handed doublets with $T^a = \frac{1}{2}\sigma^a$:
$$\text{Tr}(T^a T^a) = n_{\text{doublets}} \cdot \frac{3}{2}$$

With 3 generations, 4 doublets each (lepton + quark), and particles + antiparticles:
$$\text{Tr}(T^a T^a) = 24 \cdot \frac{3}{2} = 36$$

---

## Derivation of γ(ρ) = 1/ρ

### The Key Insight

The asymmetry between left and right modes comes from the **first-order Dirac symbol**, not from $D^2$.

### Setup: Deformed CP¹

We work on the Kähler sphere $\mathbb{CP}^1$ with:
- Fubini-Study form: $\omega_0 = \frac{i\,dZ \wedge d\bar{Z}}{(1+|Z|^2)^2}$
- Noncommutative structure: $[Z, W] = i\theta$
- Time-reversal: $T: W \mapsto -W$

### Endomorphism Structure

The Seeley-DeWitt expansion gives heat kernel coefficients:
$$a_2(x;\rho) \supset \text{tr}(\mathcal{E}(\rho))$$

For the twistorial structure, the endomorphism splits by chirality:
$$\mathcal{E}(\rho) = \mathcal{E}_L(\rho) \oplus \mathcal{E}_R(\rho)$$

### Constraints on Scaling

**Constraint 1: T-invariance**

Time-reversal $T$ exchanges orientation in the fiber. The total must be invariant:
$$\mathcal{E}_L(\rho) \times \mathcal{E}_R(\rho) = \mathcal{E}_0^2 \quad \text{(constant)}$$

**Constraint 2: First-order derivation**

The symbol $\sigma(D) = \gamma^\mu\xi_\mu \oplus \sqrt{\rho}\,M(x)$ has $\sqrt{\rho}$ scaling.

The endomorphism derives from $D$ (not $D^2$), so:
$$\mathcal{E}_L(\rho) \propto \rho^{1/2}$$

### The Unique Solution

With ansatz $\mathcal{E}_L = \rho^\alpha \mathcal{E}_0$ and $\mathcal{E}_R = \rho^{-\alpha} \mathcal{E}_0$:

From Constraint 1: $\rho^\alpha \cdot \rho^{-\alpha} = 1$ ✓ (satisfied for any $\alpha$)

From Constraint 2: $\alpha = \frac{1}{2}$

Therefore:
$$\mathcal{E}_L(\rho) = \sqrt{\rho}\,\mathcal{E}_0, \qquad \mathcal{E}_R(\rho) = \frac{1}{\sqrt{\rho}}\,\mathcal{E}_0$$

### The Geometric Factor

The spectral density ratio is:
$$\gamma(\rho) := \frac{\varrho_R(\rho)}{\varrho_L(\rho)} = \frac{\text{tr}(\mathcal{E}_R)}{\text{tr}(\mathcal{E}_L)} = \frac{1/\sqrt{\rho}}{\sqrt{\rho}} = \frac{1}{\rho}$$

$$\boxed{\gamma(\rho) = \frac{1}{\rho}}$$

---

## Uniqueness Theorem

**Theorem.** *Let $\mathcal{E}_L(\rho) = \rho^\alpha \mathcal{E}_0$ and $\mathcal{E}_R(\rho) = \rho^{-\beta} \mathcal{E}_0$. Under:*
1. *T-invariance: $\mathcal{E}_L \cdot \mathcal{E}_R = \mathcal{E}_0^2$*
2. *Dimensional homogeneity*
3. *Derivation from $\sigma(D) \sim \sqrt{\rho}\,M(x)$*

*the unique solution is $\alpha = \beta = 1/2$, giving $\gamma(\rho) = 1/\rho$.*

**Proof.**

From (1): $\rho^\alpha \cdot \rho^{-\beta} = 1$ for all $\rho$, hence $\alpha = \beta$.

From (3): $\sigma(D)$ has $\sqrt{\rho}$, so $\mathcal{E}$ (from symbol of $D$, not $D^2$) scales as $\rho^{1/2}$.

Therefore $\alpha = 1/2$ uniquely. ∎

### Why Not D²?

If one used $D^2$ (whose $\mathcal{E}$ scales as $\rho$), one would get:
- $\alpha = 1$
- $\gamma(\rho) = 1/\rho^2$
- $\sqrt{\gamma(\rho_c)} = 3/2$ at $\rho_c = 2/3$
- $w_R/w_L = 2 \times 3/2 = 3$
- $\sin^2\theta_W \approx 0.288$ ✗

Only $\alpha = 1/2$ (from $D$) gives the correct GUT value.

---

## Complete Derivation Chain

### Step 1: Algebraic Input
$$\frac{\text{Tr}(Y^2)_R}{\text{Tr}(Y^2)_L} = 4 \quad \text{[from SM algebra]}$$

### Step 2: Critical Point
$$\rho_c = \frac{2}{3} \quad \text{[from } V_{\text{eff}}(\rho) \text{ phase transition]}$$

### Step 3: Geometric Factor
$$\gamma(\rho_c) = \frac{1}{\rho_c} = \frac{3}{2} \quad \text{[from } \sigma(D) \sim \sqrt{\rho}\,M]$$

### Step 4: Weight Ratio
$$\frac{w_R}{w_L} = \sqrt{\frac{\text{Tr}(Y^2)_R}{\text{Tr}(Y^2)_L}} \times \sqrt{\gamma(\rho_c)} = \sqrt{4} \times \sqrt{\frac{3}{2}} = 2 \times \sqrt{\frac{3}{2}} = \sqrt{6}$$

### Step 5: Effective Traces

With $w_L = 1$ (convention) and $w_R^2 = 6$:
$$\text{Tr}(Y^2)_{\text{eff}} = \text{Tr}(Y^2)_L \cdot 1 + \text{Tr}(Y^2)_R \cdot 6 = 4 + 96 = 100$$
$$\text{Tr}(T^aT^a)_{\text{eff}} = 36 \cdot 1 = 36 \quad \text{[SU(2) acts only on L]}$$

### Step 6: Spectral Coefficient Ratio

With GUT normalization $k_1 = 5/3$:
$$\frac{C_{U(1)}}{C_{SU(2)}} = \frac{\text{Tr}(Y^2)_{\text{eff}}}{k_1 \times \text{Tr}(T^aT^a)_{\text{eff}}} = \frac{100}{(5/3) \times 36} = \frac{100}{60} = \frac{5}{3}$$

### Step 7: Weinberg Angle

$$\sin^2\theta_W = \frac{1}{1 + C_{U(1)}/C_{SU(2)}} = \frac{1}{1 + 5/3} = \frac{1}{8/3} = \frac{3}{8}$$

$$\boxed{\sin^2\theta_W(\Lambda_{\text{GUT}}) = \frac{3}{8} = 0.375}$$

---

## Numerical Verification

### Independence of N_norm

| $N_{\text{norm}}$ | $C_{U(1)}$ | $C_{SU(2)}$ | Ratio | $\sin^2\theta_W$ |
|-------------------|------------|-------------|-------|------------------|
| 1 | 60.0 | 36.0 | 5/3 | 0.375 |
| 100 | 0.60 | 0.36 | 5/3 | 0.375 |
| 192 | 0.3125 | 0.1875 | 5/3 | 0.375 |
| 10000 | 0.006 | 0.0036 | 5/3 | 0.375 |

The result is **independent of normalization**.

### Invariance Under Counting Conventions

| Convention | $\text{Tr}(Y^2)_L$ | $\text{Tr}(Y^2)_R$ | $\sin^2\theta_W$ |
|------------|-------------------|-------------------|------------------|
| Particles only | 2 | 8 | 0.375 |
| + Antiparticles | 4 | 16 | 0.375 |
| + J-duplication | 8 | 32 | 0.375 |

The result is **invariant under all counting conventions**.

### Alternative γ(ρ) Forms

| $\gamma(\rho)$ | Value at $\rho_c=2/3$ | $w_R/w_L$ | $\sin^2\theta_W$ |
|----------------|----------------------|-----------|------------------|
| **1/ρ** | 1.50 | 2.449 | **0.375** ✓ |
| 1/ρ² | 2.25 | 3.00 | 0.288 |
| 1-ρ | 0.33 | 1.15 | 0.68 |
| 1 | 1.00 | 2.00 | 0.47 |

Only $\gamma(\rho) = 1/\rho$ produces the GUT value.

---

## Physical Interpretation

### Why √ρ in the Dirac Symbol?

The factor $\sqrt{\rho}$ represents the **amplitude** of the spectral condensate, not its density. This is analogous to:

1. **BCS Superconductivity**: Gap $\Delta \propto \sqrt{n_{\text{pairs}}}$
2. **Spontaneous Symmetry Breaking**: VEV $v \propto \sqrt{\mu^2/\lambda}$
3. **Coherent States**: Amplitude $\alpha \propto \sqrt{n}$

In all cases, the relevant coupling is to the **amplitude** (square root of occupation), not the occupation itself.

### The Critical Point ρ_c = 2/3

The value $\rho_c = 2/3$ emerges from the effective potential $V_{\text{eff}}(\rho)$ of the spectral vacuum. This is the point of phase transition where:
- Below $\rho_c$: Spectral (pre-geometric) phase
- Above $\rho_c$: Condensed (geometric) phase

The factor $\sqrt{3/2} = \sqrt{1/\rho_c}$ is a **prediction** of the theory, not an input.

### Connection to GUT Unification

The value $\sin^2\theta_W = 3/8$ is precisely the prediction of grand unified theories (SU(5), SO(10), etc.) at the unification scale. TSQVT derives this from:
- Pure geometry of the spectral manifold
- No adjustable parameters
- No assumption of a specific GUT group

This suggests that GUT unification is a **consequence** of spectral geometry, not an independent postulate.

---

## Summary

| Element | Origin | Value |
|---------|--------|-------|
| $\text{Tr}(Y^2)_R/\text{Tr}(Y^2)_L$ | SM algebra $\mathcal{A}_F$ | 4 |
| $\rho_c$ | $V_{\text{eff}}$ phase transition | 2/3 |
| $\alpha$ | $\sigma(D) \sim \sqrt{\rho}\,M$ | 1/2 |
| $\gamma(\rho)$ | T-invariance + symbol | 1/ρ |
| $w_R/w_L$ | Combined | √6 |
| $C_{U(1)}/C_{SU(2)}$ | Spectral matching | 5/3 |
| **$\sin^2\theta_W$** | **Result** | **3/8** |

**All elements are derived. No free parameters.**

---

*Document version: 1.0*  
*Last updated: December 2024*  
*Author: Kerym Makraini, UNED Madrid*

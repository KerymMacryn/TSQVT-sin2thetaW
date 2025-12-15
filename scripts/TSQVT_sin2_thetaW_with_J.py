#!/usr/bin/env python3

"""
TSQVT_sin2_thetaW_with_J.py

Fixed: avoids double counting SU(2), configurable normalizations,
and calculation of effective traces with twistorial weights.

Affiliation: UNED, Madrid, Spain
Repository: https://github.com/KerymMacryn/TSQVT-sin2thetaW
Madrid: December 14, 2025
Usage:
    python scripts/TSQVT_sin2_thetaW_with_J.py
Author: Kerym Macryn
"""

import math
import numpy as np

# -------------------------
# Configuration
# -------------------------
N_GEN = 3                      # number of generations
INCLUDE_ANTIPARTICLES = True   # duplicate each particle state as antiparticle
INCLUDE_J_DUPLICATION = True   # duplicate whole finite space to represent J action
K1 = 5.0 / 3.0                 # GUT hypercharge normalization (used later if needed)
RHO_C = 2.0 / 3.0              # critical spectral density (for twistorial weight)
USE_TWISTORIAL_WEIGHTS = True  # whether to compute effective traces with w_R/w_L

# -------------------------
# Basis construction utilities
# -------------------------
def build_particle_basis(n_gen):
    """
    Build lists of left doublets and right singlets for n_gen generations.
    Returns:
      doublets: list of tuples ('lep', gen) and ('quark', gen, color)
      singlets: list of tuples ('eR', gen), ('uR', gen, color), ('dR', gen, color)
    """
    doublets = []
    singlets = []
    for g in range(n_gen):
        # leptonic doublet (nu_L, e_L)
        doublets.append(('lep', g))
        # quark doublets (u_L, d_L) with 3 colors
        for c in range(3):
            doublets.append(('quark', g, c))
        # right-handed singlets
        singlets.append(('eR', g))
        for c in range(3):
            singlets.append(('uR', g, c))
        for c in range(3):
            singlets.append(('dR', g, c))
    return doublets, singlets

def duplicate_for_antiparticles(doublets, singlets):
    """
    Return new lists where each state is duplicated as an antiparticle.
    For antiparticles hypercharge sign flips, but Tr(Y^2) is invariant under sign.
    We keep labels distinct for counting.
    """
    doublets_ap = doublets.copy()
    singlets_ap = singlets.copy()
    # append antiparticle-labeled entries
    doublets_ap += [('anti_' + d[0],) + d[1:] for d in doublets]
    singlets_ap += [('anti_' + s[0],) + s[1:] for s in singlets]
    return doublets_ap, singlets_ap

def duplicate_for_J(doublets, singlets):
    """
    Duplicate the entire finite space to represent the action of J (real structure).
    This simply concatenates a J-copy with distinct labels.
    """
    doublets_J = doublets + [('J_' + d[0],) + d[1:] for d in doublets]
    singlets_J = singlets + [('J_' + s[0],) + s[1:] for s in singlets]
    return doublets_J, singlets_J

# -------------------------
# Algebraic traces
# -------------------------
def algebraic_traces_Y2(doublets, singlets):
    """
    Compute Tr(Y^2)_L and Tr(Y^2)_R from lists.
    Each doublet contributes two components (ν and e) per doublet entry.
    Each quark doublet entry already encodes a single color; we counted colors in the list.
    """
    # SM hypercharges
    Y = {
        'lep': -0.5,
        'quark': 1.0/6.0,
        'eR': -1.0,
        'uR': 2.0/3.0,
        'dR': -1.0/3.0
    }
    TrY2_L = 0.0
    TrY2_R = 0.0
    # Left: each doublet contributes two components
    for d in doublets:
        tag = d[0]
        base = tag.replace('anti_', '').replace('J_', '')
        if base == 'lep':
            TrY2_L += 2 * (Y['lep'] ** 2)
        elif base == 'quark':
            TrY2_L += 2 * (Y['quark'] ** 2)
        else:
            # unknown tag: ignore
            pass
    # Right: each singlet contributes one component
    for s in singlets:
        tag = s[0]
        base = tag.replace('anti_', '').replace('J_', '')
        if base in ('eR', 'uR', 'dR'):
            TrY2_R += (Y[base] ** 2)
        else:
            # ignore unexpected labels
            pass
    return TrY2_L, TrY2_R

# -------------------------
# SU(2) trace (no double counting)
# -------------------------
def Tr_T2_total(doublets):
    """
    Compute Tr(T^a T^a) total by counting each SU(2) doublet exactly once.
    For fundamental 2-dim representation T^a = sigma^a/2, the sum Tr(T^a T^a) = 3/2 per doublet.
    """
    # number of doublets (exclude antiparticle or J prefixes when counting unique doublets)
    # We count entries whose base tag is 'lep' or 'quark' and treat each occurrence as a distinct doublet.
    n_doublets = 0
    for d in doublets:
        base = d[0].replace('anti_', '').replace('J_', '')
        if base in ('lep', 'quark'):
            n_doublets += 1
    # single doublet contribution
    single_doublet_Tr = 1.5  # 3/2
    total = n_doublets * single_doublet_Tr
    return float(total), n_doublets, float(single_doublet_Tr)

# -------------------------
# Twistorial weight and effective traces
# -------------------------
def twistorial_weight_and_effective_traces(TrY2_L, TrY2_R, TrT2_total, rho_c=RHO_C):
    """
    Compute w_R/w_L from algebraic traces and rho_c, then effective traces.
    w_L is set to 1 by convention.
    """
    if TrY2_L == 0:
        raise ValueError("TrY2_L is zero, cannot compute ratio.")
    Y2_ratio = TrY2_R / TrY2_L
    wR_over_wL = math.sqrt(Y2_ratio) * math.sqrt(1.0 / rho_c)
    wL = 1.0
    wR = wR_over_wL
    TrY2_eff = TrY2_L * (wL ** 2) + TrY2_R * (wR ** 2)
    TrT2_eff = TrT2_total * (wL ** 2)  # SU(2) acts only on L
    return {
        'Y2_ratio': Y2_ratio,
        'wR_over_wL': wR_over_wL,
        'TrY2_eff': TrY2_eff,
        'TrT2_eff': TrT2_eff
    }

# -------------------------
# Main driver
# -------------------------
def run_all(n_gen=N_GEN,
            include_antip=INCLUDE_ANTIPARTICLES,
            include_J=INCLUDE_J_DUPLICATION,
            use_weights=USE_TWISTORIAL_WEIGHTS,
            rho_c=RHO_C,
            N_norm=192.0,
            k1=K1):
    # base particles
    doublets, singlets = build_particle_basis(n_gen)
    # optionally include antiparticles
    if include_antip:
        doublets, singlets = duplicate_for_antiparticles(doublets, singlets)
    # optionally include J duplication
    if include_J:
        doublets, singlets = duplicate_for_J(doublets, singlets)

    dimH = len(doublets) + len(singlets)
    TrY2_L, TrY2_R = algebraic_traces_Y2(doublets, singlets)
    TrT2_total, n_doublets, single_doublet_Tr = Tr_T2_total(doublets)

    # twistorial effective traces
    if use_weights:
        tw = twistorial_weight_and_effective_traces(TrY2_L, TrY2_R, TrT2_total, rho_c=rho_c)
        TrY2_eff = tw['TrY2_eff']
        TrT2_eff = tw['TrT2_eff']
        wR_over_wL = tw['wR_over_wL']
    else:
        TrY2_eff = TrY2_L + TrY2_R
        TrT2_eff = TrT2_total
        wR_over_wL = 1.0

    # spectral coefficients and sin^2 theta_W
    C_U1 = (TrY2_eff / float(N_norm)) / float(k1)
    C_SU2 = (TrT2_eff / float(N_norm))
    ratio = (C_U1 / C_SU2) if C_SU2 != 0 else float('inf')
    sin2 = 1.0 / (1.0 + ratio)

    # prepare results
    results = {
        'dimH': dimH,
        'n_doublets': n_doublets,
        'single_doublet_TrT2': single_doublet_Tr,
        'TrY2_L': TrY2_L,
        'TrY2_R': TrY2_R,
        'Y2_ratio_algebra': (TrY2_R / TrY2_L) if TrY2_L != 0 else None,
        'wR_over_wL': wR_over_wL,
        'TrY2_eff': TrY2_eff,
        'TrT2_eff': TrT2_eff,
        'C_U1': C_U1,
        'C_SU2': C_SU2,
        'ratio_CU1_over_CSU2': ratio,
        'sin2_thetaW': sin2
    }
    return results

# -------------------------
# Run and print scenarios
# -------------------------
if __name__ == "__main__":
    scenarios = [
        {'include_antip': False, 'include_J': False, 'label': 'particles only'},
        {'include_antip': True,  'include_J': False, 'label': 'particles + antiparticles'},
        {'include_antip': True,  'include_J': True,  'label': 'particles + antiparticles + J-duplication'}
    ]

    for sc in scenarios:
        res = run_all(n_gen=N_GEN,
                      include_antip=sc['include_antip'],
                      include_J=sc['include_J'],
                      use_weights=USE_TWISTORIAL_WEIGHTS,
                      rho_c=RHO_C,
                      N_norm=192.0,
                      k1=K1)
        print("\n" + "="*72)
        print(f"Scenario: {sc['label']}")
        print("="*72)
        print(f"dim(H_F) = {res['dimH']}")
        print(f"Number of SU(2) doublets = {res['n_doublets']}")
        print(f"Tr(T^a T^a) per doublet = {res['single_doublet_TrT2']}")
        print(f"Tr(T^a T^a) total = {res['TrT2_eff']:.6f}")
        print(f"Tr(Y^2)_L = {res['TrY2_L']:.6f}")
        print(f"Tr(Y^2)_R = {res['TrY2_R']:.6f}")
        print(f"Algebraic Y2_R/Y2_L = {res['Y2_ratio_algebra']:.6f}")
        print(f"w_R/w_L = {res['wR_over_wL']:.6f}")
        print(f"Tr(Y^2)_eff = {res['TrY2_eff']:.6f}")
        print(f"C_U1 = {res['C_U1']:.6f}")
        print(f"C_SU2 = {res['C_SU2']:.6f}")
        print(f"C_U1 / C_SU2 = {res['ratio_CU1_over_CSU2']:.6f}")
        print(f"sin^2(theta_W) = {res['sin2_thetaW']:.6f}")

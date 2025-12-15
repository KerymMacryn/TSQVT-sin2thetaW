#!/usr/bin/env python3
"""
TSQVT_sin2_thetaW_fixed.py

Fixed: avoids double counting SU(2), configurable normalizations,
and calculation of effective traces with twistorial weights.

Affiliation: UNED, Madrid, Spain
Repository: https://github.com/KerymMacryn/TSQVT-sin2thetaW
Madrid: December 14, 2025
Usage:
    python scripts/TSQVT_sin2_thetaW_fixed.py
Author: Kerym Macryn
"""

import numpy as np
from math import sqrt

# -------------------------
# Configuración
# -------------------------
N_gen = 3                      # number of generations
include_antiparticles = False  # si True duplica estados para antipartículas
k1 = 5.0 / 3.0                 # normalización GUT para U(1)_Y
# Normalización espectral por defecto (ajustable)
N_norm_default = 192.0         # usa 192 para reproducir la convención previa; cambiar si se desea

# Twistorial weights
use_twistorial_weights = True
rho_c = 2.0 / 3.0              # punto crítico
# Si quieres forzar un ratio efectivo Y_R/Y_L distinto al de las trazas, ponlo aquí (None para derivarlo)
Y_ratio_eff_override = None    # ejemplo: 0.632 (√(2/5)) o None

# -------------------------
# Construcción de la base (lista de dobletes y singletes)
# -------------------------
def build_basis_list(n_gen=3, include_antip=False):
    """
    Returns structured lists of states to build operators.
    We do not build explicit canonical vectors; we work with indices and blocks.
    """
    doublets = []   # cada elemento: ('lep', gen) o ('quark', gen, color)
    singlets = []   # cada elemento: ('eR', gen) or ('uR', gen, color) etc.
    for g in range(n_gen):
        # leptonic doublet (nu_L, e_L)
        doublets.append(('lep', g))
        # quark doublets: 3 colores
        for c in range(3):
            doublets.append(('quark', g, c))
        # singlets
        singlets.append(('eR', g))
        for c in range(3):
            singlets.append(('uR', g, c))
        for c in range(3):
            singlets.append(('dR', g, c))
    # si se incluyen antipartículas, duplicar con etiqueta 'anti' (no cambia cargas absolutas)
    if include_antip:
        # para trazas se suman igual; aquí devolvemos la misma estructura duplicada
        doublets = doublets + [('anti_'+d[0],) + d[1:] for d in doublets]
        singlets = singlets + [('anti_'+s[0],) + s[1:] for s in singlets]
    return doublets, singlets

# -------------------------
# Trazas algebraicas Tr(Y^2) por chirality
# -------------------------
def algebraic_traces_Y2(doublets, singlets):
    """
    Calculates Tr(Y^2) by separating L and R contributions using multiplicities.
    """
    # Valores de Y por tipo (convención SM)
    Y_values = {
        'lep': -0.5,      # doublet leptónico (ν,e)_L
        'quark': 1.0/6.0, # doublet quark (u,d)_L
        'eR': -1.0,
        'uR': 2.0/3.0,
        'dR': -1.0/3.0
    }
    TrY2_L = 0.0
    TrY2_R = 0.0
    # cada doublet aporta dos componentes (ν and e) por color multiplicity implícita
    for d in doublets:
        tag = d[0]
        if tag == 'lep':
            # leptonic doublet: 2 components, color 1
            TrY2_L += 2 * (Y_values['lep']**2)
        elif tag == 'quark':
            # quark doublet: 2 components × color multiplicity 1 (color encoded in d[2])
            TrY2_L += 2 * (Y_values['quark']**2)
        else:
            # antiparticle labels if present: treat same as original doublet
            base = tag.replace('anti_','')
            if base == 'lep':
                TrY2_L += 2 * (Y_values['lep']**2)
            elif base == 'quark':
                TrY2_L += 2 * (Y_values['quark']**2)
    for s in singlets:
        tag = s[0]
        if tag.startswith('anti_'):
            base = tag.replace('anti_','')
        else:
            base = tag
        if base in ('eR','uR','dR'):
            TrY2_R += (Y_values[base]**2)
        else:
            # ignore unexpected
            pass
    # multiply by number of generations encoded in lists (already included)
    return TrY2_L, TrY2_R

# -------------------------
# Construction of SU(2) generators and trace Tr(T^a T^a)
# -------------------------
def compute_TrT2(doublets):
    """
    Constructs 2x2 blocks per doublet exactly once and sums Tr(T^a T^a).
    For each doublet, the fundamental representation with T^a = sigma^a/2 is used.

    """
    # Para la representación fundamental 2x2:
    sigma_x = np.array([[0.0, 1.0],[1.0, 0.0]])
    sigma_y = np.array([[0.0, -1.0j],[1.0j, 0.0]])
    sigma_z = np.array([[1.0, 0.0],[0.0, -1.0]])
    T1 = sigma_x / 2.0
    T2 = sigma_y / 2.0
    T3 = sigma_z / 2.0
    # Tr(T^a T^a) for fundamental 2-dim: Tr(T1^2 + T2^2 + T3^2) = 3 * Tr((sigma/2)^2)
    # compute single-doublet contribution
    single_doublet_Tr = np.trace(T1.dot(T1) + T2.dot(T2) + T3.dot(T3)).real
    # number of doublets (each doublet contributes once)
    n_doublets = len([d for d in doublets if not str(d[0]).startswith('anti_')])
    # if antiparticles included, doublets list already duplicated
    total_TrT2 = n_doublets * single_doublet_Tr
    return float(total_TrT2), n_doublets, float(single_doublet_Tr)

# -------------------------
# Final calculation with twistorial weights
# -------------------------
def compute_results(n_gen=N_gen,
                    include_antip=include_antiparticles,
                    use_weights=use_twistorial_weights,
                    rho_c_val=rho_c,
                    Y_ratio_eff_override_val=Y_ratio_eff_override,
                    N_norm=N_norm_default,
                    k1_val=k1):
    doublets, singlets = build_basis_list(n_gen, include_antip)
    TrY2_L, TrY2_R = algebraic_traces_Y2(doublets, singlets)
    TrT2_total, n_doublets, single_doublet_Tr = compute_TrT2(doublets)

    # calcular ratio Y2_R/Y2_L algebraica
    Y2_ratio_algebra = (TrY2_R / TrY2_L) if TrY2_L != 0 else float('inf')

    # determinar w_R/w_L
    if use_weights:
        if Y_ratio_eff_override_val is not None:
            Y_ratio_eff = float(Y_ratio_eff_override_val)
        else:
            # derivar desde trazas algebraicas: sqrt(Y2_R/Y2_L)
            Y_ratio_eff = sqrt(Y2_ratio_algebra)
        wR_over_wL = Y_ratio_eff * sqrt(1.0 / rho_c_val)
    else:
        wR_over_wL = 1.0

    # calcular trazas efectivas
    # asumimos w_L = 1
    wL = 1.0
    wR = wR_over_wL * wL
    TrY2_eff = TrY2_L * (wL**2) + TrY2_R * (wR**2)
    # SU(2) actúa solo en L, por tanto TrT2_eff = TrT2_total * wL^2
    TrT2_eff = TrT2_total * (wL**2)

    # normalizaciones y coeficientes espectrales
    C_U1 = (TrY2_eff / float(N_norm)) / float(k1_val)
    C_SU2 = (TrT2_eff / float(N_norm))

    ratio_C = (C_U1 / C_SU2) if C_SU2 != 0 else float('inf')
    sin2_thetaW = 1.0 / (1.0 + ratio_C)

    return {
        'dimH_effective': len(doublets) + len(singlets),
        'n_doublets': n_doublets,
        'single_doublet_TrT2': single_doublet_Tr,
        'TrY2_L': TrY2_L,
        'TrY2_R': TrY2_R,
        'Y2_ratio_algebra': Y2_ratio_algebra,
        'wR_over_wL': wR_over_wL,
        'TrY2_eff': TrY2_eff,
        'TrT2_eff': TrT2_eff,
        'C_U1': C_U1,
        'C_SU2': C_SU2,
        'ratio_CU1_over_CSU2': ratio_C,
        'sin2_thetaW': sin2_thetaW
    }

# -------------------------
# Ejecución
# -------------------------
if __name__ == "__main__":
    res = compute_results()
    print("Dim(H_F) effective =", res['dimH_effective'])
    print("Number of SU(2) doublets =", res['n_doublets'])
    print("Single doublet Tr(T^a T^a) =", res['single_doublet_TrT2'])
    print("Tr(Y^2)_L =", res['TrY2_L'])
    print("Tr(Y^2)_R =", res['TrY2_R'])
    print("Algebraic Y2_R/Y2_L =", res['Y2_ratio_algebra'])
    print("w_R/w_L =", res['wR_over_wL'])
    print("Tr(Y^2)_eff =", res['TrY2_eff'])
    print("Tr(T^a T^a)_eff =", res['TrT2_eff'])
    print("C_U1 =", res['C_U1'])
    print("C_SU2 =", res['C_SU2'])
    print("C_U1 / C_SU2 =", res['ratio_CU1_over_CSU2'])
    print("sin^2(theta_W) =", res['sin2_thetaW'])

"""
Finite Geometry (corregido)
Autor: adaptado para el notebook TSQVT
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, field


@dataclass
class SMAlgebra:
    """Standard Model algebra A_F = C ⊕ H ⊕ M_3(C)."""
    n_generations: int = 3

    generators: Dict[str, List[np.ndarray]] = field(default_factory=dict, init=False)
    dimension: int = field(init=False)

    def __post_init__(self):
        self.dimension = 1 + 4 + 9  # C + H + M3(C)
        self._construct_generators()

    def _construct_generators(self):
        # U(1)
        self.generators['U1'] = [np.array([[1.0]])]

        # SU(2) (Pauli/2)
        sigma_1 = np.array([[0, 1], [1, 0]], dtype=complex) / 2
        sigma_2 = np.array([[0, -1j], [1j, 0]], dtype=complex) / 2
        sigma_3 = np.array([[1, 0], [0, -1]], dtype=complex) / 2
        self.generators['SU2'] = [sigma_1, sigma_2, sigma_3]

        # SU(3) (Gell-Mann/2)
        lambda_1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=complex) / 2
        lambda_2 = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]], dtype=complex) / 2
        lambda_3 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]], dtype=complex) / 2
        lambda_4 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], dtype=complex) / 2
        lambda_5 = np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]], dtype=complex) / 2
        lambda_6 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=complex) / 2
        lambda_7 = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]], dtype=complex) / 2
        lambda_8 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -2]], dtype=complex) / (2 * np.sqrt(3))
        self.generators['SU3'] = [lambda_1, lambda_2, lambda_3, lambda_4,
                                   lambda_5, lambda_6, lambda_7, lambda_8]

    def casimir(self, group: str) -> float:
        casimirs = {
            'U1': 0.0,
            'SU2': 3/4,
            'SU3': 4/3,
        }
        return casimirs.get(group, 0.0)

    def dynkin_index(self, group: str, rep: str = 'fundamental', hypercharge: float = 0.0) -> float:
        # For non-abelian groups use standard values; for U(1) use Y^2 normalization
        if group == 'U1':
            # normalization: T(U1) = Y^2 (convention-dependent). We return sum of Y^2 for a single state.
            return float(hypercharge ** 2)
        indices = {
            ('SU2', 'fundamental'): 1/2,
            ('SU2', 'adjoint'): 2,
            ('SU3', 'fundamental'): 1/2,
            ('SU3', 'adjoint'): 3,
        }
        return indices.get((group, rep), 0.0)

    def hypercharge(self, particle: str) -> float:
        charges = {
            'eR': -1.0,
            'L': -0.5,
            'nuR': 0.0,
            'uR': 2/3,
            'dR': -1/3,
            'Q': 1/6,
            'H': 0.5,
        }
        return charges.get(particle, 0.0)


@dataclass
class FiniteGeometry:
    """Finite noncommutative geometry for the Standard Model."""
    n_generations: int = 3

    algebra: SMAlgebra = field(init=False)
    hilbert_dim: int = field(init=False)
    J: np.ndarray = field(init=False, repr=False)
    gamma: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.algebra = SMAlgebra(self.n_generations)
        # Use 32 states per generation as in tu ejemplo (16 particles + 16 antiparticles)
        self.hilbert_dim = 32 * self.n_generations
        self._construct_real_structure()
        self._construct_grading()

    # Propiedades convenientes que el notebook esperaba
    @property
    def j_squared(self) -> bool:
        """Boolean: J^2 == 1 (aprox)."""
        return np.allclose(self.J @ self.J, np.eye(self.hilbert_dim))

    @property
    def grading_squared(self) -> bool:
        """Boolean: γ^2 == 1 (aprox)."""
        return np.allclose(self.gamma @ self.gamma, np.eye(self.hilbert_dim))

    def _construct_real_structure(self):
        N = self.hilbert_dim
        self.J = np.zeros((N, N), dtype=complex)
        # antidiagonal ones (swap particle/antiparticle)
        for i in range(N // 2):
            self.J[i, N - 1 - i] = 1.0
            self.J[N - 1 - i, i] = 1.0

    def _construct_grading(self):
        N = self.hilbert_dim
        # first half left-handed (+1), second half right-handed (-1)
        self.gamma = np.diag([1.0] * (N // 2) + [-1.0] * (N // 2))

    def verify_ko_dimension(self, D: np.ndarray, tol: float = 1e-10) -> Dict[str, bool]:
        results = {}
        J_sq = self.J @ self.J
        results['J_squared'] = np.allclose(J_sq, np.eye(self.hilbert_dim), atol=tol)
        results['JD_commutes'] = np.allclose(self.J @ D, D @ self.J, atol=tol)
        results['J_gamma_anticommutes'] = np.allclose(self.J @ self.gamma, - self.gamma @ self.J, atol=tol)
        results['gamma_D_anticommutes'] = np.allclose(self.gamma @ D, - D @ self.gamma, atol=tol)
        return results

    def particle_content(self) -> Dict[str, Dict]:
        return {
            'leptons': {
                'eR': {'SU3': '1', 'SU2': '1', 'Y': -1.0},
                'L': {'SU3': '1', 'SU2': '2', 'Y': -0.5},
                'nuR': {'SU3': '1', 'SU2': '1', 'Y': 0.0},
            },
            'quarks': {
                'uR': {'SU3': '3', 'SU2': '1', 'Y': 2/3},
                'dR': {'SU3': '3', 'SU2': '1', 'Y': -1/3},
                'Q': {'SU3': '3', 'SU2': '2', 'Y': 1/6},
            },
        }

    def representation_projector(self, particle: str) -> np.ndarray:
        """Projector onto a simplified subspace for `particle`."""
        N = self.hilbert_dim
        P = np.zeros((N, N), dtype=complex)

        # Distribución simple y reproducible de índices por partícula
        # Asignamos bloques contiguos por tipo y generación para evitar desbordes.
        # Tamaños relativos (arbitrarios pero consistentes): eR:1, L:2, nuR:1, uR:3, dR:3, Q:6  (por generación)
        sizes_per_gen = {
            'eR': 1,
            'L': 2,
            'nuR': 1,
            'uR': 3,
            'dR': 3,
            'Q': 6,
        }
        # calcular offsets
        order = ['eR', 'L', 'nuR', 'uR', 'dR', 'Q']
        block_size = sum(sizes_per_gen.values()) * self.n_generations
        # Si block_size excede N, escalamos proporcionalmente
        if block_size > N:
            scale = N / block_size
            # asignar al menos 1 por entrada
            for k in sizes_per_gen:
                sizes_per_gen[k] = max(1, int(round(sizes_per_gen[k] * scale)))

        # construir índices
        idx = 0
        particle_indices = {}
        for p in order:
            count = sizes_per_gen[p] * self.n_generations
            rng = range(idx, min(idx + count, N))
            particle_indices[p] = rng
            idx += count
            if idx >= N:
                break

        if particle in particle_indices:
            for i in particle_indices[particle]:
                P[i, i] = 1.0
        return P

    def gauge_projector(self, gauge_group: str) -> np.ndarray:
        N = self.hilbert_dim
        P = np.zeros((N, N), dtype=complex)
        particles = self.particle_content()
        for category in ['leptons', 'quarks']:
            for particle, qnumbers in particles[category].items():
                if gauge_group == 'U1':
                    if abs(qnumbers['Y']) > 0.0:
                        P += self.representation_projector(particle)
                elif gauge_group == 'SU2':
                    if qnumbers['SU2'] == '2':
                        P += self.representation_projector(particle)
                elif gauge_group == 'SU3':
                    if qnumbers['SU3'] == '3':
                        P += self.representation_projector(particle)
        # Asegurar que P sea hermítica y con entradas reales en diagonal
        P = (P + P.conj().T) / 2.0
        return P

    def compute_C4_coefficient(
        self,
        gauge_group: str,
        D_matrices: Dict[int, np.ndarray] = None
    ) -> float:
        """
        C_4^{(a)} = (1/12) Σ_states T_R^{(a)} * (contrib)
        Implementación simplificada: trazas sobre proyectores y factores de índice.
        """
        if D_matrices is None:
            D_matrices = {}

        P = self.gauge_projector(gauge_group)
        # Normalización: contar estados cargados (traza del proyector)
        charged_states = float(np.real(np.trace(P)))

        # Para U(1) sumamos Y^2 sobre partículas (por generación)
        if gauge_group == 'U1':
            particles = self.particle_content()
            sum_Y2 = 0.0
            for category in particles.values():
                for p, q in category.items():
                    Y = float(q['Y'])
                    # multiplicidad: n_generations
                    sum_Y2 += (Y ** 2) * self.n_generations
            # dynkin-like normalization: dividir por número de estados para evitar escala arbitraria
            C4 = (1.0 / 12.0) * sum_Y2
            return float(C4)

        # Para SU(2)/SU(3) usamos dynkin index por estado fundamental
        k = self.algebra.dynkin_index(gauge_group, 'fundamental')

        # Contribución geométrica simplificada: tr(P) + trazas con D0,D1 si existen
        D0 = D_matrices.get(0, np.eye(self.hilbert_dim))
        D1 = D_matrices.get(1, np.zeros((self.hilbert_dim, self.hilbert_dim)))
        # coeficientes heurísticos (como en tu versión original)
        alpha_1 = 1.0
        alpha_2 = -0.5

        # calcular T4 como tr[P (1 + α1 D0^2 + α2 D0 D1)]
        term = np.eye(self.hilbert_dim) + alpha_1 * (D0 @ D0) + alpha_2 * (D0 @ D1)
        T4 = np.trace(P @ term)
        C4 = k * float(np.real(T4)) / 12.0
        return float(C4)

    def unimodularity_residual(self, Q: np.ndarray) -> float:
        return abs(np.trace(Q))

    def __repr__(self) -> str:
        return f"FiniteGeometry(n_gen={self.n_generations}, dim_H={self.hilbert_dim})"


# Función auxiliar pública para compatibilidad con el notebook original
def compute_C4_coefficients(yukawa: dict, majorana: dict, n_generations: int = 3) -> Dict[str, float]:
    """
    Construye una FiniteGeometry y devuelve C4 para U1, SU2, SU3.
    Esta implementación usa D_matrices triviales por defecto; si se desea,
    se pueden pasar matrices D^{(i)} más realistas.
    """
    geom = FiniteGeometry(n_generations=n_generations)
    # D_matrices por defecto (identidad y ceros) — el usuario puede reemplazar
    D_matrices = {
        0: np.eye(geom.hilbert_dim),
        1: np.zeros((geom.hilbert_dim, geom.hilbert_dim)),
    }
    C4 = {}
    for g in ['U1', 'SU2', 'SU3']:
        C4[g] = geom.compute_C4_coefficient(g, D_matrices)
    return C4

from __future__ import annotations

from dataclasses import dataclass
from math import cos, exp, sin
from random import Random
from typing import Iterable, List, Tuple

Vector = List[float]
CVector = List[complex]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + exp(-x))


def matvec(x: Vector, w: List[Vector]) -> Vector:
    # x: [in], w: [in][out]
    out = [0.0 for _ in range(len(w[0]))]
    for i, xv in enumerate(x):
        row = w[i]
        for j in range(len(out)):
            out[j] += xv * row[j]
    return out


def unit_phasor(x: Vector) -> CVector:
    return [complex(cos(v), sin(v)) for v in x]


def real_to_complex(v: Vector) -> CVector:
    return [complex(x, 0.0) for x in v]


@dataclass
class CoreParams:
    d_model: int
    seed: int = 0

    def __post_init__(self) -> None:
        rng = Random(self.seed)
        s = (self.d_model ** -0.5)

        def rand_mat() -> List[Vector]:
            return [[rng.gauss(0.0, s) for _ in range(self.d_model)] for _ in range(self.d_model)]

        self.W_k = rand_mat()
        self.W_q = rand_mat()
        self.W_v = rand_mat()
        self.W_g = rand_mat()
        self.W_o = rand_mat()


class HolographicStateCore:
    def __init__(self, params: CoreParams):
        self.params = params
        self.state: CVector = [0j for _ in range(params.d_model)]

    def reset(self) -> None:
        self.state = [0j for _ in range(self.params.d_model)]

    def step(self, x_t: Vector) -> Tuple[Vector, Vector]:
        p = self.params
        k_t = unit_phasor(matvec(x_t, p.W_k))
        q_t = unit_phasor(matvec(x_t, p.W_q))
        v_t = matvec(x_t, p.W_v)
        g_t = [_sigmoid(v) for v in matvec(x_t, p.W_g)]

        r_t = [q_t[i].conjugate() * self.state[i] for i in range(p.d_model)]
        y_core = matvec([z.real for z in r_t], p.W_o)

        write = [k_t[i] * real_to_complex(v_t)[i] for i in range(p.d_model)]
        self.state = [g_t[i] * self.state[i] + write[i] for i in range(p.d_model)]
        return y_core, g_t


def compose_affine(pair2: Tuple[Vector, CVector], pair1: Tuple[Vector, CVector]) -> Tuple[Vector, CVector]:
    g2, b2 = pair2
    g1, b1 = pair1
    g = [g2[i] * g1[i] for i in range(len(g1))]
    b = [g2[i] * b1[i] + b2[i] for i in range(len(b1))]
    return g, b


def parallel_prefix_scan(g_list: Iterable[Vector], b_list: Iterable[CVector]) -> Tuple[List[Vector], List[CVector]]:
    gs = [g[:] for g in g_list]
    bs = [b[:] for b in b_list]
    n = len(gs)
    stride = 1
    while stride < n:
        prev_gs = [g[:] for g in gs]
        prev_bs = [b[:] for b in bs]
        for i in range(n - 1, -1, -1):
            j = i - stride
            if j >= 0:
                gs[i], bs[i] = compose_affine((prev_gs[i], prev_bs[i]), (prev_gs[j], prev_bs[j]))
        stride *= 2
    return gs, bs

from __future__ import annotations

from dataclasses import dataclass
from math import exp, tanh
from random import Random
from typing import Dict, List, Tuple

from .core import CoreParams, HolographicStateCore, matvec
from .tape import LosslessTape, constant_k_cross_attention, encode_payloads

Vector = List[float]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + exp(-x))


@dataclass
class FluxConfig:
    d_model: int = 64
    k_recall: int = 8
    tape_buckets: int = 2048
    tape_probes: int = 3
    seed: int = 0


class FluxBlock:
    def __init__(self, cfg: FluxConfig):
        self.cfg = cfg
        self.core = HolographicStateCore(CoreParams(d_model=cfg.d_model, seed=cfg.seed))
        self.tape = LosslessTape(d_addr=cfg.d_model, num_buckets=cfg.tape_buckets, probes=cfg.tape_probes)

        rng = Random(cfg.seed + 1)
        s = cfg.d_model ** -0.5

        def rand_mat() -> List[Vector]:
            return [[rng.gauss(0.0, s) for _ in range(cfg.d_model)] for _ in range(cfg.d_model)]

        self.W_addr = rand_mat()
        self.W_gate = rand_mat()

    def reset(self) -> None:
        self.core.reset()

    def _addr(self, x: Vector, y_core: Vector) -> Vector:
        return [tanh(v) for v in matvec([x[i] + y_core[i] for i in range(self.cfg.d_model)], self.W_addr)]

    def step(self, x_t: Vector, meta: dict | None = None) -> Tuple[Vector, Dict[str, float]]:
        y_core, _ = self.core.step(x_t)
        q = self._addr(x_t, y_core)

        recalled = self.tape.lookup(q, self.cfg.k_recall)
        payloads = [r.payload for r in recalled]
        mem = encode_payloads(payloads, self.cfg.d_model)
        y_tape = constant_k_cross_attention(x_t, mem)

        gate = [_sigmoid(v) for v in matvec(x_t, self.W_gate)]
        y = [x_t[i] + gate[i] * y_tape[i] + (1.0 - gate[i]) * y_core[i] for i in range(self.cfg.d_model)]

        self.tape.append(addr=q, payload=x_t, meta=meta or {})

        stats = {
            "gate_mean": sum(gate) / len(gate),
            "retrieved": float(len(recalled)),
            "tape_size": float(len(self.tape)),
        }
        return y, stats

    def run(self, x: List[Vector]) -> Tuple[List[Vector], Dict[str, float]]:
        ys: List[Vector] = []
        gate_means: List[float] = []
        retrieved_counts: List[float] = []
        for t, token in enumerate(x):
            y, s = self.step(token, meta={"t": t})
            ys.append(y)
            gate_means.append(s["gate_mean"])
            retrieved_counts.append(s["retrieved"])

        n = max(1, len(x))
        return ys, {
            "gate_mean": sum(gate_means) / n,
            "retrieved_mean": sum(retrieved_counts) / n,
            "tape_size": float(len(self.tape)),
        }

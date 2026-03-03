from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt
from typing import Dict, List, Sequence

Vector = List[float]


@dataclass
class TapeRecord:
    addr: Vector
    payload: Vector
    meta: dict


class LosslessTape:
    def __init__(self, d_addr: int, num_buckets: int = 1024, probes: int = 3):
        self.d_addr = d_addr
        self.num_buckets = num_buckets
        self.probes = probes
        self.records: List[TapeRecord] = []
        self.buckets: Dict[int, List[int]] = {i: [] for i in range(num_buckets)}

    def _bucket_ids(self, key: Vector) -> List[int]:
        ids = []
        for i in range(self.probes):
            h = int(abs(sum((i + 1) * key[j] * (j + 1) for j in range(self.d_addr))))
            ids.append(h % self.num_buckets)
        return ids

    def append(self, addr: Vector, payload: Vector, meta: dict | None = None) -> int:
        idx = len(self.records)
        self.records.append(TapeRecord(addr=addr[:], payload=payload[:], meta=meta or {}))
        for bid in self._bucket_ids(addr):
            self.buckets[bid].append(idx)
        return idx

    def lookup(self, query: Vector, k: int) -> List[TapeRecord]:
        cand_idx = set()
        for bid in self._bucket_ids(query):
            cand_idx.update(self.buckets[bid])
        if not cand_idx:
            return []

        def dot(a: Vector, b: Vector) -> float:
            return sum(ai * bi for ai, bi in zip(a, b))

        ranked = sorted(cand_idx, key=lambda i: dot(self.records[i].addr, query), reverse=True)
        return [self.records[i] for i in ranked[:k]]

    def __len__(self) -> int:
        return len(self.records)


def encode_payloads(payloads: Sequence[Vector], d_model: int) -> List[Vector]:
    out: List[Vector] = []
    for p in payloads:
        if len(p) >= d_model:
            out.append(p[:d_model])
        else:
            out.append(p[:] + [0.0 for _ in range(d_model - len(p))])
    return out


def constant_k_cross_attention(query: Vector, memory: List[Vector]) -> Vector:
    if not memory:
        return [0.0 for _ in query]

    def dot(a: Vector, b: Vector) -> float:
        return sum(ai * bi for ai, bi in zip(a, b))

    scale = sqrt(max(1, len(query)))
    scores = [dot(m, query) / scale for m in memory]
    mmax = max(scores)
    weights = [exp(s - mmax) for s in scores]
    denom = sum(weights) or 1e-9
    weights = [w / denom for w in weights]

    out = [0.0 for _ in query]
    for wi, mem in zip(weights, memory):
        for j in range(len(out)):
            out[j] += wi * mem[j]
    return out

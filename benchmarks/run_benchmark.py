from __future__ import annotations

import argparse
import time
from random import Random

from flux import FluxBlock, FluxConfig


def make_data(seq_len: int, d_model: int, seed: int) -> list[list[float]]:
    rng = Random(seed)
    return [[rng.uniform(-1.0, 1.0) for _ in range(d_model)] for _ in range(seq_len)]


def benchmark(seq_len: int, d_model: int, k: int, seed: int = 0) -> None:
    x = make_data(seq_len, d_model, seed)
    block = FluxBlock(FluxConfig(d_model=d_model, k_recall=k, seed=seed))

    t0 = time.perf_counter()
    _, stats = block.run(x)
    t1 = time.perf_counter()

    total = t1 - t0
    print(f"seq_len={seq_len} d_model={d_model} k={k}")
    print(f"elapsed_s={total:.4f}")
    print(f"tokens_per_s={seq_len / total:.2f}")
    print(f"stats={stats}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--k", type=int, default=8)
    args = ap.parse_args()
    benchmark(seq_len=args.seq_len, d_model=args.d_model, k=args.k)

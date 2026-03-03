from __future__ import annotations

from random import Random

from flux import FluxBlock, FluxConfig, parallel_prefix_scan
from flux.core import compose_affine


def close_vec(a, b, tol=1e-8):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert abs(x - y) <= tol


def test_flux_block_shapes_and_tape_growth() -> None:
    cfg = FluxConfig(d_model=32, k_recall=4, seed=7)
    model = FluxBlock(cfg)

    rng = Random(1)
    x = [[rng.uniform(-1.0, 1.0) for _ in range(cfg.d_model)] for _ in range(128)]
    y, stats = model.run(x)

    assert len(y) == len(x)
    assert len(y[0]) == cfg.d_model
    assert stats["tape_size"] == 128
    assert 0.0 <= stats["gate_mean"] <= 1.0


def test_associative_compose() -> None:
    rng = Random(5)
    d = 16
    a = ([rng.random() for _ in range(d)], [complex(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(d)])
    b = ([rng.random() for _ in range(d)], [complex(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(d)])
    c = ([rng.random() for _ in range(d)], [complex(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(d)])

    left = compose_affine(c, compose_affine(b, a))
    right = compose_affine(compose_affine(c, b), a)

    close_vec(left[0], right[0])
    close_vec(left[1], right[1])


def test_parallel_scan_matches_sequential_prefix() -> None:
    rng = Random(11)
    t = 20
    d = 8
    gs = [[rng.random() for _ in range(d)] for _ in range(t)]
    bs = [[complex(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(d)] for _ in range(t)]

    pg, pb = parallel_prefix_scan(gs, bs)

    sg = []
    sb = []
    acc = ([1.0 for _ in range(d)], [0j for _ in range(d)])
    for i in range(t):
        acc = compose_affine((gs[i], bs[i]), acc)
        sg.append(acc[0])
        sb.append(acc[1])

    for i in range(t):
        close_vec(pg[i], sg[i])
        close_vec(pb[i], sb[i])

# Flux Prototype (testable research scaffold)

Ez a repository egy **tesztelhető Flux-prototípus** a concept paper alapján:
- Holographic State Core (komplex fázor állapot)
- Lossless Tape (append-only memória)
- konstans `K` visszahívás + gate alapú fúzió

## Gyors indítás

```bash
python -m pytest -q
PYTHONPATH=. python benchmarks/run_benchmark.py --seq-len 4096 --d-model 64 --k 8
```

## Mi van implementálva?

- `flux/core.py`:
  - komplex fázor kulcs (`unit_phasor`)
  - holografikus írás/olvasás
  - asszociatív affine kompozíció és párhuzamos prefix scan referencia-implementáció
- `flux/tape.py`:
  - append-only Tape rekordokkal
  - konstans probe-os bucket index
  - konstans-K recall
- `flux/block.py`:
  - `FluxBlock` mint attention drop-in jellegű blokk (`(T,D) -> (T,D)`)
  - core + tape + gate fúzió
- `benchmarks/run_benchmark.py`:
  - egyszerű throughput benchmark különböző szekvenciahosszokra
- `tests/test_flux.py`:
  - shape / tape növekedés
  - asszociativitás
  - scan helyesség (sequential prefix ellen)

## Megjegyzés

Ez kutatási prototípus, nem production LLM stack. Célja, hogy mérhetően és reprodukálhatóan lehessen vizsgálni a koncepciót.

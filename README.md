# blocked-lu-systolic

HW/SW co-design for blocked LU factorization: the CPU does panel factorization, a systolic array does the trailing updates. We ask which block size `b` minimizes total time (Q1) and which array layout — one 128×128 or four 64×64 — minimizes energy-delay product as the submatrix shrinks (Q2).

Graduate computer-architecture coursework (Fall 2026). Please don't submit it as your own.

## Setup

```sh
conda env create -f environment.yml    # or: conda create -n lu python=3.11 "numpy<2.4" matplotlib && pip install scalesim==3.0.0
conda activate lu
```

SCALE-Sim 3.0.0 needs `numpy < 2.4`.

## Layout

```
model/      lu_array_model.py   cycle model + LU update decomposition   (python3 lu_array_model.py table | csv)
            lu_updates.csv      one row per trailing update, n ∈ {2048, 4096}, b ∈ {32..256}, with predicted cycles
            fig_prelim.py       preliminary figure (model only)         (python3 fig_prelim.py → fig_prelim.pdf)
cpu/        blocked-LU implementation + CPU timing harness               (Rina)
sim/        SCALE-Sim config generation, sweep runner, report parser     (Sahana)
results/    CSVs from runs; raw SCALE-Sim reports if small
figures/    generated figures
```

## Cycle model

For an R×C array and a GEMM (M×K)·(K×N), validated against SCALE-Sim 3.0.0 in CALC mode (48 runs, full folds):

- WS/IS: `F·(2R + C + T − 2) − 1`, with (S_R, S_C, T) = (K, N, M) for WS
- OS: `F·(R + C + T − 2) − 1`, with (S_R, S_C, T) = (M, N, K)
- `F = ceil(S_R/R) · ceil(S_C/C)` folds

Trailing update at LU step i on n×n with block b: M = N = n − (i+1)·b, K = b. All of these are partial folds (K ≤ R) — the regime the held-out validation runs test.

## Reproduce

```sh
cd model
python3 lu_array_model.py csv      # → lu_updates.csv
python3 fig_prelim.py              # → fig_prelim.pdf
```

Reports are written in Overleaf; `report/` holds exported copies.

## AI tools

Claude (Anthropic) assisted with the cycle-model code and report drafting. All simulations and measurements are run by the authors.

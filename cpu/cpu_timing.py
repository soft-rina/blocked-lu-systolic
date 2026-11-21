#!/usr/bin/env python3
"""cpu_timing.py -- per-step CPU timing of blocked LU, plus the CPU-only baseline.

Re-runs the loop from blocked_lu.py with perf_counter() around each phase. The trailing update still
executes on the CPU (later steps need its result) but its time is recorded only for reference; the
timing model replaces it with the simulated array time.

    python3 cpu_timing.py --n 2048 --b 32 64 128 256 --repeats 5
    python3 cpu_timing.py --n 2048 4096 --b 32 64 128 256 --repeats 5 -o ../results/cpu_timing.csv

Outputs (appended; existing (machine, n, b) rows are skipped so runs can be resumed):
  results/cpu_timing.csv      one row per LU step: median over repeats of t_panel, t_rowblock, t_update
  results/cpu_baseline.csv    one row per n: median wall time of scipy.linalg.lu_factor on the same matrix
  results/cpu_env.txt         machine, CPU, BLAS, thread count, numpy/scipy versions

Set OMP_NUM_THREADS / OPENBLAS_NUM_THREADS / MKL_NUM_THREADS before running and keep them fixed.
Drafted with Claude (Anthropic) assisting; the LU itself is blocked_lu.py.
"""
import argparse, csv, os, platform, statistics, subprocess, sys, time
import numpy as np
from scipy.linalg import solve_triangular, lu_factor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blocked_lu import panel_lu, panel_lu_lapack  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, '..', 'results')


def timed_blocked_lu(A, b, panel=panel_lu_lapack):
    """Same algorithm as blocked_lu.blocked_lu; returns per-step phase times (seconds)."""
    A = A.copy(); n = A.shape[0]
    steps = []
    for k in range(0, n, b):
        e = min(k + b, n)
        t0 = time.perf_counter()
        piv = panel(A[k:, k:e])
        A[k:, :k] = A[k:, :k][piv]
        A[k:, e:] = A[k:, e:][piv]
        t1 = time.perf_counter()
        if e == n:
            steps.append(dict(step=k // b, M=0, N=0, K=e - k, t_panel=t1 - t0, t_rowblock=0.0, t_update=0.0))
            break
        L11 = np.tril(A[k:e, k:e], -1) + np.eye(e - k)
        A[k:e, e:] = solve_triangular(L11, A[k:e, e:], lower=True, unit_diagonal=True)
        t2 = time.perf_counter()
        A[e:, e:] -= A[e:, k:e] @ A[k:e, e:]
        t3 = time.perf_counter()
        steps.append(dict(step=k // b, M=n - e, N=n - e, K=e - k, t_panel=t1 - t0, t_rowblock=t2 - t1, t_update=t3 - t2))
    return steps


def env_info():
    try:
        cpu = subprocess.run(['lscpu'], capture_output=True, text=True).stdout
        model = next((l.split(':', 1)[1].strip() for l in cpu.splitlines() if l.startswith('Model name')), platform.processor())
    except Exception:
        model = platform.processor()
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        np.show_config()
    cfg = buf.getvalue()
    blas = 'MKL' if 'mkl' in cfg.lower() else 'OpenBLAS' if 'openblas' in cfg.lower() else 'Accelerate' if 'accelerate' in cfg.lower() else 'unknown'
    threads = {k: os.environ.get(k, 'unset') for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}
    return dict(machine=platform.node(), cpu=model, blas=blas, threads=threads,
                numpy=np.__version__, scipy=__import__('scipy').__version__, python=platform.python_version())


def existing(path, keycols):
    if not os.path.exists(path):
        return set()
    with open(path, newline='') as f:
        return {tuple(r[c] for c in keycols) for r in csv.DictReader(f)}


def append(path, rows):
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, nargs='+', default=[2048])
    ap.add_argument('--b', type=int, nargs='+', default=[32, 64, 128, 256])
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--panel', choices=['lapack', 'reference'], default='lapack')
    ap.add_argument('-o', default=os.path.join(RESULTS, 'cpu_timing.csv'))
    args = ap.parse_args()
    panel = panel_lu_lapack if args.panel == 'lapack' else panel_lu
    os.makedirs(RESULTS, exist_ok=True)
    env = env_info()
    with open(os.path.join(RESULTS, 'cpu_env.txt'), 'w') as f:
        for k, v in env.items():
            f.write(f'{k}: {v}\n')
    print('env:', env)
    machine = env['machine']
    base_path = os.path.join(RESULTS, 'cpu_baseline.csv')
    done_t = existing(args.o, ('machine', 'panel', 'n', 'b'))
    done_b = existing(base_path, ('machine', 'n'))

    for n in args.n:
        rng = np.random.default_rng(args.seed)
        A = rng.uniform(-1, 1, (n, n))
        # baseline: scipy's LAPACK LU on the same matrix
        if (machine, str(n)) not in done_b:
            ts = []
            for _ in range(args.repeats):
                t0 = time.perf_counter(); lu_factor(A, check_finite=False); ts.append(time.perf_counter() - t0)
            append(base_path, [dict(machine=machine, n=n, repeats=args.repeats, t_cpu_only_s=f'{statistics.median(ts):.6f}',
                                    t_min_s=f'{min(ts):.6f}', t_max_s=f'{max(ts):.6f}')])
            print(f'baseline n={n}: median {statistics.median(ts):.4f}s  (min {min(ts):.4f}, max {max(ts):.4f})', flush=True)
        for b in args.b:
            if (machine, args.panel, str(n), str(b)) in done_t:
                print(f'skip n={n} b={b} (done)'); continue
            t0 = time.perf_counter()
            runs = [timed_blocked_lu(A, b, panel) for _ in range(args.repeats)]
            rows = []
            for i in range(len(runs[0])):
                med = {ph: statistics.median(r[i][ph] for r in runs) for ph in ('t_panel', 't_rowblock', 't_update')}
                s = runs[0][i]
                rows.append(dict(machine=machine, panel=args.panel, n=n, b=b, step=s['step'], M=s['M'], N=s['N'], K=s['K'],
                                 t_panel_s=f"{med['t_panel']:.6f}", t_rowblock_s=f"{med['t_rowblock']:.6f}",
                                 t_update_cpu_s=f"{med['t_update']:.6f}", repeats=args.repeats))
            append(args.o, rows)
            tot_cpu = sum(float(r['t_panel_s']) + float(r['t_rowblock_s']) for r in rows)
            tot_upd = sum(float(r['t_update_cpu_s']) for r in rows)
            print(f'panel={args.panel} n={n} b={b:>3}: {len(rows)} steps  CPU(panel+rowblock)={tot_cpu:.4f}s  update-on-CPU={tot_upd:.4f}s  [{time.perf_counter()-t0:.1f}s wall]', flush=True)


if __name__ == '__main__':
    main()

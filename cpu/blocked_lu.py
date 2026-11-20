#!/usr/bin/env python3
"""blocked_lu.py -- right-looking blocked LU with partial pivoting, with the CPU/array split made explicit.

Each step k has three phases:
  1. panel   (CPU)   : unblocked LU with pivoting on the (n-k) x b column block
  2. row block (CPU) : U12 = L11^-1 A12, a triangular solve on the b x (n-e) block
  3. trailing update (ARRAY) : A22 -= L21 @ U12, a GEMM with M = N = n-e, K = b

Phase 3 should be replaced by systolic array; its (M, N, K) per step must match model/lu_updates.csv.

    python3 blocked_lu.py                  # residual check at n=256 for b in 32, 64, 128, 256
    python3 blocked_lu.py --n 512 --b 64   # one size

panel_lu written by Rina Li; outer loop and test harness drafted with Claude (Anthropic) assisting.
"""
import argparse
import numpy as np
from scipy.linalg import solve_triangular


def panel_lu(Pn):
    """Unblocked LU with partial pivoting on the panel, in place.
    Returns perm (length m) such that new row i of the panel = old row perm[i]."""
    m, nb = Pn.shape
    perm = np.arange(m)
    for j in range(min(m, nb)):
        p = j + np.argmax(np.abs(Pn[j:, j]))            # pivot: largest magnitude on/below the diagonal
        if Pn[p, j] == 0.0:
            raise np.linalg.LinAlgError(f"zero pivot in panel column {j}")
        if p != j:                                        # swap whole panel rows, incl. factored columns left of j
            Pn[[j, p]] = Pn[[p, j]]
            perm[[j, p]] = perm[[p, j]]
        Pn[j + 1:, j] /= Pn[j, j]                         # multipliers below the diagonal
        Pn[j + 1:, j + 1:] -= np.outer(Pn[j + 1:, j], Pn[j, j + 1:])   # rank-1 update of the rest of the panel
    return perm


def blocked_lu(A, b, log=None):
    """Returns (P, L, U) with P @ A = L @ U.  If `log` is a list, appends one dict per step
    with the trailing-update shape (M, N, K) so it can be checked against lu_updates.csv."""
    A = A.copy()
    n = A.shape[0]
    perm = np.arange(n)
    for k in range(0, n, b):
        e = min(k + b, n)
        # 1. panel (CPU)
        piv = panel_lu(A[k:, k:e])
        A[k:, :k] = A[k:, :k][piv]                        # apply the panel's swaps to the left of the panel ...
        A[k:, e:] = A[k:, e:][piv]                        # ... and to the right
        perm[k:] = perm[k:][piv]
        if e == n:
            break
        # 2. row block (CPU): U12 = L11^-1 A12
        L11 = np.tril(A[k:e, k:e], -1) + np.eye(e - k)
        A[k:e, e:] = solve_triangular(L11, A[k:e, e:], lower=True, unit_diagonal=True)
        # 3. trailing update (ARRAY): A22 -= L21 @ U12
        if log is not None:
            log.append({'step': k // b, 'M': n - e, 'N': n - e, 'K': e - k})
        A[e:, e:] -= A[e:, k:e] @ A[k:e, e:]
    L = np.tril(A, -1) + np.eye(n)
    U = np.triu(A)
    P = np.eye(n)[perm]
    return P, L, U


def residual(A, b):
    P, L, U = blocked_lu(A, b)
    return np.linalg.norm(P @ A - L @ U) / np.linalg.norm(A)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=256)
    ap.add_argument('--b', type=int, nargs='+', default=[32, 64, 128, 256])
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    A = rng.uniform(-1, 1, (args.n, args.n))
    ok = True
    for b in args.b:
        r = residual(A, b)
        ok &= r < 1e-12
        print(f"n={args.n:>5} b={b:>4}  ||PA-LU||/||A|| = {r:.2e}  {'OK' if r < 1e-12 else 'FAIL'}")
    # shape cross-check against the model's decomposition
    log = []
    blocked_lu(A, args.b[0], log=log)
    print(f"\nfirst three trailing updates at b={args.b[0]}: " + ', '.join(f"(M={d['M']}, N={d['N']}, K={d['K']})" for d in log[:3]))
    raise SystemExit(0 if ok else 1)

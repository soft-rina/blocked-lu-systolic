#!/usr/bin/env python3
"""Blocked-LU trailing-update array time from the validated SCALE-Sim cycle formulas.

Formulas (validated on Rina's CSEN 318 SCALE-Sim assignment, scalesim 3.0.0, CALC mode):
  OS : F*(R+C+K-2)-1   F = ceil(M/R)*ceil(N/C)   (S_R=M, S_C=N, T=K)   exact to the cycle on 17 runs
  WS : F*(2R+C+M-2)-1  F = ceil(K/R)*ceil(N/C)   (S_R=K, S_C=N, T=M)   exact to -1 on 16 runs, full folds only
R = ArrayHeight, C = ArrayWidth. GEMM is (M x K) @ (K x N).
Trailing update at LU step i (0-based) on n x n with block b:  M = N = n - (i+1)*b,  K = b.

Usage
  python3 lu_array_model.py table                 # total array cycles per (n, b), OS vs WS, 128x128
  python3 lu_array_model.py csv [-o lu_updates.csv]   # one row per trailing update with predictions
"""
import argparse, csv, math

def os_cyc(R, C, M, N, K): return math.ceil(M / R) * math.ceil(N / C) * (R + C + K - 2) - 1
def ws_cyc(R, C, M, N, K): return math.ceil(K / R) * math.ceil(N / C) * (2 * R + C + M - 2) - 1
def lu_updates(n, b): return [(n - (i + 1) * b, n - (i + 1) * b, b) for i in range(n // b) if n - (i + 1) * b > 0]
def total(n, b, R, C, fn): return sum(fn(R, C, M, N, K) for M, N, K in lu_updates(n, b))
def macs(n, b): return sum(M * N * K for M, N, K in lu_updates(n, b))

NS, BS = (2048, 4096, 8192), (32, 64, 128, 256)

def cmd_table(_):
    R = C = 128
    for n in NS:
        print(f"n={n}: trailing-update MACs ≈ {macs(n,32)/1e9:.2f} G at b=32, {macs(n,256)/1e9:.2f} G at b=256")
        print(f"   b | steps |   OS 128x128 |   WS 128x128 | OS/WS | WS util")
        for b in BS:
            o, w = total(n, b, R, C, os_cyc), total(n, b, R, C, ws_cyc)
            print(f" {b:>3} | {n//b:>5} | {o:>12,} | {w:>12,} | {o/w:5.2f} | {macs(n,b)/(R*C*w):6.1%}")

def cmd_csv(args):
    rows = []
    for n in args.n:
        for b in BS:
            ups = lu_updates(n, b)
            for i, (M, N, K) in enumerate(ups):
                phase = 'early' if i < len(ups) / 3 else 'middle' if i < 2 * len(ups) / 3 else 'late'
                rows.append(dict(n=n, b=b, step=i, phase=phase, M=M, N=N, K=K,
                                 os_pred_128=os_cyc(128, 128, M, N, K), ws_pred_128=ws_cyc(128, 128, M, N, K),
                                 os_pred_64=os_cyc(64, 64, M, N, K), ws_pred_64=ws_cyc(64, 64, M, N, K)))
    with open(args.o, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} rows -> {args.o}   (os_pred_64/ws_pred_64 are for ONE 64x64 array; for 4x64x64 split N by 4 and take the slowest shard)")

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('table').set_defaults(fn=cmd_table)
    p = sub.add_parser('csv'); p.add_argument('-o', default='lu_updates.csv'); p.add_argument('--n', type=int, nargs='+', default=[2048, 4096]); p.set_defaults(fn=cmd_csv)
    a = ap.parse_args(); a.fn(a)

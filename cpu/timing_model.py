#!/usr/bin/env python3
"""timing_model.py -- step-level timing model  T_i = t_CPU_i + t_link_i + t_array_i  for blocked LU on CPU + array.

  t_CPU   measured: panel + row block per step, from results/cpu_timing.csv (cpu_timing.py)
  t_link  modeled : PCIe, alpha + bytes/beta per transfer; panel+row block out, updated submatrix back (naive)
  t_array simulated: WS cycle formula (validated) / f_array

Writes results/timing_model.csv (one row per n, b, panel, variant) and prints a table against the CPU-only baseline.
Variants: sequential (sum of all three) and overlap (per step: max(t_CPU_{i+1}, t_array_i) -- panel i+1 overlaps
update i, which is valid because panel i+1 only needs its own b columns of A22, assumed delivered first).

    python3 timing_model.py                         # defaults: 128x128, WS, 1 GHz, FP64, alpha=2us, beta=64GB/s
    python3 timing_model.py --R 64 --C 64 --shards 4 # four 64x64 arrays, N split four ways, slowest shard counts

Drafted with Claude (Anthropic) assisting.
"""
import argparse, csv, math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lu_array_model import ws_cyc, os_cyc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, '..', 'results')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--timing', default=os.path.join(RESULTS, 'cpu_timing.csv'))
    ap.add_argument('--baseline', default=os.path.join(RESULTS, 'cpu_baseline.csv'))
    ap.add_argument('-o', default=os.path.join(RESULTS, 'timing_model.csv'))
    ap.add_argument('--R', type=int, default=128); ap.add_argument('--C', type=int, default=128)
    ap.add_argument('--shards', type=int, default=1, help='number of arrays; N is split across them, slowest shard counts')
    ap.add_argument('--dataflow', choices=['ws', 'os'], default='ws')
    ap.add_argument('--f', type=float, default=1e9, help='array clock, Hz')
    ap.add_argument('--word', type=int, default=8, help='bytes per element (8 = FP64, 2 = BF16)')
    ap.add_argument('--alpha', type=float, default=2e-6); ap.add_argument('--beta', type=float, default=64e9)
    ap.add_argument('--resident', action='store_true', help='matrix resident on the array: ship panel+row block out and the next panel columns back, never A22')
    a = ap.parse_args()
    cyc = ws_cyc if a.dataflow == 'ws' else os_cyc

    base = {(r['machine'], int(r['n'])): float(r['t_cpu_only_s']) for r in csv.DictReader(open(a.baseline))}
    steps = {}
    for r in csv.DictReader(open(a.timing)):
        key = (r['machine'], r['panel'], int(r['n']), int(r['b']))
        steps.setdefault(key, []).append(r)

    out = []
    for (machine, panel, n, b), rows in sorted(steps.items()):
        rows.sort(key=lambda r: int(r['step']))
        t_cpu, t_link, t_arr = [], [], []
        for r in rows:
            M, N, K = int(r['M']), int(r['N']), int(r['K'])
            t_cpu.append(float(r['t_panel_s']) + float(r['t_rowblock_s']))
            if M == 0:                               # last panel, no update
                t_link.append(0.0); t_arr.append(0.0); continue
            Ns = math.ceil(N / a.shards)
            t_arr.append(cyc(a.R, a.C, M, Ns, K) / a.f)
            out_b = (M * K + K * N) * a.word
            back_b = (M * K) * a.word if a.resident else (M * N) * a.word
            t_link.append((a.alpha + out_b / a.beta) + (a.alpha + back_b / a.beta))
        seq = sum(t_cpu) + sum(t_link) + sum(t_arr)
        # overlap: panel i+1 overlaps array update i; link stays serial
        ovl = t_cpu[0] + sum(t_link)
        for i in range(len(rows) - 1):
            ovl += max(t_cpu[i + 1], t_arr[i])
        ovl += t_arr[-1]
        bl = base.get((machine, n))
        for variant, tot in (('sequential', seq), ('overlap', ovl)):
            out.append(dict(machine=machine, panel=panel, n=n, b=b, R=a.R, C=a.C, shards=a.shards, dataflow=a.dataflow,
                            resident=int(a.resident), variant=variant,
                            t_cpu_s=f'{sum(t_cpu):.6f}', t_link_s=f'{sum(t_link):.6f}', t_array_s=f'{sum(t_arr):.6f}',
                            total_s=f'{tot:.6f}', baseline_s=f'{bl:.6f}' if bl else '', speedup=f'{bl / tot:.4f}' if bl else ''))
    os.makedirs(RESULTS, exist_ok=True)
    with open(a.o, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
    print(f"{a.shards}x{a.R}x{a.C} {a.dataflow.upper()} @ {a.f/1e9:g} GHz, {a.word} B/elem, PCIe {a.alpha*1e6:g} us + {a.beta/1e9:g} GB/s"
          f"{', matrix resident' if a.resident else ''}\n")
    print(f"{'machine':<8} {'panel':<9} {'n':>5} {'b':>4} {'variant':<10} {'t_CPU':>8} {'t_link':>8} {'t_array':>8} {'total':>8} {'baseline':>8} {'speedup':>7}")
    for r in out:
        print(f"{r['machine']:<8} {r['panel']:<9} {r['n']:>5} {r['b']:>4} {r['variant']:<10} {float(r['t_cpu_s']):>8.4f} {float(r['t_link_s']):>8.4f} "
              f"{float(r['t_array_s']):>8.5f} {float(r['total_s']):>8.4f} {r['baseline_s']:>8} {r['speedup']:>7}")
    print(f"\nwrote {a.o}")


if __name__ == '__main__':
    main()

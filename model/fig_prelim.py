#!/usr/bin/env python3
"""fig_prelim.py -- model-only preliminary figure created for the mid-progress report (Q1, array side).
Total trailing-update array cycles vs block size b, OS vs WS, single 128x128 array, from lu_array_model.
NOT validated against SCALE-Sim yet (partial-fold regime). Run: python3 fig_prelim.py -> fig_prelim.pdf
"""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lu_array_model import total, os_cyc, ws_cyc, macs, NS, BS
R = C = 128
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
for n, mk in zip(NS, ('o', 's', '^')):
    a1.plot(BS, [total(n, b, R, C, os_cyc) / 1e6 for b in BS], mk + '--', color='tab:red', label=f'OS, n={n}')
    a1.plot(BS, [total(n, b, R, C, ws_cyc) / 1e6 for b in BS], mk + '-', color='tab:blue', label=f'WS, n={n}')
    a2.plot(BS, [100 * macs(n, b) / (R * C * total(n, b, R, C, ws_cyc)) for b in BS], mk + '-', color='tab:blue', label=f'WS, n={n}')
    a2.plot(BS, [100 * macs(n, b) / (R * C * total(n, b, R, C, os_cyc)) for b in BS], mk + '--', color='tab:red', label=f'OS, n={n}')
for a in (a1, a2):
    a.set_xscale('log', base=2); a.set_xticks(BS); a.set_xticklabels(BS); a.set_xlabel('block size $b$'); a.grid(True, alpha=0.3)
a1.set_yscale('log'); a1.set_ylabel('total array cycles, all LU steps (M)'); a1.set_title('(a) array time vs. $b$, 128$\\times$128', fontsize=10)
a2.set_ylabel('array utilization over the factorization (%)'); a2.set_title('(b) utilization vs. $b$', fontsize=10); a2.set_ylim(0, 100)
a1.legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig('fig_prelim.pdf'); print('wrote fig_prelim.pdf')

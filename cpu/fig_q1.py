#!/usr/bin/env python3
"""fig_q1.py -- Plot 1 of the proposal: total blocked-LU time vs. block size b, from results/timing_model.csv.

Left: total time per b for each (panel, variant), CPU-only LAPACK baseline as a reference line, one panel per n.
Right: stacked t_CPU / t_link / t_array for the LAPACK-panel sequential variant, so the breakdown is visible.

    python3 fig_q1.py                      # -> ../figures/fig_q1.pdf
Drafted with Claude (Anthropic) assisting.
"""
import csv, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
rows = list(csv.DictReader(open(os.path.join(HERE, '..', 'results', 'timing_model.csv'))))
ns = sorted({int(r['n']) for r in rows})
fig, axes = plt.subplots(1, 1 + len(ns), figsize=(4.2 * (1 + len(ns)), 3.6))
axes = list(axes)
styles = {('lapack', 'sequential'): ('tab:blue', '-', 'o'), ('lapack', 'overlap'): ('tab:blue', '--', 's'),
          ('reference', 'sequential'): ('tab:gray', '-', 'o'), ('reference', 'overlap'): ('tab:gray', '--', 's')}

for ax, n in zip(axes, ns):
    sub = [r for r in rows if int(r['n']) == n]
    for (panel, variant), (c, ls, mk) in styles.items():
        pts = sorted((int(r['b']), float(r['total_s'])) for r in sub if r['panel'] == panel and r['variant'] == variant)
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=c, ls=ls, marker=mk, label=f'{panel} panel, {variant}')
    bl = float(sub[0]['baseline_s'])
    ax.axhline(bl, color='tab:red', lw=1.2, label=f'CPU-only LAPACK ({bl*1e3:.0f} ms)')
    ax.set_xscale('log', base=2); ax.set_xticks([32, 64, 128, 256]); ax.set_xticklabels([32, 64, 128, 256])
    ax.set_xlabel('block size $b$'); ax.set_ylabel('total time (s)'); ax.set_title(f'$n={n}$', fontsize=10)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7)

ax = axes[-1]
n = ns[-1]
sub = sorted((r for r in rows if int(r['n']) == n and r['panel'] == 'lapack' and r['variant'] == 'sequential'), key=lambda r: int(r['b']))
bs = [str(r['b']) for r in sub]
cpu = [float(r['t_cpu_s']) for r in sub]; link = [float(r['t_link_s']) for r in sub]; arr = [float(r['t_array_s']) for r in sub]
ax.bar(bs, cpu, label='$t_{CPU}$ (measured)', color='tab:blue')
ax.bar(bs, link, bottom=cpu, label='$t_{link}$ (PCIe model)', color='tab:orange')
ax.bar(bs, arr, bottom=[c + l for c, l in zip(cpu, link)], label='$t_{array}$ (simulated)', color='tab:green')
ax.set_xlabel('block size $b$'); ax.set_ylabel('time (s)'); ax.set_title(f'breakdown, LAPACK panel, $n={n}$', fontsize=10)
ax.legend(fontsize=7); ax.grid(True, axis='y', alpha=0.3)

fig.tight_layout()
out = os.path.join(HERE, '..', 'figures', 'fig_q1.pdf')
fig.savefig(out); print('wrote', out)

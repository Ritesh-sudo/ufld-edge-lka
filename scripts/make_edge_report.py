"""Join results/benchmark*.csv with results/backend_accuracy.csv into one table + accuracy-vs-latency plot."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = Path('results')
acc = {r['backend']: r for r in csv.DictReader(open(R / 'backend_accuracy.csv'))}
# Backends whose accuracy row has a different label than the benchmark row.
ACC_KEY = {'torch-mps': 'torch-cpu', 'coreml-gpu': 'coreml-cpu', 'coreml-all': 'coreml-ne', 'onnx-coreml': None}
rows = []
for f in ('benchmark.csv', 'benchmark_int8.csv'):
    for r in csv.DictReader(open(R / f)):
        if r['stage'] != 'model':
            continue
        name = 'onnx-cpu-int8' if 'int8' in r['note'] else r['backend']
        key = name if name in acc else ACC_KEY.get(name)
        a = acc.get(key)
        rows.append(dict(backend=name, mean_ms=float(r['mean_ms']), p50_ms=r['p50_ms'], p95_ms=r['p95_ms'], fps=r['fps_mean'],
                         accuracy_pct=a['accuracy_pct'] if a else '', acc_source=('measured' if key == name else f'same as {key}') if a else 'not measured'))
rows.sort(key=lambda r: r['mean_ms'])
with open(R / 'edge_summary.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)

fig, ax = plt.subplots(figsize=(7, 4.2))
for r in rows:
    if r['accuracy_pct']:
        ax.scatter(r['mean_ms'], float(r['accuracy_pct']), s=40)
        ax.annotate(r['backend'], (r['mean_ms'], float(r['accuracy_pct'])), fontsize=7, xytext=(4, 3), textcoords='offset points')
ax.set_xscale('log'); ax.set_xlabel('latency, model only (ms, log scale), M4, batch 1, 288x800')
ax.set_ylabel('TuSimple accuracy (%)'); ax.set_title('UFLD ResNet-18: accuracy vs latency by backend')
ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(R / 'edge_accuracy_vs_latency.png', dpi=150)
for r in rows:
    print(f"{r['backend']:14s} {r['mean_ms']:6.2f} ms {r['fps']:>6s} FPS  acc={r['accuracy_pct'] or 'n/a':>8s} ({r['acc_source']})")

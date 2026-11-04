"""One fair latency harness + one table/chart for UFLD, CLRNet and the OpenCV Hough baseline.

python scripts/compare.py --out results/comparison.csv

Fairness rules (CLAUDE.md): same device pairs, batch size 1, same warm-up (30) and timed runs (200) on one fixed real
frame, each model at its NATIVE input size, authors' unmodified weights. Two numbers per row:
  model        forward pass only (host<->device copies + sync included, image decode/resize excluded)
  model+decode forward + the model's own post-processing (UFLD: softmax/argmax decode; CLRNet: conf filter + lane NMS +
               spline sampling; Hough has no network, so its single number is the whole detector)
Accuracy is read from results/backend_accuracy.csv (full 2,782-image TuSimple test set, same LaneEval for all).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ufld  # noqa: F401

import argparse
import csv

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import time_fn  # noqa: E402

from baselines.opencv_hough import detect_lanes  # noqa: E402
from third_party.clrnet import CLRConfig, decode_lanes, lanes_to_tusimple, load_clrnet, preprocess_bgr  # noqa: E402
from ufld.backends import load_coreml, load_torch  # noqa: E402
from ufld.config import load_config  # noqa: E402
from ufld.evaluation import generate_tusimple_lines  # noqa: E402
from ufld.parity import sample_inputs  # noqa: E402


def sync(dev):
    if dev.type == 'mps':
        torch.mps.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/tusimple_res18.py')
    ap.add_argument('--ufld-ckpt', default='data/checkpoints/tusimple_18.pth')
    ap.add_argument('--clrnet-ckpt', default='data/checkpoints/clrnet_dl/tusimple_r18.pth')
    ap.add_argument('--coreml', default='data/exports/ufld_res18_fp16.mlpackage')
    ap.add_argument('--warmup', type=int, default=30)
    ap.add_argument('--runs', type=int, default=200)
    ap.add_argument('--out', default='results/comparison.csv')
    args = ap.parse_args()

    ucfg, ccfg = load_config(args.config), CLRConfig()
    name = [l.split()[0].lstrip('/') for l in open(Path(ucfg.data_root) / 'test.txt')][0]
    frame = cv2.imread(str(Path(ucfg.data_root) / name))
    ux = sample_inputs(ucfg, 1)[0]
    cx = preprocess_bgr(frame, ccfg)
    hs = ucfg.test_h_samples

    rows = []   # (key, label, native input, device, model_fn, total_fn)
    for dev_name in ('mps', 'cpu'):
        dev = torch.device(dev_name)
        u = load_torch(ucfg, args.ufld_ckpt, dev)
        rows.append((f'ufld-torch-{dev_name}', f'UFLD R18 (PyTorch, {dev_name})', '288x800', dev_name, lambda u=u: u(ux),
                     lambda u=u: generate_tusimple_lines(torch.from_numpy(u(ux)[0]), ucfg.griding_num, ucfg.orig_size[1])))
        m = load_clrnet(args.clrnet_ckpt, dev, ccfg)
        xt = torch.from_numpy(cx).to(dev)

        def fwd(m=m, xt=xt, dev=dev):
            with torch.no_grad():
                o = m(xt)
            sync(dev)
            return o.cpu()
        rows.append((f'clrnet-torch-{dev_name}', f'CLRNet R18 (PyTorch, {dev_name})', '320x800', dev_name, fwd,
                     lambda f=fwd: lanes_to_tusimple(decode_lanes(f(), ccfg), hs, ccfg)))
    ane = load_coreml(args.coreml, 'ALL')
    rows.append(('ufld-coreml-ane', 'UFLD R18 (Core ML fp16, ANE)', '288x800', 'ane', lambda: ane(ux),
                 lambda: generate_tusimple_lines(torch.from_numpy(ane(ux)[0]), ucfg.griding_num, ucfg.orig_size[1])))
    hough = lambda: detect_lanes(frame, hs)  # noqa: E731
    rows.append(('opencv-hough', 'OpenCV Canny+Hough', '720x1280', 'cpu', None, hough))

    acc = {r['backend']: float(r['accuracy_pct']) for r in csv.DictReader(open('results/backend_accuracy.csv'))}
    acc_key = {'ufld-torch-mps': 'torch-cpu', 'ufld-torch-cpu': 'torch-cpu', 'clrnet-torch-mps': 'clrnet-r18-mps',
               'clrnet-torch-cpu': 'clrnet-r18-mps', 'ufld-coreml-ane': 'coreml-ne', 'opencv-hough': 'opencv-hough'}
    out = []
    for key, label, size, dev, mf, tf in rows:
        mm = time_fn(mf, args.warmup, args.runs) if mf else None
        tm = time_fn(tf, args.warmup, args.runs)
        a = acc[acc_key[key]]
        out.append(dict(key=key, model=label, input=size, device=dev,
                        model_ms=f'{mm[0]:.2f}' if mm else '', model_p95_ms=f'{mm[2]:.2f}' if mm else '',
                        total_ms=f'{tm[0]:.2f}', total_p95_ms=f'{tm[2]:.2f}', accuracy_pct=f'{a:.4f}'))
        print(f"{label:34s} {size:9s} model={out[-1]['model_ms'] or 'n/a':>6s} ms  +decode={tm[0]:6.2f} ms  acc={a:.2f}%")
    Path(args.out).parent.mkdir(exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader(); w.writerows(out)

    color = {'ufld': '#2a6fdb', 'clrnet': '#e07b00', 'opencv': '#777777'}
    marker = {'mps': 'o', 'cpu': 's', 'ane': '^'}
    # label offsets (points) chosen per point so the zoom panel stays readable
    off = {'ufld-coreml-ane': (6, -14), 'ufld-torch-mps': (6, -14), 'ufld-torch-cpu': (6, -14),
           'clrnet-torch-mps': (6, 6), 'clrnet-torch-cpu': (-100, 8), 'opencv-hough': (6, 6)}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={'width_ratios': [1, 1.4]})
    for ax, zoom in zip(axes, (False, True)):
        for r in out:
            x, y = float(r['total_ms']), float(r['accuracy_pct'])
            ax.scatter(x, y, s=55, c=color[r['key'].split('-')[0]], marker=marker[r['device']], zorder=3)
            if (zoom and y > 90) or (not zoom and r['key'] == 'opencv-hough'):
                ax.annotate(r['model'].replace(' R18', ''), (x, y), fontsize=7, xytext=off[r['key']], textcoords='offset points')
        ax.set_xscale('log'); ax.grid(alpha=.3)
        ax.set_xlabel('latency incl. post-processing (ms, log)')
        if zoom:
            ax.set_ylim(95.3, 97.2); ax.set_title('zoom: deep models only')
        else:
            ax.set_ylabel('TuSimple accuracy (%)'); ax.set_title('full range')
    fig.suptitle('Lane detection on TuSimple: accuracy vs latency (Apple M4, batch 1, native input; blue UFLD, orange CLRNet, grey Hough)', fontsize=9)
    fig.tight_layout(); fig.savefig('results/comparison_accuracy_vs_latency.png', dpi=150)

if __name__ == '__main__':
    main()

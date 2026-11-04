"""Score the authors' CLRNet ResNet-18 TuSimple checkpoint with OUR TuSimple metric (same LaneEval as UFLD), and time it.

python scripts/eval_clrnet.py --ckpt data/checkpoints/clrnet_dl/tusimple_r18.pth --device mps --out results/backend_accuracy.csv

Timing: batch 1, native 320x800 input, warm-up excluded. `model` = network forward (+ sync);
`model+decode` = forward + confidence filter + lane NMS + spline sampling to TuSimple rows (image decode/resize excluded).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ufld  # noqa: F401  (sets the MPS CPU-fallback env var before torch)

import argparse
import csv
import json
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm

from third_party.clrnet import CLRConfig, decode_lanes, lanes_to_tusimple, load_clrnet, preprocess_bgr
from ufld.config import load_config
from ufld.device import get_device
from ufld.evaluation import LaneEval


def sync(device):
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/tusimple_res18.py', help='UFLD config, used only for data paths and h_samples')
    ap.add_argument('--ckpt', default='data/checkpoints/clrnet_dl/tusimple_r18.pth')
    ap.add_argument('--device', default=None)
    ap.add_argument('--subset', type=int, default=None, help='quick run; skips the metric (it needs the full set)')
    ap.add_argument('--warmup', type=int, default=30)
    ap.add_argument('--out', default=None, help='append accuracy row to this CSV')
    ap.add_argument('--bench-csv', default=None)
    args = ap.parse_args()

    ucfg, cfg = load_config(args.config), CLRConfig()
    device = get_device(args.device)
    model = load_clrnet(args.ckpt, device, cfg)
    names = [l.split()[0].lstrip('/') for l in open(Path(ucfg.data_root) / 'test.txt')][:args.subset]
    pred_path = Path(ucfg.test_work_dir) / 'pred_clrnet.json'
    pred_path.parent.mkdir(parents=True, exist_ok=True)

    t_model, t_total = [], []
    with open(pred_path, 'w') as f, torch.no_grad():
        for i, n in enumerate(tqdm(names, desc='clrnet')):
            x = torch.from_numpy(preprocess_bgr(cv2.imread(str(Path(ucfg.data_root) / n)), cfg))
            t0 = time.perf_counter()
            out = model(x.to(device))
            sync(device)
            t1 = time.perf_counter()
            lanes = lanes_to_tusimple(decode_lanes(out.cpu(), cfg), ucfg.test_h_samples, cfg)
            t2 = time.perf_counter()
            if i >= args.warmup:
                t_model.append((t1 - t0) * 1000)
                t_total.append((t2 - t0) * 1000)
            f.write(json.dumps({'lanes': lanes, 'h_samples': list(ucfg.test_h_samples), 'raw_file': n, 'run_time': 10}) + '\n')

    tm, tt = np.array(t_model), np.array(t_total)
    lat = {s: (a.mean(), np.percentile(a, 50), np.percentile(a, 95)) for s, a in (('model', tm), ('model+decode', tt))}
    for s, (m, p50, p95) in lat.items():
        print(f'clrnet {device} {s:13s} mean={m:.2f} ms p50={p50:.2f} p95={p95:.2f} ({1000 / m:.1f} FPS)  n={len(tm)} (warm-up {args.warmup} excluded)')
    if args.subset:
        return
    res = LaneEval.bench_one_submit(pred_path, Path(ucfg.data_root) / 'test_label.json')
    print(f"clrnet: images={len(names)} Accuracy={res['Accuracy'] * 100:.4f}% FP={res['FP']:.4f} FN={res['FN']:.4f}")
    if args.out:
        p = Path(args.out)
        new = not p.exists()
        with open(p, 'a', newline='') as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(['backend', 'images', 'accuracy_pct', 'fp', 'fn'])
            w.writerow([f'clrnet-r18-{device.type}', len(names), f"{res['Accuracy'] * 100:.4f}", f"{res['FP']:.6f}", f"{res['FN']:.6f}"])
    if args.bench_csv:
        p = Path(args.bench_csv)
        new = not p.exists()
        with open(p, 'a', newline='') as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(['backend', 'input', 'batch', 'warmup', 'runs', 'stage', 'mean_ms', 'p50_ms', 'p95_ms', 'fps_mean', 'note'])
            for s, (m, p50, p95) in lat.items():
                w.writerow([f'clrnet-r18-{device.type}', '320x800', 1, args.warmup, len(tm), s, f'{m:.3f}', f'{p50:.3f}', f'{p95:.3f}', f'{1000 / m:.1f}',
                            'pure-PyTorch NMS port; timed over the test set (not a fixed frame)'])


if __name__ == '__main__':
    main()

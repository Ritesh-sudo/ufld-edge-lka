"""Latency benchmark across backends. Batch size 1, warm-up excluded, fixed number of timed runs.

python scripts/benchmark.py --config configs/tusimple_res18.py --ckpt data/checkpoints/tusimple_18.pth \
    --onnx data/exports/ufld_res18.onnx --coreml data/exports/ufld_res18_fp16.mlpackage \
    --backends torch-mps torch-cpu onnx-cpu coreml-cpu coreml-gpu coreml-ne coreml-all --out results/benchmark.csv

Two numbers per backend: `model` = forward pass only (incl. host<->device copies and sync);
`model+decode` = forward + turning logits into lane x-coordinates (softmax/argmax, as in
evaluation). Image loading/resizing is excluded from both. The input is one fixed real frame,
so every backend does identical work. M4 numbers are not comparable to the paper's desktop GPU.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ufld  # noqa: F401

import argparse
import csv
import platform
import time

import numpy as np
import torch

from ufld.backends import load_coreml, load_onnx, load_torch
from ufld.config import load_config
from ufld.evaluation import generate_tusimple_lines
from ufld.parity import sample_inputs

COREML_UNITS = {'coreml-cpu': 'CPU_ONLY', 'coreml-gpu': 'CPU_AND_GPU', 'coreml-ne': 'CPU_AND_NE', 'coreml-all': 'ALL'}


def make_backend(name, args, cfg):
    if name == 'torch-mps':
        return load_torch(cfg, args.ckpt, torch.device('mps'))
    if name == 'torch-cpu':
        return load_torch(cfg, args.ckpt, torch.device('cpu'))
    if name == 'onnx-cpu':
        return load_onnx(args.onnx)
    if name == 'onnx-coreml':  # ONNX Runtime's Core ML execution provider
        return load_onnx(args.onnx, ['CoreMLExecutionProvider', 'CPUExecutionProvider'])
    if name in COREML_UNITS:
        return load_coreml(args.coreml, COREML_UNITS[name])
    raise ValueError(name)


def time_fn(fn, warmup, runs):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    return ts.mean(), np.percentile(ts, 50), np.percentile(ts, 95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--onnx')
    ap.add_argument('--coreml')
    ap.add_argument('--backends', nargs='+', required=True)
    ap.add_argument('--warmup', type=int, default=30)
    ap.add_argument('--runs', type=int, default=200)
    ap.add_argument('--out', default='results/benchmark.csv')
    ap.add_argument('--tag', default='fp32/fp16 as exported', help='free-text precision note')
    args = ap.parse_args()

    cfg = load_config(args.config)
    x = sample_inputs(cfg, 1)[0]
    h, w = cfg.input_size
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with open(out, 'a', newline='') as f:
        wr = csv.writer(f)
        if new:
            wr.writerow(['backend', 'input', 'batch', 'warmup', 'runs', 'stage', 'mean_ms', 'p50_ms', 'p95_ms',
                         'fps_mean', 'host', 'note'])
        for name in args.backends:
            fn = make_backend(name, args, cfg)
            model_only = lambda: fn(x)  # noqa: E731
            with_decode = lambda: generate_tusimple_lines(torch.from_numpy(fn(x)[0]), cfg.griding_num, cfg.orig_size[1])  # noqa: E731
            for stage, g in (('model', model_only), ('model+decode', with_decode)):
                mean, p50, p95 = time_fn(g, args.warmup, args.runs)
                print(f'{name:12s} {stage:13s} mean={mean:7.2f} ms  p50={p50:7.2f}  p95={p95:7.2f}  ({1000 / mean:6.1f} FPS)')
                wr.writerow([name, f'{h}x{w}', 1, args.warmup, args.runs, stage, f'{mean:.3f}', f'{p50:.3f}',
                             f'{p95:.3f}', f'{1000 / mean:.1f}', platform.platform(terse=True), args.tag])
            f.flush()


if __name__ == '__main__':
    main()

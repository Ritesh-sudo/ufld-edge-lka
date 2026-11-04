"""Score the Canny+Hough baseline with the official TuSimple metric and time it.

python scripts/eval_hough.py --config configs/tusimple_res18.py --out results/backend_accuracy.csv
Runs on the raw 1280x720 frames (no resize), CPU only. Latency = detect_lanes() per frame
(edges + Hough + line fit), image decoding excluded, batch size 1.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
import time

import cv2
import numpy as np
from tqdm import tqdm

from baselines.opencv_hough import detect_lanes
from ufld.config import load_config
from ufld.evaluation import LaneEval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--subset', type=int, default=None)
    ap.add_argument('--out', default='results/backend_accuracy.csv')
    ap.add_argument('--bench-csv', default='results/benchmark_hough.csv')
    args = ap.parse_args()

    cfg = load_config(args.config)
    names = [l.split()[0] for l in open(Path(cfg.data_root) / 'test.txt')]
    names = [n.lstrip('/') for n in names][:args.subset]
    pred = Path(cfg.test_work_dir) / 'pred_opencv-hough.json'
    pred.parent.mkdir(parents=True, exist_ok=True)
    ts = []
    with open(pred, 'w') as f:
        for n in tqdm(names, desc='hough'):
            img = cv2.imread(str(Path(cfg.data_root) / n))
            t0 = time.perf_counter()
            lanes = detect_lanes(img, cfg.test_h_samples)
            ts.append((time.perf_counter() - t0) * 1000)
            f.write(json.dumps({'lanes': lanes, 'h_samples': list(cfg.test_h_samples), 'raw_file': n, 'run_time': 10}) + '\n')
    res = LaneEval.bench_one_submit(pred, Path(cfg.data_root) / 'test_label.json')
    ts = np.array(ts)
    print(f"opencv-hough: images={len(names)} Accuracy={res['Accuracy'] * 100:.4f}% FP={res['FP']:.4f} FN={res['FN']:.4f} "
          f"latency mean={ts.mean():.2f} p50={np.percentile(ts, 50):.2f} p95={np.percentile(ts, 95):.2f} ms")
    if args.subset:
        return
    for path, header, row in (
        (args.out, ['backend', 'images', 'accuracy_pct', 'fp', 'fn'],
         ['opencv-hough', len(names), f"{res['Accuracy'] * 100:.4f}", f"{res['FP']:.6f}", f"{res['FN']:.6f}"]),
        (args.bench_csv, ['backend', 'input', 'images', 'mean_ms', 'p50_ms', 'p95_ms', 'note'],
         ['opencv-hough', '720x1280', len(names), f'{ts.mean():.3f}', f'{np.percentile(ts, 50):.3f}', f'{np.percentile(ts, 95):.3f}',
          'CPU, detect_lanes only, no warm-up (stateless)'])):
        p = Path(path)
        new = not p.exists()
        with open(p, 'a', newline='') as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(header)
            w.writerow(row)


if __name__ == '__main__':
    main()

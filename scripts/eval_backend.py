"""TuSimple Accuracy/FP/FN for any backend (torch / onnx / coreml), so speed is always paired with accuracy.

python scripts/eval_backend.py --config configs/tusimple_res18.py --ckpt data/checkpoints/tusimple_18.pth \
    --backend coreml-ne --coreml data/exports/ufld_res18_fp16.mlpackage --out results/backend_accuracy.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ufld  # noqa: F401

import argparse
import csv
import json

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import make_backend  # noqa: E402
from ufld.config import load_config
from ufld.data import build_test_loader
from ufld.evaluation import LaneEval, generate_tusimple_lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--backend', required=True)
    ap.add_argument('--onnx')
    ap.add_argument('--coreml')
    ap.add_argument('--subset', type=int, default=None)
    ap.add_argument('--label', default=None, help='row name in the CSV (default: backend)')
    ap.add_argument('--out', default='results/backend_accuracy.csv')
    args = ap.parse_args()

    cfg = load_config(args.config)
    fn = make_backend(args.backend, args, cfg)
    loader = build_test_loader(cfg, torch.device('cpu'), batch_size=1, subset=args.subset)
    pred = Path(cfg.test_work_dir) / f'pred_{args.label or args.backend}.json'
    pred.parent.mkdir(parents=True, exist_ok=True)
    with open(pred, 'w') as f:
        for img, names in tqdm(loader, desc=args.backend):
            out = torch.from_numpy(fn(img.numpy()))
            f.write(json.dumps({'lanes': generate_tusimple_lines(out[0], cfg.griding_num, cfg.orig_size[1]),
                                'h_samples': list(cfg.test_h_samples), 'raw_file': names[0], 'run_time': 10}) + '\n')
    res = LaneEval.bench_one_submit(pred, Path(cfg.data_root) / 'test_label.json')
    n = len(loader.dataset)
    print(f"{args.label or args.backend}: images={n} Accuracy={res['Accuracy'] * 100:.4f}% FP={res['FP']:.4f} FN={res['FN']:.4f}")

    out = Path(args.out)
    new = not out.exists()
    with open(out, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['backend', 'images', 'accuracy_pct', 'fp', 'fn'])
        w.writerow([args.label or args.backend, n, f"{res['Accuracy'] * 100:.4f}", f"{res['FP']:.6f}", f"{res['FN']:.6f}"])


if __name__ == '__main__':
    main()

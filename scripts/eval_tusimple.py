"""Evaluate a checkpoint on the TuSimple test set with the official metric (Accuracy, FP, FN).

python scripts/eval_tusimple.py --config configs/tusimple_res18.py --ckpt data/checkpoints/tusimple_18.pth
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ufld  # noqa: F401  (must precede torch: sets the MPS CPU-fallback env var)

import argparse
import csv
import logging

from ufld.checkpoint import load_model_weights
from ufld.config import load_config, parse_overrides
from ufld.data import build_test_loader
from ufld.device import get_device
from ufld.evaluation import evaluate_tusimple
from ufld.model import build_model
from ufld.utils import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--device', default=None, help='cuda | mps | cpu (default: best available)')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--subset', type=int, default=None, help='evaluate only the first N test images')
    ap.add_argument('--out', default=None, help='append a result row to this CSV')
    ap.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE', help='config overrides')
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    cfg = load_config(args.config, parse_overrides(args.set))
    set_seed(cfg.seed)
    device = get_device(args.device)

    # No auxiliary branch at inference; it exists only to help training.
    model = build_model(cfg, pretrained=False, use_aux=False)
    load_model_weights(model, args.ckpt, device='cpu')
    model.to(device)

    loader = build_test_loader(cfg, device, args.batch_size, args.subset)
    res = evaluate_tusimple(model, loader, device, cfg, cfg.test_work_dir)
    n = len(loader.dataset)
    print(f"images={n} device={device} Accuracy={res['Accuracy'] * 100:.2f}% FP={res['FP']:.4f} FN={res['FN']:.4f}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        new = not out.exists()
        with open(out, 'a', newline='') as f:
            w = csv.writer(f)
            if new:
                w.writerow(['config', 'ckpt', 'device', 'images', 'accuracy_pct', 'fp', 'fn'])
            w.writerow([Path(args.config).name, Path(args.ckpt).name, device, n,
                        f"{res['Accuracy'] * 100:.4f}", f"{res['FP']:.6f}", f"{res['FN']:.6f}"])


if __name__ == '__main__':
    main()

"""Export the inference model (no aux branch) to ONNX and check it against PyTorch.

python scripts/export_onnx.py --config configs/tusimple_res18.py --ckpt data/checkpoints/tusimple_18.pth \
    --out data/exports/ufld_res18.onnx
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ufld  # noqa: F401  (must precede torch: sets the MPS CPU-fallback env var)

import argparse

import torch

from ufld.backends import load_onnx, load_torch
from ufld.checkpoint import load_model_weights
from ufld.config import load_config
from ufld.model import build_model
from ufld.parity import sample_inputs, compare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', default='data/exports/ufld_res18.onnx')
    ap.add_argument('--opset', type=int, default=17)
    ap.add_argument('--n-check', type=int, default=8, help='real test frames used for the parity check')
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = build_model(cfg, pretrained=False, use_aux=False)
    load_model_weights(model, args.ckpt, device='cpu')
    model.eval()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    h, w = cfg.input_size
    # Fixed shape, batch 1: the head is Linear(fea_dim, ...) on a flattened map, so the input
    # size is baked into the weights anyway. dynamo=False uses the stable TorchScript exporter.
    torch.onnx.export(model, (torch.zeros(1, 3, h, w),), str(out), opset_version=args.opset,
                      input_names=['image'], output_names=['logits'], dynamo=False)
    print(f'wrote {out} ({out.stat().st_size / 1e6:.1f} MB)')

    xs = sample_inputs(cfg, args.n_check)
    ref = load_torch(cfg, args.ckpt, torch.device('cpu'))
    compare('onnx-cpu vs torch-cpu', ref, load_onnx(str(out)), xs)


if __name__ == '__main__':
    main()

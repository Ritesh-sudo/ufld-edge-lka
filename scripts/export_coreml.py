"""Convert the inference model to Core ML (ML Program) and check it against PyTorch.

python scripts/export_coreml.py --config configs/tusimple_res18.py --ckpt data/checkpoints/tusimple_18.pth \
    --out data/exports/ufld_res18_fp16.mlpackage --precision fp16

fp16 is what the Neural Engine runs natively; fp32 is exported for the precision comparison.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ufld  # noqa: F401

import argparse

import coremltools as ct
import torch

from ufld.backends import load_coreml, load_torch
from ufld.checkpoint import load_model_weights
from ufld.config import load_config
from ufld.model import build_model
from ufld.parity import compare, sample_inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', default='data/exports/ufld_res18_fp16.mlpackage')
    ap.add_argument('--precision', choices=['fp16', 'fp32'], default='fp16')
    ap.add_argument('--n-check', type=int, default=8)
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = build_model(cfg, pretrained=False, use_aux=False)
    load_model_weights(model, args.ckpt, device='cpu')
    model.eval()

    h, w = cfg.input_size
    traced = torch.jit.trace(model, torch.zeros(1, 3, h, w))
    mlmodel = ct.convert(
        traced, convert_to='mlprogram', inputs=[ct.TensorType(name='image', shape=(1, 3, h, w))],
        outputs=[ct.TensorType(name='logits')], minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT16 if args.precision == 'fp16' else ct.precision.FLOAT32)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out))
    print(f'wrote {out}')

    xs = sample_inputs(cfg, args.n_check)
    ref = load_torch(cfg, args.ckpt, torch.device('cpu'))
    for cu in ('CPU_ONLY', 'ALL'):
        compare(f'coreml-{args.precision}/{cu} vs torch-cpu', ref, load_coreml(str(out), cu), xs)


if __name__ == '__main__':
    main()

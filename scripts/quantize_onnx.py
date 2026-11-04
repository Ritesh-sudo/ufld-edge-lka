"""INT8 static post-training quantization of the ONNX model (ONNX Runtime, QDQ format).

python scripts/quantize_onnx.py --config configs/tusimple_res18.py --onnx data/exports/ufld_res18.onnx \
    --out data/exports/ufld_res18_int8.onnx --n-calib 100

How it works: weights are quantized per output channel (each filter gets its own scale, which
matters for conv nets). Activation ranges are unknown statically, so a calibration set is run
through the fp32 model and each tensor's min/max fixes its int8 scale. Quantize/Dequantize
(QDQ) nodes are inserted around ops, which lets runtimes fuse them into integer kernels.

Calibration uses images only (no labels), taken evenly across the TuSimple *test* list because
that is the only split present locally. This is mild leakage; the Colab train split would be cleaner.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ufld  # noqa: F401

import argparse

import numpy as np
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

from ufld.config import load_config
from ufld.data import LaneTestDataset, _img_transform


class FrameReader(CalibrationDataReader):
    def __init__(self, cfg, n):
        ds = LaneTestDataset(cfg.data_root, f'{cfg.data_root}/test.txt', img_transform=_img_transform(cfg))
        idx = np.linspace(0, len(ds) - 1, n).astype(int)
        self.it = iter([{'image': ds[i][0].unsqueeze(0).numpy().astype(np.float32)} for i in idx])

    def get_next(self):
        return next(self.it, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--onnx', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n-calib', type=int, default=100)
    args = ap.parse_args()

    cfg = load_config(args.config)
    pre = str(Path(args.out).with_suffix('.pre.onnx'))
    quant_pre_process(args.onnx, pre)
    quantize_static(pre, args.out, FrameReader(cfg, args.n_calib), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8, per_channel=True)
    Path(pre).unlink()
    print(f'wrote {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()

"""Save real NMS inputs (and our PyTorch NMS result) from CLRNet on TuSimple test images.

The Colab notebook feeds the same inputs to upstream's CUDA NMS and checks the kept lanes are identical.
python scripts/dump_clrnet_nms_cases.py --n 400 --out tests/data/clrnet_nms_cases.npz
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ufld  # noqa: F401

import argparse

import cv2
import numpy as np
import torch

from third_party.clrnet import CLRConfig, lane_nms, load_clrnet, preprocess_bgr
from third_party.clrnet.decode import nms_inputs
from ufld.config import load_config
from ufld.device import get_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/tusimple_res18.py')
    ap.add_argument('--ckpt', default='data/checkpoints/clrnet_dl/tusimple_r18.pth')
    ap.add_argument('--n', type=int, default=400)
    ap.add_argument('--out', default='tests/data/clrnet_nms_cases.npz')
    args = ap.parse_args()
    ucfg, cfg = load_config(args.config), CLRConfig()
    device = get_device()
    model = load_clrnet(args.ckpt, device, cfg)
    names = [l.split()[0].lstrip('/') for l in open(Path(ucfg.data_root) / 'test.txt')]
    sel = np.linspace(0, len(names) - 1, args.n).astype(int)
    boxes, scores, keeps, offs = [], [], [], [0]
    with torch.no_grad():
        for i in sel:
            x = torch.from_numpy(preprocess_bgr(cv2.imread(str(Path(ucfg.data_root) / names[i])), cfg))
            _, box, sc = nms_inputs(model(x.to(device)).cpu(), cfg)
            if len(box):
                idx, num = lane_nms(box, sc, cfg.nms_thres, cfg.max_lanes)
                keep = idx[:num].numpy()
            else:
                keep = np.zeros(0, dtype=np.int64)
            boxes.append(box.numpy()); scores.append(sc.numpy()); keeps.append(keep); offs.append(offs[-1] + len(box))
    keep_off = np.cumsum([0] + [len(k) for k in keeps])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, boxes=np.concatenate(boxes), scores=np.concatenate(scores), box_off=np.array(offs),
                        keep=np.concatenate(keeps), keep_off=keep_off, names=np.array([names[i] for i in sel]),
                        nms_thres=cfg.nms_thres, top_k=cfg.max_lanes)
    print(f'{args.n} images, {offs[-1]} boxes, {keep_off[-1]} kept -> {args.out}')


if __name__ == '__main__':
    main()

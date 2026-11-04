"""Helpers to check that an exported model reproduces PyTorch's outputs."""
from __future__ import annotations

import os

import numpy as np


def sample_inputs(cfg, n: int) -> list[np.ndarray]:
    """First `n` real test frames, preprocessed exactly like evaluation (1,3,H,W float32).

    Real frames matter: random noise barely exercises the argmax/softmax decision the lanes
    depend on. Falls back to seeded noise if the dataset is not on disk.
    """
    from .data import LaneTestDataset, _img_transform
    list_path = os.path.join(cfg.data_root, 'test.txt')
    if os.path.exists(list_path):
        ds = LaneTestDataset(cfg.data_root, list_path, img_transform=_img_transform(cfg))
        return [ds[i][0].unsqueeze(0).numpy().astype(np.float32) for i in range(min(n, len(ds)))]
    rng = np.random.default_rng(0)
    return [rng.standard_normal((1, 3, *cfg.input_size)).astype(np.float32) for _ in range(n)]


def compare(label: str, ref_fn, test_fn, xs) -> dict:
    """Max/mean abs logit difference, plus how often the per-(row,lane) argmax class agrees."""
    max_abs, mean_abs, agree = 0.0, [], []
    for x in xs:
        a, b = ref_fn(x), test_fn(x)
        d = np.abs(a - b)
        max_abs = max(max_abs, float(d.max()))
        mean_abs.append(float(d.mean()))
        agree.append(float((a.argmax(1) == b.argmax(1)).mean()))
    res = {'max_abs_diff': max_abs, 'mean_abs_diff': float(np.mean(mean_abs)),
           'argmax_agreement': float(np.mean(agree))}
    print(f"[{label}] {len(xs)} frames: max|diff|={res['max_abs_diff']:.2e} "
          f"mean|diff|={res['mean_abs_diff']:.2e} argmax agreement={res['argmax_agreement'] * 100:.3f}%")
    return res

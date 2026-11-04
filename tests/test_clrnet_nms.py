"""The pure-PyTorch lane NMS must reproduce CLRNet's CUDA kernel.

`kernel_oracle` is a line-by-line scalar transliteration of nms_kernel.cu (devIoU + the greedy collect),
float32 arithmetic, no vectorization, so it can be read side by side with the CUDA source. The real CUDA
op is compared on Colab (notebooks/clrnet_nms_check.ipynb) using the saved real inputs in tests/data.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from third_party.clrnet import lane_nms

N_OFFSETS, N_STRIPS = 72, 71
CASES = Path(__file__).parent / 'data' / 'clrnet_nms_cases.npz'


def dev_iou(a, b, thr):
    f = np.float32
    start_a, start_b = int(a[2] * f(N_STRIPS) + f(0.5)), int(b[2] * f(N_STRIPS) + f(0.5))   # (int) truncation
    start = max(start_a, start_b)
    end_a = int(f(start_a) + a[4] - f(1) + f(0.5) - f((a[4] - 1) < 0))
    end_b = int(f(start_b) + b[4] - f(1) + f(0.5) - f((b[4] - 1) < 0))
    end = min(end_a, end_b, N_OFFSETS - 1)
    if end < start:
        return False
    dist = f(0)
    for i in range(5 + start, 5 + end + 1):
        dist += abs(a[i] - b[i])
    return bool(dist < f(thr) * f(end - start + 1))


def kernel_oracle(boxes, scores, thr, top_k):
    boxes = boxes.astype(np.float32)
    idx = np.argsort(-scores, kind='stable')
    n = len(idx)
    removed = np.zeros(n, bool)
    keep = []
    for i in range(n):
        if removed[i]:
            continue
        keep.append(int(idx[i]))
        for j in range(i + 1, n):
            if dev_iou(boxes[idx[i]], boxes[idx[j]], thr):
                removed[j] = True
        if len(keep) == top_k:
            break
    return keep


def run_torch(boxes, scores, thr, top_k):
    keep, num = lane_nms(torch.from_numpy(boxes), torch.from_numpy(scores), thr, top_k)
    return keep[:num].tolist()


def random_lanes(rng, n):
    """Lane-like proposals: start row, length, and straight-ish x(row), with some near-duplicates."""
    b = np.zeros((n, 77), np.float32)
    for i in range(n):
        if i and rng.random() < 0.4:                       # near-duplicate of an earlier lane
            b[i] = b[rng.integers(0, i)]
            b[i, 5:] += rng.normal(0, 8, 72)
            continue
        start = rng.integers(0, 40)
        b[i, 2] = start / N_STRIPS
        b[i, 4] = rng.integers(5, 72 - start)
        b[i, 5:] = rng.uniform(0, 799) + rng.uniform(-6, 6) * np.arange(72) * rng.uniform(0, 1)
    return b


@pytest.mark.parametrize('seed', range(5))
def test_random_cases_match_oracle(seed):
    rng = np.random.default_rng(seed)
    b = random_lanes(rng, 60)
    s = rng.random(60).astype(np.float32)
    for thr in (10, 50, 150):
        assert run_torch(b, s, thr, 5) == kernel_oracle(b, s, thr, 5)


def test_edge_cases_zero_length_and_disjoint_ranges():
    b = np.zeros((3, 77), np.float32)
    b[0, 2], b[0, 4], b[0, 5:] = 0.0, 10, 100.      # rows 0..9
    b[1, 2], b[1, 4], b[1, 5:] = 30 / 71, 10, 100.  # rows 30..39: no shared rows with lane 0 -> never suppressed
    b[2, 2], b[2, 4], b[2, 5:] = 0.0, 0, 100.       # zero length: end < start
    s = np.array([0.9, 0.8, 0.7], np.float32)
    assert run_torch(b, s, 50, 5) == kernel_oracle(b, s, 50, 5)


@pytest.mark.skipif(not CASES.exists(), reason='saved real NMS inputs not present')
def test_real_clrnet_inputs_match_oracle_and_saved_result():
    z = np.load(CASES)
    bo, ko = z['box_off'], z['keep_off']
    thr, top_k = float(z['nms_thres']), int(z['top_k'])
    for i in range(0, len(bo) - 1, 4):                 # every 4th image: oracle is slow pure Python
        b, s = z['boxes'][bo[i]:bo[i + 1]], z['scores'][bo[i]:bo[i + 1]]
        if len(b) == 0:
            continue
        want = kernel_oracle(b, s, thr, top_k)
        assert run_torch(b, s, thr, top_k) == want == z['keep'][ko[i]:ko[i + 1]].tolist()

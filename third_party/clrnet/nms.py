"""Pure PyTorch port of CLRNet's CUDA lane NMS (clrnet/ops/csrc/nms_kernel.cu).

Plain NMS on boxes does not suit lanes, so CLRNet compares two lane proposals directly. Each proposal is
[cls0, cls1, start_y, start_x, length, x_0 ... x_71], with start_y*71 and length in strip units and the x
values in pixels. Two lanes A and B "overlap" if, over the rows both cover, their mean horizontal gap
is under `thresh` pixels. Greedy NMS then keeps the highest-scoring lane and suppresses every lower-scoring
lane that overlaps it, up to `top_k` kept lanes.

Behavior is meant to match the CUDA kernel, including its quirks: rounding via `+0.5` then truncation to
int, the `- ((len-1) < 0)` adjustment for zero-length lanes, an inclusive row range, and the strict `<`
on the summed distance. float32 sums may differ from the kernel's sequential order in the last bits,
which only matters for a pair sitting exactly on the threshold; verified on Colab against the real op.
"""
from __future__ import annotations

import torch

N_OFFSETS = 72
N_STRIPS = N_OFFSETS - 1
PROP_SIZE = 5 + N_OFFSETS


def _overlap_matrix(boxes: torch.Tensor, thresh: float) -> torch.Tensor:
    """(N, N) bool: M[i, j] = devIoU(boxes[i], boxes[j]) from the CUDA kernel."""
    start = torch.trunc(boxes[:, 2] * N_STRIPS + 0.5)                       # (N,) int-like, as `(int)(x + 0.5)`
    length = boxes[:, 4]
    end = torch.trunc(start + length - 1 + 0.5 - (length - 1 < 0).to(boxes.dtype))
    s = torch.maximum(start[:, None], start[None, :])                       # (N, N)
    e = torch.minimum(torch.minimum(end[:, None], end[None, :]), torch.tensor(float(N_OFFSETS - 1)))
    valid = e >= s                                                          # `if (end < start) return false`
    k = torch.arange(N_OFFSETS, device=boxes.device, dtype=boxes.dtype)     # row index 0..71
    in_range = (k[None, None, :] >= s[..., None]) & (k[None, None, :] <= e[..., None])
    xs = boxes[:, 5:]                                                       # (N, 72)
    dist = ((xs[:, None, :] - xs[None, :, :]).abs() * in_range).sum(-1)
    return valid & (dist < thresh * (e - s + 1))


def lane_nms(boxes: torch.Tensor, scores: torch.Tensor, overlap: float, top_k: int):
    """Returns (keep, num_to_keep) like the CUDA op: `keep[:num_to_keep]` are indices into `boxes`, best first."""
    assert boxes.shape[1] == PROP_SIZE, f'expected {PROP_SIZE} columns, got {boxes.shape[1]}'
    n = boxes.shape[0]
    order = torch.sort(scores, descending=True)[1]
    m = _overlap_matrix(boxes[order].float(), overlap).cpu().numpy()        # sorted order on both axes
    removed = [False] * n
    keep: list[int] = []
    for i in range(n):                                                      # greedy, sequential by nature
        if removed[i]:
            continue
        keep.append(int(order[i]))
        for j in range(i + 1, n):                                           # the kernel only looks at later (lower-score) lanes
            if m[i, j]:
                removed[j] = True
        if len(keep) == top_k:
            break
    out = torch.zeros(n, dtype=torch.long)
    out[:len(keep)] = torch.tensor(keep, dtype=torch.long)
    return out.to(boxes.device), len(keep)

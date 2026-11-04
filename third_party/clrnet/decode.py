"""Turn the head output into lanes, then into TuSimple format. Ported from CLRHead.get_lanes / predictions_to_pred and TuSimple.pred2lanes."""
from __future__ import annotations

import numpy as np
import torch
from scipy.interpolate import InterpolatedUnivariateSpline

from .model import CLRConfig
from .nms import lane_nms


def nms_inputs(output: torch.Tensor, cfg: CLRConfig = CLRConfig()):
    """output (B=1, 192, 78) -> (pred, box, scores) for the lanes above the confidence threshold.

    `box` is what the NMS op sees: [cls0, cls1, start_y, start_x, length (strips), x_0..x_71 (pixels)].
    """
    pred = output[0]
    scores = torch.softmax(pred[:, :2], dim=1)[:, 1]
    keep = scores >= cfg.conf_threshold
    pred, scores = pred[keep], scores[keep]
    n_strips = cfg.num_points - 1
    box = torch.cat([pred[:, :4], pred[:, 5:]], dim=-1).detach().clone()
    box[:, 4] = box[:, 4] * n_strips
    box[:, 5:] = box[:, 5:] * (cfg.img_w - 1)
    return pred, box, scores


def decode_lanes(output: torch.Tensor, cfg: CLRConfig = CLRConfig(), nms_fn=lane_nms) -> torch.Tensor:
    """output (B=1, 192, 78) -> kept predictions (K, 78) after confidence filter and lane NMS."""
    pred, box, scores = nms_inputs(output, cfg)
    if pred.shape[0] == 0:
        return pred
    idx, num = nms_fn(box, scores, cfg.nms_thres, cfg.max_lanes)
    pred = pred[idx[:num]].clone()
    pred[:, 5] = torch.round(pred[:, 5] * (cfg.num_points - 1))
    return pred


def lanes_to_tusimple(pred: torch.Tensor, h_samples, cfg: CLRConfig = CLRConfig()) -> list[list[int]]:
    """Kept predictions -> list of lanes, x (raw-frame px) at each h_sample, -2 where absent."""
    n_strips = cfg.num_points - 1
    prior_ys = torch.linspace(1, 0, steps=cfg.num_points, dtype=torch.float64)
    ys_q = np.array(h_samples) / cfg.ori_img_h
    lanes = []
    for lane in pred.detach().cpu():
        xs = lane[6:].clone()
        start = min(max(0, int(round(lane[2].item() * n_strips))), n_strips)
        end = min(start + int(round(lane[5].item())) - 1, len(prior_ys) - 1)
        # If the lane does not start at the bottom of the image, extend until x leaves the image.
        inside = ((xs[:start] >= 0.) & (xs[:start] <= 1.)).numpy()
        mask = ~(inside[::-1].cumprod()[::-1].astype(bool))
        xs[end + 1:] = -2
        sub = xs[:start]
        sub[torch.from_numpy(mask)] = -2
        xs[:start] = sub
        ys = prior_ys[xs >= 0]
        xs = xs[xs >= 0].flip(0).double()
        ys = ys.flip(0)
        ys = (ys * (cfg.ori_img_h - cfg.cut_height) + cfg.cut_height) / cfg.ori_img_h
        if len(xs) <= 1:
            continue
        pts = np.stack([xs.numpy(), ys.numpy()], axis=1)
        spline = InterpolatedUnivariateSpline(pts[:, 1], pts[:, 0], k=min(3, len(pts) - 1))
        x = spline(ys_q)
        x[(ys_q < pts[:, 1].min() - 0.01) | (ys_q > pts[:, 1].max() + 0.01)] = -2.
        out = (x * cfg.ori_img_w).astype(int)
        out[x < 0] = -2
        lanes.append(out.tolist())
    return lanes

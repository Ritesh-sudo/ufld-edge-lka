"""Classic pre-deep-learning lane baseline: Canny edges + probabilistic Hough transform.

Pipeline: grayscale -> Gaussian blur -> Canny edges -> mask to a trapezoid in front of the
car -> HoughLinesP (line segments) -> split segments by slope sign into left/right -> least-squares
fit one straight line per side (weighted by segment length) -> sample x at TuSimple's h_samples.

It can only ever produce straight ego lanes (at most 2), so it cannot represent curves or the
outer lanes. Weak by design: it is the reference point deep models are compared with.
All parameters are in `HoughConfig` (no magic numbers in the code).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

NO_LANE = -2


@dataclass(frozen=True)
class HoughConfig:
    blur_ksize: int = 5
    canny_low: int = 60
    canny_high: int = 150
    roi_top_frac: float = 0.55        # ignore everything above this fraction of image height (sky/horizon)
    roi_top_width_frac: float = 0.25  # trapezoid top width as a fraction of image width
    hough_rho: float = 1.0
    hough_theta: float = np.pi / 180
    hough_threshold: int = 30
    min_line_len: int = 30
    max_line_gap: int = 20
    min_abs_slope: float = 0.4        # dy/dx of a segment; rejects near-horizontal clutter
    max_abs_slope: float = 4.0


def _roi_mask(h: int, w: int, c: HoughConfig) -> np.ndarray:
    top = int(h * c.roi_top_frac)
    half = int(w * c.roi_top_width_frac / 2)
    poly = np.array([[(0, h), (w // 2 - half, top), (w // 2 + half, top), (w, h)]], dtype=np.int32)
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, poly, 255)
    return m


def detect_lanes(frame_bgr: np.ndarray, h_samples, cfg: HoughConfig = HoughConfig()) -> list[list[int]]:
    """Return [left, right] lanes as x per h_sample (raw-frame pixels, -2 where absent).

    A side with no usable segments gives an all -2 lane, which the TuSimple metric simply counts
    as a miss. h_samples are in the same pixel units as `frame_bgr`.
    """
    h, w = frame_bgr.shape[:2]
    g = cv2.GaussianBlur(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), (cfg.blur_ksize, cfg.blur_ksize), 0)
    edges = cv2.bitwise_and(cv2.Canny(g, cfg.canny_low, cfg.canny_high), _roi_mask(h, w, cfg))
    segs = cv2.HoughLinesP(edges, cfg.hough_rho, cfg.hough_theta, cfg.hough_threshold,
                           minLineLength=cfg.min_line_len, maxLineGap=cfg.max_line_gap)
    side = {'left': [], 'right': []}
    for x1, y1, x2, y2 in (segs.reshape(-1, 4) if segs is not None else []):
        if x1 == x2 or y1 == y2:
            continue
        slope = (y2 - y1) / (x2 - x1)            # image y grows downward: left lane has negative slope
        if not cfg.min_abs_slope <= abs(slope) <= cfg.max_abs_slope:
            continue
        # Accept a segment only if it sits on the matching side of the image center.
        if (slope < 0 and max(x1, x2) < w / 2) or (slope > 0 and min(x1, x2) >= w / 2):
            side['left' if slope < 0 else 'right'].append((x1, y1, x2, y2))

    top = h * cfg.roi_top_frac
    lanes = []
    for name in ('left', 'right'):
        s = side[name]
        if not s:
            lanes.append([NO_LANE] * len(h_samples))
            continue
        # Fit x = m*y + b through segment endpoints, weighted by segment length.
        pts = np.array([[x1, y1, x2, y2] for x1, y1, x2, y2 in s], dtype=float)
        wt = np.repeat(np.hypot(pts[:, 2] - pts[:, 0], pts[:, 3] - pts[:, 1]), 2)
        y = pts[:, [1, 3]].ravel()
        x = pts[:, [0, 2]].ravel()
        m, b = np.polyfit(y, x, 1, w=wt)
        lanes.append([int(round(m * yy + b)) if (yy >= top and 0 <= m * yy + b < w) else NO_LANE for yy in h_samples])
    return lanes

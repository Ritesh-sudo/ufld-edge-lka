"""Lateral offset and lane curvature from detected lanes.

Input is what the detector produces: up to 4 lanes, each a list of x positions (raw-frame
pixels) at fixed rows `y_samples`, with -2 meaning "no lane at this row" (TuSimple convention).

There is NO camera calibration for TuSimple, so the metric conversion rests on two stated
assumptions in `CameraModel`: lanes are `lane_width_m` wide, and the camera has focal length
`focal_px`. Everything is flat-road, pinhole-camera approximate. Treat meters as indicative
(good for relative changes and thresholds), not survey-grade.

How the numbers are derived, per image row y where both ego-lane edges are visible:
  w(y)  = x_right - x_left                       lane width in pixels
  Z(y)  = focal_px * lane_width_m / w(y)         distance ahead (pinhole: a 3.7 m object
                                                 spans focal*3.7/Z pixels at distance Z)
  X(y)  = (x_center(y) - cx) * lane_width_m / w(y)   lateral position of the lane center
                                                 relative to the camera axis, in meters
Lateral offset = -X at the nearest row (how far the car is from the lane center).
Curvature: fit X = a Z^2 + b Z + c over the visible rows; curvature at the car is
kappa = 2a / (1 + b^2)^1.5  (1/m; sign: positive = lane bends to the right, ahead).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NO_LANE = -2


@dataclass(frozen=True)
class CameraModel:
    lane_width_m: float = 3.7       # US highway lane width
    focal_px: float = 1000.0        # ~65 deg horizontal FOV on a 1280-wide frame; assumption
    cx: float = 640.0               # vehicle centerline = image center column (camera mounted centered)


@dataclass(frozen=True)
class EgoLanes:
    left: np.ndarray    # x per y_sample (NO_LANE where missing)
    right: np.ndarray
    left_idx: int
    right_idx: int


@dataclass(frozen=True)
class LaneState:
    offset_m: float            # + = vehicle is right of lane center
    curvature_1pm: float       # + = lane bends right ahead; 0 = straight
    lane_width_px: float       # at the nearest usable row
    n_rows: int                # rows used for the fit


def _x_at(lane: np.ndarray, ys: np.ndarray, y_ref: float) -> float | None:
    """Lane x at row y_ref: linear fit to the lane's lowest few valid points (extrapolates a short way)."""
    ok = lane != NO_LANE
    if ok.sum() < 4:
        return None
    y, x = ys[ok], lane[ok].astype(float)
    near = np.argsort(-y)[:8]  # lowest (closest) 8 points
    slope, icpt = np.polyfit(y[near], x[near], 1)
    return float(slope * y_ref + icpt)


def select_ego_lanes(lanes, ys, cam: CameraModel = CameraModel(), y_ref: float | None = None) -> EgoLanes | None:
    """Nearest lane on each side of the vehicle centerline at row `y_ref` (default: bottom sample row).

    Returns None unless a lane exists on BOTH sides. We refuse to guess the missing side:
    a wrong offset is worse than no offset for a controller (it should hand over instead).
    """
    ys = np.asarray(ys, dtype=float)
    y_ref = float(ys.max()) if y_ref is None else y_ref
    best_l = best_r = None
    for i, lane in enumerate(lanes):
        x = _x_at(np.asarray(lane), ys, y_ref)
        if x is None:
            continue
        if x < cam.cx and (best_l is None or x > best_l[0]):
            best_l = (x, i)
        elif x >= cam.cx and (best_r is None or x < best_r[0]):
            best_r = (x, i)
    if best_l is None or best_r is None:
        return None
    return EgoLanes(np.asarray(lanes[best_l[1]]), np.asarray(lanes[best_r[1]]), best_l[1], best_r[1])


def lane_state(ego: EgoLanes, ys, cam: CameraModel = CameraModel(), min_rows: int = 6) -> LaneState | None:
    """Lateral offset and curvature from an ego-lane pair; None if too few rows see both edges."""
    ys = np.asarray(ys, dtype=float)
    both = (ego.left != NO_LANE) & (ego.right != NO_LANE) & (ego.right > ego.left)
    if both.sum() < min_rows:
        return None
    y = ys[both]
    xl, xr = ego.left[both].astype(float), ego.right[both].astype(float)
    w = xr - xl
    m_per_px = cam.lane_width_m / w
    Z = cam.focal_px * m_per_px
    X = ((xl + xr) / 2 - cam.cx) * m_per_px

    # Offset at the nearest row; a short linear fit over the closest rows smooths single-row noise.
    order = np.argsort(Z)
    near = order[:min(8, len(order))]
    b1, c1 = np.polyfit(Z[near], X[near], 1)
    offset = -float(c1)                      # lane center at Z=0 is c1 right of camera => car is -c1 from center

    # Curvature: quadratic in Z (needs a spread of distances to be meaningful)
    if len(Z) >= 3 and np.ptp(Z) > 1.0:
        a, b, _ = np.polyfit(Z, X, 2)
        kappa = float(2 * a / (1 + b * b) ** 1.5)
    else:
        kappa = 0.0
    return LaneState(offset_m=offset, curvature_1pm=kappa, lane_width_px=float(w[order[0]]), n_rows=int(both.sum()))

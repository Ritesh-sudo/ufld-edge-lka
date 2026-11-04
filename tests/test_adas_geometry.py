"""Geometry checked against ground truth built from the same camera model, run forwards.

Synthetic road: flat ground, pinhole camera. A lane center at lateral position Xc(Z) = X0 + k Z^2 / 2
(a circular arc of curvature k, parabola approximation) is projected to pixels; the recovered
offset and curvature must match the inputs we chose.
"""
import numpy as np
import pytest

from adas.geometry import NO_LANE, CameraModel, lane_state, select_ego_lanes

CAM = CameraModel()
YS = np.arange(160, 720, 10)
Y_HORIZON, CAM_H = 140.0, 1.5


def road(offset_m=0.0, kappa=0.0, y_min=300):
    """Return 4 lanes (two ego + two outer) at YS; car is `offset_m` right of lane center."""
    Z = CAM.focal_px * CAM_H / (YS - Y_HORIZON)
    center = -offset_m + kappa * Z ** 2 / 2         # lane center lateral position relative to the camera
    lanes = []
    for shift in (-1.5, -0.5, 0.5, 1.5):            # lane edges in units of lane widths
        x = CAM.cx + CAM.focal_px * (center + shift * CAM.lane_width_m) / Z
        x = np.where(YS >= y_min, x, NO_LANE)
        lanes.append(np.rint(x).astype(int))
    return lanes


def test_centered_straight():
    ego = select_ego_lanes(road(), YS, CAM)
    s = lane_state(ego, YS, CAM)
    assert abs(s.offset_m) < 0.05 and abs(s.curvature_1pm) < 1e-4


@pytest.mark.parametrize('off', [-0.8, -0.3, 0.4, 1.0])
def test_offset_recovered(off):
    s = lane_state(select_ego_lanes(road(offset_m=off), YS, CAM), YS, CAM)
    assert s.offset_m == pytest.approx(off, abs=0.08)


@pytest.mark.parametrize('k', [-1 / 300, -1 / 600, 1 / 600, 1 / 300])
def test_curvature_recovered(k):
    s = lane_state(select_ego_lanes(road(kappa=k), YS, CAM), YS, CAM)
    assert s.curvature_1pm == pytest.approx(k, rel=0.2, abs=5e-4)
    assert np.sign(s.curvature_1pm) == np.sign(k)


def test_ego_selection_picks_inner_lanes():
    ego = select_ego_lanes(road(), YS, CAM)
    assert (ego.left_idx, ego.right_idx) == (1, 2)


def test_missing_side_returns_none():
    lanes = road()
    lanes[2][:] = NO_LANE
    lanes[3][:] = NO_LANE
    assert select_ego_lanes(lanes, YS, CAM) is None


def test_too_few_rows_returns_none():
    ego = select_ego_lanes(road(y_min=690), YS, CAM)  # only 4 rows visible
    assert ego is None or lane_state(ego, YS, CAM) is None

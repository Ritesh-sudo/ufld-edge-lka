import cv2
import numpy as np

from baselines.opencv_hough import NO_LANE, detect_lanes

YS = list(range(160, 720, 10))


def test_finds_two_straight_lanes():
    img = np.zeros((720, 1280, 3), np.uint8)
    cv2.line(img, (300, 720), (580, 400), (255, 255, 255), 8)    # left: x = 300 + (720-y)*0.875
    cv2.line(img, (980, 720), (700, 400), (255, 255, 255), 8)
    left, right = detect_lanes(img, YS)
    for lane, x_at_700 in ((left, 317), (right, 963)):
        got = lane[YS.index(700)]
        assert got != NO_LANE and abs(got - x_at_700) < 12
    assert left[0] == NO_LANE and right[0] == NO_LANE            # above the ROI: no lane reported


def test_blank_image_gives_no_lanes():
    left, right = detect_lanes(np.zeros((720, 1280, 3), np.uint8), YS)
    assert set(left) == set(right) == {NO_LANE}

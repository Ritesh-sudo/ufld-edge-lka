"""CLRNet port: strict checkpoint load and end-to-end decoding on a real TuSimple frame (skipped without the checkpoint/data)."""
from pathlib import Path

import cv2
import pytest
import torch

from third_party.clrnet import CLRConfig, decode_lanes, lanes_to_tusimple, load_clrnet, preprocess_bgr

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / 'data' / 'checkpoints' / 'clrnet_dl' / 'tusimple_r18.pth'
TEST_LIST = ROOT / 'data' / 'tusimple' / 'test.txt'
pytestmark = pytest.mark.skipif(not CKPT.exists(), reason='CLRNet checkpoint not downloaded')
H_SAMPLES = list(range(160, 720, 10))


def test_strict_load_and_output_shape():
    m = load_clrnet(str(CKPT), torch.device('cpu'))
    with torch.no_grad():
        out = m(torch.rand(1, 3, 320, 800))
    assert out.shape == (1, 192, 6 + 72)


@pytest.mark.skipif(not TEST_LIST.exists(), reason='TuSimple test set not present')
def test_real_frame_gives_plausible_lanes():
    cfg = CLRConfig()
    m = load_clrnet(str(CKPT), torch.device('cpu'))
    name = TEST_LIST.read_text().split()[0].lstrip('/')
    img = cv2.imread(str(ROOT / 'data' / 'tusimple' / name))
    with torch.no_grad():
        out = m(torch.from_numpy(preprocess_bgr(img, cfg)))
    lanes = lanes_to_tusimple(decode_lanes(out, cfg), H_SAMPLES, cfg)
    assert 1 <= len(lanes) <= cfg.max_lanes
    for lane in lanes:
        assert len(lane) == len(H_SAMPLES)
        xs = [x for x in lane if x != -2]
        # upstream keeps spline samples slightly past the right edge (max ~1318 on the full test set); ported as is
        assert len(xs) >= 5 and all(0 <= x < 1400 for x in xs)

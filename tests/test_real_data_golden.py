"""Golden tests on real TuSimple frames (data/tusimple, fetched by scripts/fetch_tusimple_kaggle.py).

Skipped unless the real training sample has been fetched and converted.
"""
import random
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from test_data_golden import CFG, ROOT, _convert, _load_script, _make_pair

REAL = ROOT / 'data' / 'tusimple'
pytestmark = pytest.mark.skipif(not (REAL / 'train_gt.txt').exists(), reason='real TuSimple sample not prepared')


@pytest.fixture(scope='module')
def ref_converted(ref, tmp_path_factory):
    """The original converter run on a copy of our real sample (images + annotation json only)."""
    dst = tmp_path_factory.mktemp('real_ref')
    names = [line.split()[0] for line in (REAL / 'train_gt.txt').read_text().splitlines()]
    for n in names:
        (dst / n).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REAL / n, dst / n)
    shutil.copy(REAL / 'label_data_0313.json', dst / 'label_data_0313.json')
    script = _load_script(ref.root / 'scripts' / 'convert_tusimple.py', 'convert_ref_real')
    names, lines = script.get_tusimple_list(str(dst), ['label_data_0313.json'])
    script.generate_segmentation_and_train_list(str(dst), lines, names)
    return dst


def test_converter_matches_reference_on_real_frames(ref_converted):
    assert (REAL / 'train_gt.txt').read_text() == (ref_converted / 'train_gt.txt').read_text()
    pngs = sorted(p.relative_to(ref_converted) for p in ref_converted.rglob('*.png'))
    assert len(pngs) >= 100
    for p in pngs:
        assert (REAL / p).read_bytes() == (ref_converted / p).read_bytes(), p


@pytest.mark.parametrize('use_aux', [False, True])
def test_real_labels_identical_without_augmentation(ref, use_aux):
    ref_ds, our_ds = _make_pair(ref, REAL, use_aux)
    assert len(ref_ds) == len(our_ds) >= 100
    lane_cells = 0
    for i in range(len(ref_ds)):
        want, got = ref_ds[i], our_ds[i]
        np.testing.assert_array_equal(got[1], want[1])
        torch.testing.assert_close(got[0], want[0], rtol=0, atol=0)
        if use_aux:
            torch.testing.assert_close(got[2], want[2], rtol=0, atol=0)
        lane_cells += int((got[1] < CFG.griding_num).sum())
    assert lane_cells > 0  # real lanes are present, not a vacuous pass


def test_real_augmented_samples_identical_given_same_seed(ref):
    ref_ds, our_ds = _make_pair(ref, REAL, True, simu=True)
    for i in range(0, min(len(ref_ds), 60), 3):
        random.seed(i), np.random.seed(i)
        want = ref_ds[i]
        random.seed(i), np.random.seed(i)
        got = our_ds[i]
        np.testing.assert_array_equal(got[1], want[1])
        torch.testing.assert_close(got[0], want[0], rtol=0, atol=0)
        torch.testing.assert_close(got[2], want[2], rtol=0, atol=0)

"""Golden tests on real TuSimple frames: data/tusimple must hold converted data (train_gt.txt).

Locally that is the small sample from scripts/fetch_tusimple_kaggle.py; on Colab it is the full
training set. Either way an evenly spaced sample of UFLD_GOLDEN_N (default 200) images is tested.
Skipped when no converted real data is present. UFLD_REAL_DATA points at a different root.
"""
import json
import os
import random
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from test_data_golden import CFG, ROOT, _load_script, _make_pair

REAL = Path(os.environ.get('UFLD_REAL_DATA', ROOT / 'data' / 'tusimple'))
N = int(os.environ.get('UFLD_GOLDEN_N', '200'))
pytestmark = pytest.mark.skipif(not (REAL / 'train_gt.txt').exists(), reason='real TuSimple data not prepared')


@pytest.fixture(scope='module')
def sample(tmp_path_factory):
    """(list file of the sampled lines, the sampled lines)."""
    lines = (REAL / 'train_gt.txt').read_text().splitlines()
    chosen = lines[::max(1, len(lines) // N)][:N]
    list_file = tmp_path_factory.mktemp('real_sample') / 'train_gt_sample.txt'
    list_file.write_text('\n'.join(chosen) + '\n')
    return list_file, chosen


@pytest.fixture(scope='module')
def ref_converted(ref, sample, tmp_path_factory):
    """The original converter run on a copy of the sampled frames (annotations restricted to them)."""
    _, chosen = sample
    names = [line.split()[0] for line in chosen]
    rows = {}
    for f in sorted(REAL.glob('label_data_*.json')):
        if '.full' not in f.name:
            for line in open(f):
                rows[json.loads(line)['raw_file']] = line.strip()
    dst = tmp_path_factory.mktemp('real_ref')
    for n in names:
        (dst / n).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REAL / n, dst / n)
    (dst / 'label_data_sample.json').write_text('\n'.join(rows[n] for n in names) + '\n')
    script = _load_script(ref.root / 'scripts' / 'convert_tusimple.py', 'convert_ref_real')
    tusimple_names, lines = script.get_tusimple_list(str(dst), ['label_data_sample.json'])
    script.generate_segmentation_and_train_list(str(dst), lines, tusimple_names)
    return dst


def test_converter_matches_reference_on_real_frames(sample, ref_converted):
    _, chosen = sample
    assert (ref_converted / 'train_gt.txt').read_text().splitlines() == chosen
    assert len(chosen) >= 1
    for line in chosen:
        mask = line.split()[1]
        assert (REAL / mask).read_bytes() == (ref_converted / mask).read_bytes(), mask


@pytest.mark.parametrize('use_aux', [False, True])
def test_real_labels_identical_without_augmentation(ref, sample, use_aux):
    ref_ds, our_ds = _make_pair(ref, REAL, use_aux, list_file=sample[0])
    assert len(ref_ds) == len(our_ds) >= 1
    lane_cells = 0
    for i in range(len(ref_ds)):
        want, got = ref_ds[i], our_ds[i]
        np.testing.assert_array_equal(got[1], want[1])
        torch.testing.assert_close(got[0], want[0], rtol=0, atol=0)
        if use_aux:
            torch.testing.assert_close(got[2], want[2], rtol=0, atol=0)
        lane_cells += int((got[1] < CFG.griding_num).sum())
    assert lane_cells > 0  # real lanes are present, not a vacuous pass


def test_real_augmented_samples_identical_given_same_seed(ref, sample):
    ref_ds, our_ds = _make_pair(ref, REAL, True, simu=True, list_file=sample[0])
    for i in range(0, min(len(ref_ds), 60), 3):
        random.seed(i), np.random.seed(i)
        want = ref_ds[i]
        random.seed(i), np.random.seed(i)
        got = our_ds[i]
        np.testing.assert_array_equal(got[1], want[1])
        torch.testing.assert_close(got[0], want[0], rtol=0, atol=0)
        torch.testing.assert_close(got[2], want[2], rtol=0, atol=0)

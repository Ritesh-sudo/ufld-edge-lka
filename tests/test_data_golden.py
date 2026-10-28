"""Golden tests: the ported data pipeline must produce identical labels to the original."""
import importlib.util
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torchvision.transforms as T
from PIL import Image

import ufld.transforms as mt
from synthetic import make_tusimple
from ufld.config import load_config
from ufld.data import LaneClsDataset

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / 'configs' / 'tusimple_res18.py')


def _load_script(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _convert(script, root):
    """Run a convert_tusimple.py (original or port) over `root`, as its __main__ does."""
    names, line_txt = script.get_tusimple_list(root, ['label_data_0601.json', 'label_data_0531.json', 'label_data_0313.json'])
    script.generate_segmentation_and_train_list(root, line_txt, names)


@pytest.fixture(scope='module')
def datasets(ref, tmp_path_factory):
    """The same synthetic raw data converted by the original script and by ours."""
    raw = make_tusimple(tmp_path_factory.mktemp('raw'))
    out = {}
    for tag, script_path in [('ref', ref.root / 'scripts' / 'convert_tusimple.py'),
                             ('ours', ROOT / 'scripts' / 'convert_tusimple.py')]:
        dst = tmp_path_factory.mktemp(tag)
        shutil.copytree(raw, dst, dirs_exist_ok=True)
        _convert(_load_script(script_path, f'convert_{tag}'), str(dst))
        out[tag] = dst
    return out


def test_converter_outputs_identical(datasets):
    ref_dir, our_dir = datasets['ref'], datasets['ours']
    assert (ref_dir / 'train_gt.txt').read_text() == (our_dir / 'train_gt.txt').read_text()
    pngs = sorted(p.relative_to(ref_dir) for p in ref_dir.rglob('*.png'))
    assert len(pngs) >= 4
    for p in pngs:
        assert (ref_dir / p).read_bytes() == (our_dir / p).read_bytes()


def test_row_anchors_in_config_match_reference(ref):
    assert CFG.row_anchor == ref.constant.tusimple_row_anchor


def _make_pair(ref, root, use_aux, simu=False, list_file=None):
    list_file = str(list_file or root / 'train_gt.txt')
    imgt = T.Compose([T.Resize((288, 800)), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    segt = T.Compose([ref.mytransforms.FreeScaleMask((36, 100)), ref.mytransforms.MaskToTensor()])
    ref_ds = ref.dataset.LaneClsDataset(
        str(root), list_file, img_transform=imgt, simu_transform=(
            ref.mytransforms.Compose2([ref.mytransforms.RandomRotate(6), ref.mytransforms.RandomUDoffsetLABEL(100),
                                       ref.mytransforms.RandomLROffsetLABEL(200)]) if simu else None),
        griding_num=100, row_anchor=list(CFG.row_anchor), segment_transform=segt, use_aux=use_aux, num_lanes=4)
    our_ds = LaneClsDataset(
        str(root), list_file, row_anchor=CFG.row_anchor, griding_num=100, num_lanes=4, use_aux=use_aux,
        img_transform=imgt, segment_transform=T.Compose([mt.FreeScaleMask((36, 100)), mt.MaskToTensor()]),
        simu_transform=(mt.Compose2([mt.RandomRotate(6), mt.RandomUDoffsetLABEL(100), mt.RandomLROffsetLABEL(200)])
                        if simu else None))
    return ref_ds, our_ds


@pytest.mark.parametrize('use_aux', [False, True])
def test_labels_identical_without_augmentation(ref, datasets, use_aux):
    ref_ds, our_ds = _make_pair(ref, datasets['ref'], use_aux)
    assert len(ref_ds) == len(our_ds) >= 4
    for i in range(len(ref_ds)):
        want, got = ref_ds[i], our_ds[i]
        assert len(want) == len(got)
        np.testing.assert_array_equal(got[1], want[1])           # row-anchor class labels
        assert got[1].shape == (56, 4)
        torch.testing.assert_close(got[0], want[0], rtol=0, atol=0)  # image tensor
        if use_aux:
            torch.testing.assert_close(got[2], want[2], rtol=0, atol=0)  # segmentation mask


def test_labels_are_nontrivial(ref, datasets):
    # guards against a vacuous pass: real lanes, "no lane" cells, and varied cells all occur
    _, our_ds = _make_pair(ref, datasets['ref'], False)
    labels = np.stack([our_ds[i][1] for i in range(len(our_ds))])
    assert (labels == 100).any() and (labels < 100).any() and len(np.unique(labels)) > 20


def test_augmented_samples_identical_given_same_seed(ref, datasets):
    ref_ds, our_ds = _make_pair(ref, datasets['ref'], True, simu=True)
    for i in range(len(ref_ds)):
        random.seed(i), np.random.seed(i)
        want = ref_ds[i]
        random.seed(i), np.random.seed(i)
        got = our_ds[i]
        np.testing.assert_array_equal(got[1], want[1])
        torch.testing.assert_close(got[0], want[0], rtol=0, atol=0)
        torch.testing.assert_close(got[2], want[2], rtol=0, atol=0)


def _write_288_dataset(root):
    """One 288x800 frame with a vertical lane per id at known columns."""
    (root / 'clips').mkdir(parents=True)
    Image.fromarray(np.zeros((288, 800, 3), np.uint8)).save(root / 'clips/a.jpg')
    mask = np.zeros((288, 800), np.uint8)
    for lane_id, col in enumerate([100, 300, 500, 700], start=1):
        mask[:, col - 2:col + 3] = lane_id
    Image.fromarray(mask).save(root / 'clips/a.png')
    (root / 'train_gt.txt').write_text('clips/a.jpg clips/a.png 1 1 1 1\n')


def test_sample_tmp_branch_only_matters_when_mask_height_is_288(ref, tmp_path):
    # Real TuSimple masks are 720 high, so `if h != 288` always runs and the fix is a no-op
    # (test_labels_identical_without_augmentation above). The original only breaks at h == 288:
    _write_288_dataset(tmp_path)
    ref_ds, our_ds = _make_pair(ref, tmp_path, False)
    with pytest.raises(UnboundLocalError):
        ref_ds[0]
    labels = our_ds[0][1]  # (56, 4): anchors used as-is, so each lane stays in its own cell
    cell = 799 / 99
    expected = np.array([int(c // cell) for c in [100, 300, 500, 700]])
    np.testing.assert_array_equal(labels, np.tile(expected, (56, 1)))

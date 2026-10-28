from pathlib import Path

import pytest

from ufld.config import dump_config, load_config

CONFIGS = Path(__file__).resolve().parents[1] / 'configs'


def test_res18_values():
    cfg = load_config(CONFIGS / 'tusimple_res18.py')
    assert cfg.input_size == (288, 800)
    assert cfg.griding_num == 100 and cfg.num_lanes == 4
    assert cfg.cls_num_per_lane == len(cfg.row_anchor) == 56
    assert cfg.row_anchor[0] == 64 and cfg.row_anchor[-1] == 284
    assert cfg.shp_loss_w == 0.0 and cfg.sim_loss_w == 1.0


def test_res34_inherits_and_overrides():
    cfg = load_config(CONFIGS / 'tusimple_res34.py')
    assert cfg.backbone == '34' and cfg.griding_num == 100


def test_overrides_and_unknown_key():
    assert load_config(CONFIGS / 'tusimple_res18.py', {'batch_size': 4}).batch_size == 4
    with pytest.raises(KeyError):
        load_config(CONFIGS / 'tusimple_res18.py', {'btach_size': 4})


def test_invalid_values_rejected():
    with pytest.raises(AssertionError):
        load_config(CONFIGS / 'tusimple_res18.py', {'input_size': (288, 801)})
    with pytest.raises(AssertionError):
        load_config(CONFIGS / 'tusimple_res18.py', {'backbone': '50'})


def test_dump_roundtrip(tmp_path):
    import json
    path = dump_config(load_config(CONFIGS / 'tusimple_res18.py'), tmp_path)
    assert json.loads(path.read_text())['row_anchor'][:2] == [64, 68]

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('arrange_tusimple', ROOT / 'scripts' / 'arrange_tusimple.py')
arrange_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(arrange_mod)


def _frame(root, raw):
    p = root / raw
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b'jpg')
    return {'raw_file': raw, 'h_samples': [160], 'lanes': [[100]]}


def _kaggle_like(tmp_path):
    """train_set/ and test_set/ each with clips/, sharing the date folders 0531 and 0601."""
    src = tmp_path / 'raw' / 'TUSimple'
    train = [_frame(src / 'train_set', f'clips/{d}/{c}/20.jpg')
             for d, c in [('0313-1', '100'), ('0531', '200'), ('0601', '300')]]
    test = [_frame(src / 'test_set', f'clips/{d}/{c}/20.jpg')
            for d, c in [('0530', '400'), ('0531', '500'), ('0601', '600')]]
    (src / 'train_set' / 'label_data_0313.json').write_text('\n'.join(map(json.dumps, train)))
    (src / 'test_label.json').write_text('\n'.join(map(json.dumps, test)))
    (src / 'seg_label').mkdir()
    (src / 'seg_label' / 'label_data_ignored.json').write_text('{}')   # must not be picked up
    return src.parent


def test_merges_train_and_test_clips(tmp_path):
    src, dst = _kaggle_like(tmp_path), tmp_path / 'out'
    result = arrange_mod.arrange(src, dst)
    assert result['clips_moved'] == 6
    assert result['annotations'] == {'label_data_0313.json': 3, 'test_label.json': 3}
    for raw in ['clips/0531/200/20.jpg', 'clips/0531/500/20.jpg', 'clips/0601/300/20.jpg', 'clips/0601/600/20.jpg']:
        assert (dst / raw).is_file()          # overlapping date folders hold clips from both splits
    assert not (dst / 'label_data_ignored.json').exists()


def test_fails_loudly_when_an_annotated_image_is_missing(tmp_path):
    src, dst = _kaggle_like(tmp_path), tmp_path / 'out'
    (src / 'TUSimple' / 'test_set' / 'clips' / '0530' / '400' / '20.jpg').unlink()
    with pytest.raises(FileNotFoundError, match='missing'):
        arrange_mod.arrange(src, dst)


def test_copy_keeps_the_source(tmp_path):
    src, dst = _kaggle_like(tmp_path), tmp_path / 'out'
    arrange_mod.arrange(src, dst, copy=True)
    assert (src / 'TUSimple' / 'test_set' / 'clips' / '0530' / '400' / '20.jpg').is_file()

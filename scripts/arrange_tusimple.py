"""Arrange an extracted TuSimple download into the layout convert_tusimple.py expects.

The Kaggle mirror splits the data into train_set/ and test_set/, each with its own clips/. The
original tooling wants one root:

    <dst>/clips/<date>/<clip>/<frame>.jpg   (train and test clips merged)
    <dst>/label_data_*.json, test_tasks_0627.json (if present), test_label.json

Clip folders are moved (instant on the same disk; --copy to keep the source). Date folders such
as 0531 and 0601 exist in both splits, so merging is done per clip. Afterwards every image named
by an annotation file is checked, and the script fails if any is missing.

  python scripts/arrange_tusimple.py --src /content/tusimple_raw --dst /content/tusimple
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ANNOTATIONS = ('label_data_*.json', 'test_tasks_*.json', 'test_label.json')


def _clip_dirs(src: Path):
    for clips in sorted(p for p in src.rglob('clips') if p.is_dir() and not p.is_symlink()):
        for date in sorted(d for d in clips.iterdir() if d.is_dir() and not d.is_symlink()):
            for clip in sorted(c for c in date.iterdir() if c.is_dir() and not c.is_symlink()):
                yield date.name, clip


def arrange(src: Path, dst: Path, copy: bool = False) -> dict:
    src, dst = Path(src), Path(dst)
    moved = 0
    for date, clip in _clip_dirs(src):
        target = dst / 'clips' / date / clip.name
        if target.exists():
            raise FileExistsError(f'{target} already exists (same clip in two places?)')
        target.parent.mkdir(parents=True, exist_ok=True)
        if copy:
            shutil.copytree(clip, target)
        else:
            shutil.move(str(clip), str(target))
        moved += 1

    found = {}
    for pattern in ANNOTATIONS:
        for f in sorted(src.rglob(pattern)):
            if f.is_file() and not f.is_symlink() and 'seg_label' not in f.parts:
                found.setdefault(f.name, f)
    dst.mkdir(parents=True, exist_ok=True)
    for name, f in found.items():
        shutil.copy2(f, dst / name)

    # Every image an annotation refers to must exist.
    counts, missing = {}, []
    for name in found:
        rows = [json.loads(line) for line in open(dst / name) if line.strip()]
        counts[name] = len(rows)
        missing += [r['raw_file'] for r in rows if not (dst / r['raw_file']).is_file()]
    if missing:
        raise FileNotFoundError(f'{len(missing)} annotated images are missing, e.g. {missing[:3]}')
    if not any(n.startswith('label_data_') for n in found) or 'test_label.json' not in found:
        raise FileNotFoundError(f'need label_data_*.json and test_label.json; found {sorted(found)}')
    return {'clips_moved': moved, 'annotations': counts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='extracted download')
    ap.add_argument('--dst', required=True, help='merged output root')
    ap.add_argument('--copy', action='store_true', help='copy instead of move')
    args = ap.parse_args()
    try:
        print(arrange(Path(args.src), Path(args.dst), args.copy))
    except (FileNotFoundError, FileExistsError) as e:
        sys.exit(f'error: {e}')


if __name__ == '__main__':
    main()

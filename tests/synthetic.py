"""Builds a tiny TuSimple-shaped dataset (raw JSON + frames) for tests."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

H_SAMPLES = list(range(160, 720, 10))  # 56 rows, as in the real benchmark


def _lane(x_bottom, direction, top_cut=None, bottom_cut=None):
    """x at each h_sample for a straight lane; -2 where not annotated."""
    xs = [x_bottom + direction * 0.3 * (710 - y) for y in H_SAMPLES]
    xs = [int(round(x)) for x in xs]
    for i, y in enumerate(H_SAMPLES):
        if (top_cut is not None and y < top_cut) or (bottom_cut is not None and y > bottom_cut):
            xs[i] = -2
    return xs


# lanes per frame: (x_bottom, +1 = leaning right-to-left i.e. a left lane, -1 = right lane, cuts)
FRAMES = {
    'four': [(300, +1), (500, +1), (800, -1), (1000, -1)],
    'two': [(450, +1), (850, -1)],
    'five': [(150, +1), (300, +1), (500, +1), (800, -1), (1000, -1)],
    'partial': [(300, +1, 300, None), (500, +1, None, 500), (800, -1), (1000, -1, 250, None)],
}


def make_tusimple(root, seed=0):
    root = Path(root)
    rng = np.random.default_rng(seed)
    entries = []
    for n, (name, lanes) in enumerate(FRAMES.items()):
        raw = f'clips/0601/{n}/20.jpg'
        (root / raw).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8)).save(root / raw, quality=95)
        entries.append({'raw_file': raw, 'h_samples': H_SAMPLES,
                        'lanes': [_lane(l[0], l[1], *(l[2:] + (None,) * (4 - len(l)))[:2]) for l in lanes]})
    split = len(entries) // 2
    for fname, chunk in [('label_data_0313.json', entries[:split]), ('label_data_0531.json', entries[split:]),
                         ('label_data_0601.json', entries[:1])]:
        (root / fname).write_text('\n'.join(json.dumps(e) for e in chunk))
    test = [{'raw_file': e['raw_file'], 'h_samples': H_SAMPLES, 'lanes': e['lanes'], 'run_time': 10}
            for e in entries[:2]]
    (root / 'test_tasks_0627.json').write_text('\n'.join(json.dumps({'raw_file': t['raw_file']}) for t in test))
    (root / 'test_label.json').write_text('\n'.join(json.dumps(t) for t in test))
    return root

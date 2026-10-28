"""Fetch the parts of TuSimple we need from the Kaggle mirror manideep1108/tusimple.

The official TuSimple download links are dead. Kaggle serves each file as a one-member zip,
so every download is unwrapped and written to a path chosen here (never the zip member's name).
Downloads are resumable (existing files are skipped), rate-limited, retried with backoff, and
failures are logged to <root>/fetch_failures.log.

  python scripts/fetch_tusimple_kaggle.py meta            # label json files
  python scripts/fetch_tusimple_kaggle.py test            # frame 20 of every test clip
  python scripts/fetch_tusimple_kaggle.py train --n 200   # frame 20 of N evenly spaced train clips

Needs Kaggle credentials (~/.kaggle/kaggle.json or env vars); see the kaggle package docs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import io
import json
import random
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

DATASET = 'manideep1108/tusimple'
PREFIX = 'TUSimple'
MAX_FAIL_STREAK = 5
MAX_BYTES = 64 * 1024 * 1024  # no single TuSimple file is anywhere near this

class RateLimited(RuntimeError):
    pass


_lock = threading.Lock()
_next_slot = 0.0


def _throttle(per_second: float) -> None:
    """Global rate limit shared by all worker threads."""
    global _next_slot
    with _lock:
        now = time.monotonic()
        wait = max(0.0, _next_slot - now)
        _next_slot = max(now, _next_slot) + 1.0 / per_second
    if wait:
        time.sleep(wait)


def fetch_one(api, remote: str, dest: Path, per_second: float, retries: int = 5) -> str:
    """Download TUSimple/<remote> to `dest`. Returns 'ok', 'skipped', or raises after retries."""
    if dest.exists() and dest.stat().st_size > 0:
        return 'skipped'
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            _throttle(per_second)
            with tempfile.TemporaryDirectory() as tmp:
                api.dataset_download_file(DATASET, f'{PREFIX}/{remote}', path=tmp, force=True, quiet=True)
                zips = list(Path(tmp).rglob('*.zip'))
                if len(zips) != 1:
                    # the client can return without writing anything when rate-limited
                    raise RateLimited(f'expected one zip, got {zips}')
                with zipfile.ZipFile(zips[0]) as z:
                    members = z.infolist()
                    if len(members) != 1 or members[0].file_size > MAX_BYTES:
                        raise RuntimeError(f'unexpected zip contents: {[m.filename for m in members]}')
                    data = z.read(members[0])
            part = dest.with_name(dest.name + '.part')
            part.write_bytes(data)
            part.replace(dest)
            return 'ok'
        except Exception as e:  # network errors, 429/5xx, bad zip
            if attempt == retries - 1:
                raise
            limited = isinstance(e, RateLimited) or '429' in str(e)
            # rate limits need minutes to clear; other errors are usually transient
            delay = min(300, 30 * 2 ** attempt) if limited else min(60, 2 ** attempt * 2)
            time.sleep(delay + random.random() * delay * 0.1)  # backoff with jitter
    raise AssertionError('unreachable')


def fetch_many(api, items: list[tuple[str, Path]], root: Path, workers: int, per_second: float) -> dict:
    counts = {'ok': 0, 'skipped': 0, 'failed': 0}
    failures = root / 'fetch_failures.log'
    start = time.monotonic()
    streak = 0
    with ThreadPoolExecutor(workers) as pool:
        futs = {pool.submit(fetch_one, api, r, d, per_second): r for r, d in items}
        for n, fut in enumerate(as_completed(futs), 1):
            try:
                result = fut.result()
                counts[result] += 1
                streak = 0
            except Exception as e:
                counts['failed'] += 1
                streak += 1
                if streak >= MAX_FAIL_STREAK:  # fail fast instead of hammering a blocked API
                    for f in futs:
                        f.cancel()
                    print(f'{streak} failures in a row, stopping; rerun later to resume. {counts}', flush=True)
                    break
                with open(failures, 'a') as f:
                    f.write(f'{futs[fut]}\t{type(e).__name__}: {e}\n')
            if n % 100 == 0 or n == len(items):
                print(f'{n}/{len(items)} {counts} {time.monotonic() - start:.0f}s', flush=True)
    return counts


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['meta', 'test', 'train'])
    ap.add_argument('--root', default='data/tusimple')
    ap.add_argument('--n', type=int, default=200, help='train clips to fetch')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--rate', type=float, default=0.5, help='max requests per second')
    args = ap.parse_args()
    root = Path(args.root)

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    if args.stage == 'meta':
        # Optional files: report which exist rather than failing the whole stage.
        for remote, local in [('test_label.json', 'test_label.json'),
                              ('test_set/test_tasks_0627.json', 'test_tasks_0627.json'),
                              ('train_set/label_data_0313.json', 'label_data_0313.full.json')]:
            try:
                print(remote, '->', fetch_one(api, remote, root / local, args.rate))
            except Exception as e:
                print(remote, '-> NOT AVAILABLE:', type(e).__name__, str(e)[:120])
    elif args.stage == 'test':
        names = [r['raw_file'] for r in _rows(root / 'test_label.json')]
        items = [(f'test_set/{n}', root / n) for n in names]
        print(fetch_many(api, items, root, args.workers, args.rate))
    else:
        rows = _rows(root / 'label_data_0313.full.json')
        step = max(1, len(rows) // args.n)
        picked = rows[::step][:args.n]  # evenly spaced, for variety across clips
        (root / 'label_data_0313.json').write_text('\n'.join(json.dumps(r) for r in picked) + '\n')
        items = [(f'train_set/{r["raw_file"]}', root / r['raw_file']) for r in picked]
        print(fetch_many(api, items, root, args.workers, args.rate))


if __name__ == '__main__':
    main()

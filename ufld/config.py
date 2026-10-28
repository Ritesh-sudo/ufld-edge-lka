"""Python-file configs.

A config is a plain .py file of module-level variables (configs/tusimple_res18.py).
`load_config` executes it and returns a read-only namespace. A config may set
`base = 'other.py'` (path relative to itself) to inherit and override another config.

changed: replaces the original utils/config.py + addict + argparse merging, which pulled in
extra dependencies and mixed CLI parsing with config loading.
"""
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

BACKBONES = ('18', '34')
OPTIMIZERS = ('Adam', 'SGD')
SCHEDULERS = ('cos', 'multi')


def _exec_config(path: Path) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location(f'_cfg_{path.stem}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values = {k: v for k, v in vars(module).items() if not k.startswith('_')}
    base = values.pop('base', None)
    if base is not None:
        merged = _exec_config(path.parent / base)
        merged.update(values)
        return merged
    return values


def _validate(cfg: dict[str, Any]) -> None:
    h, w = cfg['input_size']
    assert h % 32 == 0 and w % 32 == 0, f'input_size {cfg["input_size"]} must be divisible by 32'
    anchors = cfg['row_anchor']
    assert anchors == sorted(anchors) and 0 <= anchors[0] and anchors[-1] < h, \
        'row_anchor must be sorted and lie inside the input height'
    assert cfg['backbone'] in BACKBONES, f'backbone must be one of {BACKBONES}'
    assert cfg['optimizer'] in OPTIMIZERS and cfg['scheduler'] in SCHEDULERS
    assert cfg['num_lanes'] > 0 and cfg['griding_num'] > 1


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> SimpleNamespace:
    """Load a config file, apply `overrides` (unknown keys are an error), validate, derive."""
    cfg = _exec_config(Path(path).resolve())
    for key, value in (overrides or {}).items():
        if key not in cfg:
            raise KeyError(f'unknown config key: {key!r}')
        cfg[key] = value
    cfg['row_anchor'] = list(cfg['row_anchor'])
    cfg['input_size'] = tuple(cfg['input_size'])
    _validate(cfg)
    cfg['cls_num_per_lane'] = len(cfg['row_anchor'])  # derived: one classification per row anchor
    return SimpleNamespace(**cfg)


def dump_config(cfg: SimpleNamespace, out_dir: str | Path) -> Path:
    """Write the resolved config next to a run's outputs, for reproducibility."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'config.json'
    path.write_text(json.dumps(vars(cfg), indent=2, default=list))
    return path


def parse_overrides(pairs: list[str]) -> dict[str, Any]:
    """["batch_size=4", "optimizer='SGD'"] -> {"batch_size": 4, "optimizer": "SGD"} (values are python literals)."""
    out = {}
    for pair in pairs:
        key, _, raw = pair.partition('=')
        try:
            out[key] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            out[key] = raw  # bare string, e.g. optimizer=SGD
    return out

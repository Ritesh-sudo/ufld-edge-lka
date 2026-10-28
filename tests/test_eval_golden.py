"""Golden tests: ported TuSimple scoring/decoding/schedulers must equal the original."""
import json

import numpy as np
import pytest
import torch

from ufld.evaluation import LaneEval, generate_tusimple_lines
from ufld.scheduler import CosineAnnealingLR, MultiStepLR

H = list(range(160, 720, 10))


def _lane(rng, x0, slope, dropout=0.0):
    xs = [int(x0 + slope * (710 - y) + rng.normal(0, 6)) for y in H]
    return [-2 if (rng.random() < dropout or x < 0 or x > 1279) else x for x in xs]


def _write_case(tmp_path, seed):
    """A prediction/ground-truth pair with perturbed, missing and extra lanes."""
    rng = np.random.default_rng(seed)
    gts, preds = [], []
    for n in range(40):
        gt_lanes = [_lane(rng, x0, s) for x0, s in [(300, .3), (500, .3), (800, -.3), (1000, -.3)][:rng.integers(2, 5)]]
        if n % 9 == 0:  # >4 gt lanes exercises the "drop the worst" rule
            gt_lanes.append(_lane(rng, 150, .3))
        pred_lanes = [[x + int(rng.normal(0, 8)) if x >= 0 else x for x in lane] for lane in gt_lanes]
        if n % 4 == 0:
            pred_lanes = pred_lanes[1:]                      # missed lane (FN)
        if n % 5 == 0:
            pred_lanes.append(_lane(rng, 640, .01, 0.3))     # spurious lane (FP)
        if n % 11 == 0:
            pred_lanes = pred_lanes * 2                      # too many predictions
        name = f'clips/c/{n}/20.jpg'
        gts.append({'raw_file': name, 'h_samples': H, 'lanes': gt_lanes})
        preds.append({'raw_file': name, 'h_samples': H, 'lanes': pred_lanes, 'run_time': 10})
    pf, gf = tmp_path / f'pred{seed}.json', tmp_path / f'gt{seed}.json'
    pf.write_text('\n'.join(map(json.dumps, preds)))
    gf.write_text('\n'.join(map(json.dumps, gts)))
    return pf, gf


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_metric_matches_reference(ref, tmp_path, seed):
    pf, gf = _write_case(tmp_path, seed)
    want = {r['name']: r['value'] for r in json.loads(ref.lane.LaneEval.bench_one_submit(str(pf), str(gf)))}
    got = LaneEval.bench_one_submit(pf, gf)
    assert 0 < got['Accuracy'] < 1 and got['FP'] > 0 and got['FN'] > 0  # non-vacuous
    for k in ('Accuracy', 'FP', 'FN'):
        assert got[k] == pytest.approx(want[k], abs=1e-9)


def test_line_decoding_matches_reference(ref):
    g = torch.Generator().manual_seed(0)
    out = torch.randn(101, 56, 4, generator=g) * 4
    out[100, ::3, 0] = 50          # force some "no lane" rows
    want = ref.eval_wrapper.generate_tusimple_lines(out, (288, 800), 100)
    got = generate_tusimple_lines(out, 100, 1280)
    assert got == want
    assert any(x == -2 for lane in got for x in lane) and any(x > 0 for lane in got for x in lane)


@pytest.mark.parametrize('kind', ['cos', 'multi'])
def test_scheduler_matches_reference(ref, kind):
    def run(sched_cls, **kw):
        opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
        sched = sched_cls(opt, **kw)
        lrs = []
        for it in range(300):
            lrs.append(opt.param_groups[0]['lr'])
            sched.step(it)
        return lrs

    if kind == 'cos':
        kw = dict(T_max=300, eta_min=0, warmup='linear', warmup_iters=20)
        assert run(CosineAnnealingLR, **kw) == run(ref.factory.CosineAnnealingLR, **kw)
    else:
        kw = dict(steps=[3, 6], gamma=0.1, iters_per_epoch=30, warmup='linear', warmup_iters=20)
        lrs = run(MultiStepLR, **kw)
        assert lrs == run(ref.factory.MultiStepLR, **kw) and lrs[-1] < lrs[40]

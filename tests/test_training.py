"""End-to-end checks of the training loop on a tiny synthetic TuSimple set."""
import importlib.util
from pathlib import Path

import pytest
import torch

import ufld.config as config
from synthetic import make_tusimple
from ufld.checkpoint import load_model_weights, save_checkpoint
from ufld.config import load_config, parse_overrides
from ufld.data import build_train_dataset
from ufld.device import get_device
from ufld.engine import train_one_epoch
from ufld.losses import LaneLoss
from ufld.model import build_model
from ufld.scheduler import get_optimizer, get_scheduler
from ufld.utils import set_seed

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def cfg(tmp_path_factory):
    root = make_tusimple(tmp_path_factory.mktemp('tusimple'))
    spec = importlib.util.spec_from_file_location('convert', ROOT / 'scripts' / 'convert_tusimple.py')
    conv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conv)
    names, lines = conv.get_tusimple_list(str(root), ['label_data_0601.json', 'label_data_0531.json', 'label_data_0313.json'])
    conv.generate_segmentation_and_train_list(str(root), lines, names)
    return load_config(ROOT / 'configs' / 'tusimple_res18.py',
                       {'data_root': str(root), 'num_workers': 0, 'batch_size': 4, 'learning_rate': 1e-3,
                        'warmup': None, 'epoch': 1})


def _loader(cfg):
    ds = build_train_dataset(cfg)
    ds.simu_transform = None  # no augmentation: the same 4 samples every step, so loss must fall
    return torch.utils.data.DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)


def test_loss_decreases_on_device(cfg):
    set_seed(0)
    device = get_device()
    model = build_model(cfg, pretrained=False).to(device)   # no download in tests
    opt = get_optimizer(model, cfg)
    loader = _loader(cfg)
    sched = get_scheduler(opt, cfg, len(loader) * 12)
    loss_fn = LaneLoss(cfg)
    losses = [train_one_epoch(model, loader, loss_fn, opt, sched, device, cfg, epoch=e) for e in range(12)]
    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] < 0.5 * losses[0], losses


def test_checkpoint_roundtrip_and_strictness(cfg, tmp_path):
    model = build_model(cfg, pretrained=False)
    opt = get_optimizer(model, cfg)
    save_checkpoint(model, opt, 3, tmp_path / 'c.pth')
    # the training checkpoint (with aux weights) loads into an inference model: aux_* is tolerated
    infer = build_model(cfg, pretrained=False, use_aux=False)
    ckpt = load_model_weights(infer, tmp_path / 'c.pth')
    assert ckpt['epoch'] == 3
    for k, v in infer.state_dict().items():
        assert torch.equal(v, model.state_dict()[k])
    # ...but a real mismatch is an error, not silently ignored
    bad = {'model': {k: v for k, v in infer.state_dict().items() if not k.startswith('cls.')}}
    torch.save(bad, tmp_path / 'bad.pth')
    with pytest.raises(RuntimeError, match='missing'):
        load_model_weights(infer, tmp_path / 'bad.pth')


def test_module_prefix_is_stripped(cfg, tmp_path):
    model = build_model(cfg, pretrained=False, use_aux=False)
    torch.save({'model': {'module.' + k: v for k, v in model.state_dict().items()}}, tmp_path / 'ddp.pth')
    load_model_weights(build_model(cfg, pretrained=False, use_aux=False), tmp_path / 'ddp.pth')


def test_parse_overrides():
    assert parse_overrides(['batch_size=4', 'optimizer=SGD', "backbone='34'", 'lr=1e-3']) == {
        'batch_size': 4, 'optimizer': 'SGD', 'backbone': '34', 'lr': 1e-3}


def test_checkpoint_write_is_atomic(cfg, tmp_path):
    model = build_model(cfg, pretrained=False)
    save_checkpoint(model, get_optimizer(model, cfg), 0, tmp_path / 'last.pth')
    assert [p.name for p in tmp_path.iterdir()] == ['last.pth']  # no stray .tmp left behind
    save_checkpoint(model, get_optimizer(model, cfg), 1, tmp_path / 'last.pth')  # overwrite in place
    assert torch.load(tmp_path / 'last.pth', weights_only=True)['epoch'] == 1


def test_lr_schedule_continues_after_resume(cfg):
    """Why train.py builds the scheduler before restoring the optimizer."""
    c = load_config(ROOT / 'configs' / 'tusimple_res18.py', {'warmup': None, 'epoch': 100})
    lin = lambda: torch.nn.Linear(2, 2)

    opt_a = get_optimizer(lin(), c)
    sched_a = get_scheduler(opt_a, c, iters_per_epoch=10)
    for g in range(300):                       # 30 epochs, then "disconnect"
        sched_a.step(g)
    saved = opt_a.state_dict()
    sched_a.step(300)
    expected = opt_a.param_groups[0]['lr']

    opt_b = get_optimizer(lin(), c)            # correct order: scheduler first, then restore
    sched_b = get_scheduler(opt_b, c, iters_per_epoch=10)
    opt_b.load_state_dict(saved)
    sched_b.step(300)
    assert opt_b.param_groups[0]['lr'] == pytest.approx(expected)

    opt_c = get_optimizer(lin(), c)            # original order: restore first -> wrong base lr
    opt_c.load_state_dict(saved)
    sched_c = get_scheduler(opt_c, c, iters_per_epoch=10)
    sched_c.step(300)
    assert opt_c.param_groups[0]['lr'] != pytest.approx(expected, rel=0.05)

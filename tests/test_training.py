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

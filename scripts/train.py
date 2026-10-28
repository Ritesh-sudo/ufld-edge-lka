"""Train UFLD on TuSimple.

Sanity run (local, MPS):  python scripts/train.py --config configs/tusimple_res18.py --subset 200 --epochs 2
Full run (Colab, CUDA):   python scripts/train.py --config configs/tusimple_res18.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ufld  # noqa: F401  (must precede torch: sets the MPS CPU-fallback env var)

import argparse
import datetime
import logging

import torch
from torch.utils.tensorboard import SummaryWriter

from ufld.checkpoint import load_model_weights, save_checkpoint
from ufld.config import dump_config, load_config, parse_overrides
from ufld.data import build_train_loader
from ufld.device import get_device
from ufld.engine import train_one_epoch
from ufld.losses import LaneLoss
from ufld.model import build_model
from ufld.scheduler import get_optimizer, get_scheduler
from ufld.utils import set_seed

log = logging.getLogger('train')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--device', default=None, help='cuda | mps | cpu (default: best available)')
    ap.add_argument('--subset', type=int, default=None, help='train on the first N images only')
    ap.add_argument('--epochs', type=int, default=None, help='override cfg.epoch')
    ap.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE', help='config overrides')
    args = ap.parse_args()

    overrides = parse_overrides(args.set)
    if args.epochs is not None:
        overrides['epoch'] = args.epochs
    cfg = load_config(args.config, overrides)

    work_dir = Path(cfg.log_path) / f"{datetime.datetime.now():%Y%m%d_%H%M%S}_{Path(args.config).stem}"
    work_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(work_dir / 'train.log')])
    set_seed(cfg.seed)
    device = get_device(args.device)
    dump_config(cfg, work_dir)  # the exact resolved config, next to the checkpoints
    log.info('work dir: %s | device: %s | seed: %d', work_dir, device, cfg.seed)

    loader = build_train_loader(cfg, device, args.subset)
    model = build_model(cfg, pretrained=cfg.resume is None and cfg.finetune is None).to(device)
    optimizer = get_optimizer(model, cfg)

    if cfg.finetune is not None:
        # only the backbone ('model.*' keys) is taken from the finetune checkpoint
        state = torch.load(cfg.finetune, map_location='cpu', weights_only=True)['model']
        model.load_state_dict({k: v for k, v in state.items() if k.startswith('model.')}, strict=False)
    start_epoch = 0
    if cfg.resume is not None:
        ckpt = torch.load(cfg.resume, map_location='cpu', weights_only=True)
        load_model_weights(model, cfg.resume)
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        log.info('resumed from %s at epoch %d', cfg.resume, start_epoch)

    scheduler = get_scheduler(optimizer, cfg, len(loader))
    loss_fn = LaneLoss(cfg)
    writer = SummaryWriter(work_dir)
    log.info('%d iterations/epoch, %d epochs', len(loader), cfg.epoch)

    for epoch in range(start_epoch, cfg.epoch):
        mean_loss = train_one_epoch(model, loader, loss_fn, optimizer, scheduler, device, cfg, epoch, writer)
        log.info('epoch %d mean loss %.4f', epoch, mean_loss)
        save_checkpoint(model, optimizer, epoch, work_dir / 'last.pth')
        if (epoch + 1) % cfg.save_every == 0 or epoch + 1 == cfg.epoch:
            save_checkpoint(model, optimizer, epoch, work_dir / f'ep{epoch:03d}.pth')
    writer.close()


if __name__ == '__main__':
    main()

"""One training epoch. Ported from train.py (inference/calc_loss/train), minus DDP and .cuda()."""
import torch
from tqdm import tqdm

from ufld.metrics import AccTopk, MeanIoU, MultiLabelAcc


def build_metrics(cfg) -> dict:
    metrics = {'top1': MultiLabelAcc(), 'top2': AccTopk(cfg.griding_num, 2), 'top3': AccTopk(cfg.griding_num, 3)}
    if cfg.use_aux:
        metrics['iou'] = MeanIoU(cfg.num_lanes + 1)
    return metrics


def train_one_epoch(model, loader, loss_fn, optimizer, scheduler, device, cfg, epoch, writer=None, log_every=20):
    """Train for one epoch; returns the mean total loss."""
    model.train()
    metrics = build_metrics(cfg)
    total, steps = 0.0, 0
    bar = tqdm(loader, desc=f'epoch {epoch}')
    for b_idx, batch in enumerate(bar):
        global_step = epoch * len(loader) + b_idx
        if cfg.use_aux:
            img, cls_label, seg_label = batch
            seg_label = seg_label.long().to(device)
        else:
            (img, cls_label), seg_label = batch, None
        img, cls_label = img.to(device), cls_label.long().to(device)

        out = model(img)
        cls_out, seg_out = out if cfg.use_aux else (out, None)
        loss, parts = loss_fn(cls_out, cls_label, seg_out, seg_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(global_step)

        for m in metrics.values():
            m.reset()
        with torch.no_grad():
            pred = cls_out.argmax(dim=1)
            for name in ('top1', 'top2', 'top3'):
                metrics[name].update(pred, cls_label)
            if cfg.use_aux:
                metrics['iou'].update(seg_out.argmax(dim=1), seg_label)

        total += loss.item()
        steps += 1
        bar.set_postfix(loss=f'{loss.item():.3f}', **{k: f'{m.get():.3f}' for k, m in metrics.items()})
        if writer is not None:
            if global_step % log_every == 0:
                for name, value in parts.items():
                    writer.add_scalar(f'loss/{name}', value.item(), global_step)
                for name, m in metrics.items():
                    writer.add_scalar(f'metric/{name}', m.get(), global_step)
            writer.add_scalar('meta/lr', optimizer.param_groups[0]['lr'], global_step)
    return total / max(steps, 1)

"""Save/load checkpoints safely (weights_only) and device-agnostically."""
import os
from pathlib import Path

import torch


def save_checkpoint(model, optimizer, epoch: int, path: str | Path) -> None:
    # changed: store the epoch inside the file; the original parsed it out of "epNNN.pth".
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file and rename, so a Colab disconnect mid-write (esp. on Drive) can
    # never leave a truncated last.pth that the next resume would try to load.
    tmp = path.with_name(path.name + '.tmp')
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}, tmp)
    os.replace(tmp, path)


def load_model_weights(model, path: str | Path, device='cpu') -> dict:
    """Load `path` (the authors' or our own checkpoint) into `model`.

    Strips a DataParallel/DDP "module." prefix. changed: the original used
    load_state_dict(strict=False), which hides a genuinely missing layer. Here the only
    tolerated difference is training-only `aux_*` weights present in the file but absent
    from an inference model; anything else raises.
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)
    state = ckpt['model'] if 'model' in ckpt else ckpt
    state = {(k[len('module.'):] if k.startswith('module.') else k): v for k, v in state.items()}
    result = model.load_state_dict(state, strict=False)
    unexpected = [k for k in result.unexpected_keys if not k.startswith('aux_')]
    if result.missing_keys or unexpected:
        raise RuntimeError(f'checkpoint mismatch: missing={result.missing_keys} unexpected={unexpected}')
    return ckpt

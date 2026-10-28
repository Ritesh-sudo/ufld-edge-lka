"""One place that decides where tensors live: cuda -> mps -> cpu."""
import logging

import torch

logger = logging.getLogger(__name__)


def get_device(prefer: str | None = None) -> torch.device:
    """Return the best available device, or `prefer` if given ("cuda", "mps", "cpu").

    Asking for a device that is not available raises instead of silently using
    another one, so a benchmark can never run on the wrong backend by accident.
    """
    if prefer is not None:
        device = torch.device(prefer)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device 'cuda' requested but CUDA is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("device 'mps' requested but MPS is not available")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info("using device: %s", device)
    return device

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (all devices) so a run can be repeated."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

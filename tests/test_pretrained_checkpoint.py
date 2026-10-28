"""The authors' released TuSimple ResNet-18 checkpoint must load strictly and match the original model.

Skipped unless data/checkpoints/tusimple_18.pth exists (gitignored; see README of the original repo).
"""
from pathlib import Path

import pytest
import torch

from ufld.checkpoint import load_model_weights
from ufld.config import load_config
from ufld.device import get_device
from ufld.model import build_model

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / 'data' / 'checkpoints' / 'tusimple_18.pth'
pytestmark = pytest.mark.skipif(not CKPT.exists(), reason='pretrained checkpoint not downloaded')
CFG = load_config(ROOT / 'configs' / 'tusimple_res18.py')


def _x():
    return torch.randn(2, 3, 288, 800, generator=torch.Generator().manual_seed(7))


def test_loads_strictly_and_matches_reference(ref):
    ours = build_model(CFG, pretrained=False, use_aux=False)
    load_model_weights(ours, CKPT)  # raises on any missing key; the file has no aux_* weights
    theirs = ref.parsingNet(pretrained=False, backbone='18', cls_dim=(101, 56, 4), use_aux=False)
    theirs.load_state_dict(torch.load(CKPT, map_location='cpu', weights_only=True)['model'])
    ours.eval(), theirs.eval()
    with torch.no_grad():
        got, want = ours(_x()), theirs(_x())
    torch.testing.assert_close(got, want, rtol=0, atol=1e-5)
    assert got.std() > 1e-2  # not degenerate (e.g. all-equal logits)


@pytest.fixture
def strict_fp32():
    # Ampere+ GPUs run cuDNN convs/matmuls in TF32 by default (~1e-3 relative error), which would make
    # this comparison measure TF32 instead of the port. No effect on CPU/MPS.
    old = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = old


def test_cpu_and_accelerator_agree(strict_fp32):
    model = build_model(CFG, pretrained=False, use_aux=False)
    load_model_weights(model, CKPT)
    model.eval()
    with torch.no_grad():
        cpu = model(_x())
        dev = get_device()
        on_dev = model.to(dev)(_x().to(dev)).cpu()
    assert (cpu.argmax(1) == on_dev.argmax(1)).float().mean() > 0.999
    torch.testing.assert_close(on_dev, cpu, rtol=1e-3, atol=1e-3)

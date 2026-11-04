"""Exported ONNX / Core ML models must reproduce PyTorch on real test frames.

Tolerances (stated, not arbitrary): ONNX is fp32 so logits must agree to 1e-3 and the argmax
lane decisions exactly; Core ML runs fp16, so we require >= 99% argmax agreement (measured 99.7%+).
Skipped without the released checkpoint (gitignored). Random weights are not used because
near-tied logits would make an argmax comparison meaningless.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import ufld  # noqa: F401
from ufld.backends import load_coreml, load_onnx, load_torch
from ufld.checkpoint import load_model_weights
from ufld.config import load_config
from ufld.model import build_model
from ufld.parity import compare, sample_inputs

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / 'data' / 'checkpoints' / 'tusimple_18.pth'
pytestmark = pytest.mark.skipif(not CKPT.exists(), reason='pretrained checkpoint not downloaded')
CFG = load_config(ROOT / 'configs' / 'tusimple_res18.py')


@pytest.fixture(scope='module')
def model():
    m = build_model(CFG, pretrained=False, use_aux=False)
    load_model_weights(m, CKPT)
    return m.eval()


@pytest.fixture(scope='module')
def xs():
    return sample_inputs(CFG, 4)


@pytest.fixture(scope='module')
def ref():
    return load_torch(CFG, str(CKPT), torch.device('cpu'))


def test_onnx_matches_torch(model, xs, ref, tmp_path):
    pytest.importorskip('onnxruntime')
    path = tmp_path / 'm.onnx'
    torch.onnx.export(model, (torch.zeros(1, 3, *CFG.input_size),), str(path), opset_version=17,
                      input_names=['image'], output_names=['logits'], dynamo=False)
    res = compare('onnx', ref, load_onnx(str(path)), xs)
    assert res['max_abs_diff'] < 1e-3
    assert res['argmax_agreement'] == 1.0


@pytest.mark.skipif(sys.platform != 'darwin', reason='Core ML prediction needs macOS')
def test_coreml_matches_torch(model, xs, ref, tmp_path):
    ct = pytest.importorskip('coremltools')
    traced = torch.jit.trace(model, torch.zeros(1, 3, *CFG.input_size))
    ml = ct.convert(traced, convert_to='mlprogram', inputs=[ct.TensorType(name='image', shape=(1, 3, *CFG.input_size))],
                    outputs=[ct.TensorType(name='logits')], minimum_deployment_target=ct.target.macOS14)
    path = tmp_path / 'm.mlpackage'
    ml.save(str(path))
    res = compare('coreml-fp16', ref, load_coreml(str(path), 'CPU_ONLY'), xs)
    assert res['argmax_agreement'] >= 0.99
    assert np.isfinite(res['max_abs_diff'])

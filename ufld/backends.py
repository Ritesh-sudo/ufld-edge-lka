"""Uniform inference wrappers so one benchmark/eval harness can time every backend.

Each loader returns a callable `f(x: np.ndarray[1,3,H,W] float32) -> np.ndarray` of logits with
shape (1, griding_num+1, rows, lanes). The call includes host<->device copies and a device sync,
so the measured time is the true end-to-end model time a caller would see.
"""
from __future__ import annotations

import numpy as np
import torch

from .checkpoint import load_model_weights
from .model import build_model


def load_torch(cfg, ckpt: str, device: torch.device):
    model = build_model(cfg, pretrained=False, use_aux=False)
    load_model_weights(model, ckpt, device='cpu')
    model.to(device).eval()

    @torch.no_grad()
    def run(x: np.ndarray) -> np.ndarray:
        out = model(torch.from_numpy(x).to(device))
        if device.type == 'mps':
            torch.mps.synchronize()
        elif device.type == 'cuda':
            torch.cuda.synchronize()
        return out.cpu().numpy()
    return run


def load_onnx(path: str, providers: list[str] | None = None):
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=providers or ['CPUExecutionProvider'])
    name = sess.get_inputs()[0].name
    return lambda x: sess.run(None, {name: x})[0]


def load_coreml(path: str, compute_units: str = 'ALL'):
    """compute_units: ALL (CPU+GPU+Neural Engine), CPU_ONLY, CPU_AND_GPU, CPU_AND_NE."""
    import coremltools as ct
    model = ct.models.MLModel(path, compute_units=getattr(ct.ComputeUnit, compute_units))
    spec = model.get_spec()
    in_name, out_name = spec.description.input[0].name, spec.description.output[0].name
    return lambda x: np.asarray(model.predict({in_name: x})[out_name])

"""Inference-only CLRNet (Zheng et al., CVPR 2022), ported to plain PyTorch.

Upstream: https://github.com/Turoad/CLRNet (Apache-2.0, see LICENSE_CLRNET.txt). Kept isolated from `ufld/`.
The port removes the old-mmcv dependency and replaces the CUDA lane-NMS op with a pure PyTorch one,
so it runs on CUDA, MPS and CPU with current PyTorch. Training code and the auxiliary segmentation
decoder (training-time loss only) are not ported.
"""
from .model import CLRConfig, CLRNet, load_clrnet, preprocess_bgr
from .decode import decode_lanes, lanes_to_tusimple
from .nms import lane_nms

__all__ = ['CLRConfig', 'CLRNet', 'load_clrnet', 'preprocess_bgr', 'decode_lanes', 'lanes_to_tusimple', 'lane_nms']

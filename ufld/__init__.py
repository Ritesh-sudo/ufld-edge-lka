"""Ultra-Fast Lane Detection, modernized for CUDA / MPS / CPU.

Ported from https://github.com/cfzd/Ultra-Fast-Lane-Detection (Qin et al., ECCV 2020).
"""
import os

# changed: the original assumed CUDA. On Apple Silicon, let PyTorch fall back to the CPU
# for any op MPS does not implement. PyTorch emits a UserWarning for each fallback, so
# the behavior change is visible in the logs rather than silent. This variable is read
# when torch loads, so it must be set before `import torch` (importing `ufld` first does it).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

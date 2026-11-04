# Edge-Deployable Lane Detection for Lane-Keep Assist

A reproduction and modernization of **Ultra Fast Structure-aware Deep Lane Detection** (Qin, Wang, Li; ECCV 2020), with an edge-deployment extension aimed at ADAS lane-keep assist. Work in progress; see `CLAUDE.md` for the plan and phases.

## Credit

The model, losses, data pipeline and evaluation are **ported from the authors' code**, <https://github.com/cfzd/Ultra-Fast-Lane-Detection> (MIT, Copyright (c) 2020 cfzd; notice in `THIRD_PARTY_LICENSE_UFLD.txt`). The port changes the engineering (CUDA/MPS/CPU support, no DALI/DDP, Python configs, tests), not the method. Where original logic was changed, a `# changed:` comment says what and why. Please cite the original work:

```bibtex
@InProceedings{qin2020ultra,
  author    = {Qin, Zequn and Wang, Huanyu and Li, Xi},
  title     = {Ultra Fast Structure-aware Deep Lane Detection},
  booktitle = {The European Conference on Computer Vision (ECCV)},
  year      = {2020}
}
```

## Status (Phases 1-4; Colab training still pending)

| | Accuracy | FP | FN |
|---|---|---|---|
| Paper, ResNet-18 (Table 5; accuracy and runtime only, no FP/FN) | 95.87% | n/a | n/a |
| Authors' released ResNet-18 checkpoint (their README) | 95.82% | n/a | n/a |
| **This port + the released checkpoint**, full TuSimple test set (2,782 images), Apple M4 (MPS) | **95.82%** (95.8154) | 0.1904 | 0.0391 |
| This port, trained from scratch | *not yet run* (Colab, `notebooks/colab_train.ipynb`) | | |

The released-checkpoint row shows the port (model, data decoding, official metric) reproduces the authors' number. It says nothing yet about training.

**Verification.** `pytest` includes golden tests that run the original code (cloned into `reference/`, gitignored) next to the port and require identical outputs: model with shared weights, all losses, row-anchor labels, converter masks, the TuSimple metric, LR schedules. The tests on real TuSimple frames run on Colab (the local machine has only the test set). Not yet verified: CUDA execution (notebook written, not yet run on Colab).

## Edge deployment (Phase 3)

Apple M4, batch 1, 288x800 input, 30 warm-up + 200 timed runs, model forward only (host/device copies included, image loading excluded). Authors' released ResNet-18 checkpoint, full TuSimple test set (2,782 images). Plot: `results/edge_accuracy_vs_latency.png`; raw data: `results/edge_summary.csv`, `benchmark*.csv`, `backend_accuracy.csv`.

| Backend | Mean ms | p95 ms | FPS | Accuracy |
|---|---|---|---|---|
| Core ML, Neural Engine (fp16) | 2.38 | 2.54 | 420 | 95.8142% |
| ONNX Runtime + Core ML provider | 3.08 | 3.30 | 324 | not measured |
| Core ML, CPU+GPU (fp16) | 4.36 | 4.60 | 229 | 95.8164%* |
| ONNX Runtime CPU, INT8 | 7.44 | 7.58 | 134 | 95.7758% |
| PyTorch MPS | 7.88 | 8.17 | 127 | 95.8154%* |
| Core ML, CPU (fp16) | 8.08 | 8.55 | 124 | 95.8164% |
| ONNX Runtime CPU, fp32 | 16.18 | 19.71 | 62 | 95.8154% |
| PyTorch CPU | 27.19 | 27.91 | 37 | 95.8154% |

\* Not run separately: value taken from the same model on another compute unit (CPU run / MPS run).

- **Parity.** ONNX matches PyTorch to 2.5e-5 max logit difference with identical lane decisions; Core ML fp16 agrees on 99.7-99.8% of per-row decisions and costs ~0.001 accuracy points (`tests/test_export_parity.py`).
- **INT8** (ONNX Runtime static PTQ, QDQ, per-channel): -0.04 accuracy points, 2.2x faster than ONNX fp32 on CPU, 4x smaller (61.7 MB vs 245 MB). Calibrated on 100 *test-set* frames (images only, no labels) because only the test split exists locally; this is mild leakage. This path does not use the Neural Engine, so the fastest (Core ML fp16) and the INT8 results are separate points.
- Decoding logits to lane coordinates adds ~0.2-0.3 ms (about 3 ms on ONNX CPU). M4 numbers are not comparable to the paper's desktop-GPU FPS.

```bash
python scripts/export_onnx.py   --config configs/tusimple_res18.py --ckpt data/checkpoints/tusimple_18.pth
python scripts/export_coreml.py --config configs/tusimple_res18.py --ckpt data/checkpoints/tusimple_18.pth
python scripts/quantize_onnx.py --config configs/tusimple_res18.py --onnx data/exports/ufld_res18.onnx --out data/exports/ufld_res18_int8.onnx
python scripts/benchmark.py     --config configs/tusimple_res18.py --ckpt data/checkpoints/tusimple_18.pth --onnx data/exports/ufld_res18.onnx --coreml data/exports/ufld_res18_fp16.mlpackage --backends torch-mps torch-cpu onnx-cpu coreml-ne
python scripts/eval_backend.py  --config configs/tusimple_res18.py --ckpt data/checkpoints/tusimple_18.pth --backend onnx-cpu --onnx data/exports/ufld_res18.onnx
```

## ADAS layer (Phase 4)

`adas/geometry.py` turns the detector's lanes into the two quantities a lane-keep-assist (LKA) controller needs; `adas/departure.py` adds a lane-departure warning (LDW); `scripts/demo_video.py` renders both on video (`--video clip.mp4`, or `--frames <glob>`).

- **Lateral offset** (m, + = car right of lane center): the cross-track error. Ego lanes are the nearest detected lane on each side of the image center; if either side is missing, no offset is produced (a controller should hand over, not act on a guess).
- **Curvature** (1/m): a quadratic fit of the lane-center line in road coordinates. Positive = road bends right ahead.
- **Warning state machine:** OK / WARN_LEFT / WARN_RIGHT / LOST, with hysteresis (warn at 0.6 m, clear at 0.4 m), a 3-frame debounce, and a LOST state after 5 frames without an estimate.
- **How this would feed a controller.** A typical LKA steering law is `steer = Kp * offset + Kd * d(offset)/dt + steer_ff(curvature)`: the offset is the feedback error term, curvature (together with speed) gives the feed-forward steering needed to follow the bend, and LOST/low-confidence tells the controller to disengage and warn the driver. This repo stops at the perception outputs; no controller is implemented or tested here.

**Limits, stated plainly.** There is no camera calibration for TuSimple, so meters rest on two assumptions (3.7 m lane width, 1000 px focal length on a 1280-wide frame; flat road, pinhole camera). The geometry is verified against synthetic roads built from the same model (`tests/test_adas_*.py`), which proves the math is self-consistent, **not** that the metric values match a real vehicle: there is no ground-truth offset to compare with. Treat meters as indicative. The released checkpoint is trained on TuSimple's highway view; a different camera will degrade it. The demo video has not yet been recorded on a real dashcam clip (only 20th frames of TuSimple test clips exist locally, so there is no consecutive-frame footage to use).

## Quick start

```bash
uv venv --python 3.12 venv && source venv/bin/activate && uv pip install -r requirements.txt
pytest
python scripts/eval_tusimple.py --config configs/tusimple_res18.py --ckpt data/checkpoints/tusimple_18.pth
```

Data layout, the Kaggle mirror (the official TuSimple links are dead), and training commands are in `CLAUDE.md` (Commands) and `notebooks/colab_train.ipynb`.

## Layout

`ufld/` model, losses, data, evaluation · `scripts/` train, eval, dataset tools · `configs/` all hyperparameters · `tests/` unit and golden tests · `results/` metrics CSVs

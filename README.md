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

## Status (Phases 1-5 (CUDA-NMS check and Colab training pending); Colab training still pending)

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

## Comparison: UFLD vs CLRNet vs classic baseline (Phase 5)

Same TuSimple test set (2,782 images), same metric code (`LaneEval`) for every row, authors' unmodified weights. Latency: Apple M4, batch 1, one fixed real frame, 30 warm-up + 200 timed runs, each model at its native input size; "+decode" includes the model's own post-processing (UFLD: softmax/argmax; CLRNet: confidence filter + lane NMS + spline sampling). Image decoding/resizing is excluded. Run-to-run variation on this machine is about +/-10%. Data: `results/comparison.csv`; chart: `results/comparison_accuracy_vs_latency.png`.

| Model | Input | Device | Model ms | +decode ms | Accuracy | FP | FN |
|---|---|---|---|---|---|---|---|
| Predict no lanes (floor) | n/a | n/a | n/a | n/a | 43.02% | 0.9527 | 0.9761 |
| OpenCV Canny + Hough | 720x1280 | CPU | n/a | 1.7 | 51.06% | 0.9840 | 0.9915 |
| UFLD ResNet-18 | 288x800 | Core ML (ANE, fp16) | 2.4 | 2.6 | 95.81% | n/a | n/a |
| UFLD ResNet-18 | 288x800 | PyTorch MPS | 9.1 | 9.5 | 95.82% | 0.1904 | 0.0391 |
| UFLD ResNet-18 | 288x800 | PyTorch CPU | 30.1 | 30.3 | 95.82% | 0.1904 | 0.0391 |
| CLRNet ResNet-18 | 320x800 | PyTorch MPS | 17.2 | 17.5 | **96.84%** | 0.0231 | 0.0194 |
| CLRNet ResNet-18 | 320x800 | PyTorch CPU | 181.1 | 180.5 | 96.84%* | | |

\* CPU accuracy not run separately; same weights as the MPS row. The UFLD Core ML FP/FN were not recorded.

**Reading it.** CLRNet is about 1 accuracy point better than UFLD (96.84 vs 95.82) and roughly 2x slower on the same device (MPS), 6x on CPU. Note the two use different FP/FN conventions (UFLD's released numbers have far higher FP, 0.19 vs 0.02), which is the real accuracy difference: UFLD hallucinates extra lanes. On the TuSimple metric the headline gap looks small because both are near its ceiling. CLRNet has no Core ML export here, so the 2.4 ms Neural Engine point is UFLD-only; that favors UFLD in the "best available" comparison and is not like-for-like.

**CLRNet reproduction.** Paper (CVPR 2022, ResNet-18, TuSimple): Acc 96.84, FDR 2.28, FNR 1.92. This port, authors' checkpoint, our metric: **96.8357**, FP 2.31, FN 1.94. Weights: [GitHub release `models/tusimple_r18.pth.zip`](https://github.com/Turoad/CLRNet/releases/tag/models) (sha256 of the zip `e0293100...c465`). Code is an inference-only port in `third_party/clrnet/` (no old mmcv; plain PyTorch on MPS/CPU/CUDA; the CUDA lane-NMS replaced by `nms.py`), credited to [Turoad/CLRNet](https://github.com/Turoad/CLRNet) (Apache-2.0, `third_party/clrnet/LICENSE_CLRNET.txt`). Please cite Zheng et al., CVPR 2022.

**What is and is not verified for the NMS port.** Verified: it equals a line-by-line scalar transliteration of the CUDA kernel on random stress cases and on 100 real CLRNet inputs (`tests/test_clrnet_nms.py`), and the end-to-end accuracy reproduces the paper. **Not yet verified: against the real CUDA op.** That needs a GPU runtime: run `notebooks/clrnet_nms_check.ipynb` on Colab and record the result here. Until then the port is unproven against CUDA.

**Why CLRNet is more accurate (and slower).** UFLD classifies one global feature vector into column cells at fixed rows. CLRNet keeps 192 lane priors and refines each across three feature-pyramid levels (cross-layer refinement): high-level features find the lane, low-level features place it precisely, then lane-NMS picks the final lanes.

**Reading the metric.** TuSimple accuracy counts a row as correct when prediction and ground truth are both "no lane", so predicting nothing scores 43%; Hough is only ~8 points above that.

## Quick start

```bash
uv venv --python 3.12 venv && source venv/bin/activate && uv pip install -r requirements.txt
pytest
python scripts/eval_tusimple.py --config configs/tusimple_res18.py --ckpt data/checkpoints/tusimple_18.pth
```

Data layout, the Kaggle mirror (the official TuSimple links are dead), and training commands are in `CLAUDE.md` (Commands) and `notebooks/colab_train.ipynb`.

## Layout

`ufld/` model, losses, data, evaluation · `scripts/` train, eval, dataset tools · `configs/` all hyperparameters · `tests/` unit and golden tests · `results/` metrics CSVs

# CLAUDE.md

## Project: Edge-Deployable Lane Detection for Lane-Keep Assist

Reproduction and extension of **"Ultra Fast Structure-aware Deep Lane Detection"**
(Qin, Wang, Li — ECCV 2020). Original code: https://github.com/cfzd/Ultra-Fast-Lane-Detection

This is a portfolio project. The goal is a clean, honest, well-documented reproduction
plus an original edge-deployment contribution, framed around how lane detection feeds a
real ADAS lane-keep-assist (LKA) system.

**Owner:** Ritesh Singh (M.S. AI & Data Science, SJSU)

---

## Goals (in order)

1. **Reproduce** the TuSimple ResNet-18 baseline and land close to the paper's reported
   accuracy. Reference numbers (checked against the paper, arXiv:2004.11757v4, Table 5):
   - Paper, ResNet-18: **95.87%** accuracy (ResNet-34: 96.06%). Table 5 reports accuracy and
     runtime only; **the paper has no FP/FN for TuSimple**, so FP/FN are reported for our runs
     but there is nothing to compare them to.
   - Authors' released ResNet-18 checkpoint (original repo README): 95.82%.
   - Our port + the released checkpoint, full test set (2,782 images), Apple M4 / MPS:
     **95.8154%** (FP 0.1904, FN 0.0391), `results/pretrained_eval.csv`.
2. **Modernize** the code to run on current PyTorch across CUDA (Colab), MPS (Apple
   Silicon), and CPU from a single codebase.
3. **Benchmark on edge hardware** (original contribution): latency/FPS on the M4 with
   PyTorch MPS, ONNX, and Core ML (Apple Neural Engine), plus INT8 quantization, reporting
   the accuracy-vs-speed trade-off.
4. **Make it ADAS-shaped**: compute lateral offset from lane center and lane curvature
   from predictions; build a dashcam video demo with a lane-departure warning overlay.
5. **Compare against CLRNet** (Zheng et al., CVPR 2022; https://github.com/Turoad/CLRNet)
   using its pretrained TuSimple weights, inference only, no training. CLRNet is the
   "stronger, slower" reference point for the accuracy-vs-latency chart. Also include a
   classic OpenCV baseline (Canny + Hough) as the pre-deep-learning reference.
6. **Stretch:** compare with UFLDv2, evaluate CULane hard categories (night, shadow,
   no-line, curve) to document failure modes, or closed-loop LKA in a simulator.

---

## Hardware & Environment

- **Local dev:** M4 MacBook, PyTorch with MPS backend. Used for development, debugging,
  small-subset sanity training, benchmarking, export, and the demo.
- **Training:** Google Colab T4 (CUDA). Full training runs happen there, not locally.
- **Workflow:** develop locally → push to GitHub → pull and train in Colab → download
  checkpoint → benchmark and demo locally.
- Claude Code cannot drive Colab. For training, produce scripts/notebooks I run myself.

---

## Key Technical Concepts (explain these when touching related code)

- **Row anchors:** lanes are predicted as a column-cell classification at a fixed set of
  image rows, instead of per-pixel segmentation. This is why the model is fast.
- **Gridding:** each row is split into column cells, plus one extra "no lane" class.
- **Structural loss:** similarity loss (neighboring rows' predictions stay close) and
  shape loss (lanes stay smooth / near-straight locally).
- **Auxiliary segmentation branch:** used only during training to improve features;
  removed at inference, so it adds no runtime cost.
- TuSimple reference config in the original repo: 288×800 input, 100 gridding cells,
  56 row anchors, 4 lanes. Verify against the repo's config before relying on these.

When you modify code that implements any of these, briefly explain what the code does and
why. I need to be able to explain every part of this project in interviews.

### CLRNet notes (Phase 5)

- CLRNet refines lane anchors across feature-pyramid levels (cross-layer refinement),
  using high-level features to find lanes and low-level features to localize them
  precisely. Explain this when working on its code, and contrast it with UFLD's row
  anchors.
- Its repo uses a custom CUDA extension for lane NMS and depends on older library
  versions (notably an old mmcv). Expect environment work on Colab.
- On the M4, replace the CUDA NMS with a pure PyTorch implementation. Verify it matches
  the CUDA version's outputs on Colab (same kept lanes on a sample of test images) before
  using it for benchmarks.
- Check that the pretrained TuSimple weight links still work before building around them.
- Keep CLRNet code isolated in `third_party/clrnet/` with its own requirements file, so its
  old dependencies do not leak into the UFLD environment.

---

## Repository Layout (target)

```
.
├── CLAUDE.md
├── README.md
├── configs/            # dataset/model configs (tusimple_res18.py, etc.)
├── ufld/               # model, losses, data loading (modernized from original)
├── scripts/
│   ├── train.py
│   ├── eval_tusimple.py
│   ├── benchmark.py    # latency/FPS across backends
│   ├── export_onnx.py
│   ├── export_coreml.py
│   └── demo_video.py   # lane overlay + offset/curvature + departure warning
├── adas/               # lateral offset, curvature, departure logic
├── baselines/
│   └── opencv_hough.py # classic Canny + Hough lane baseline
├── third_party/
│   └── clrnet/         # CLRNet code, isolated deps, PyTorch NMS port
│   (scripts/compare.py runs all models through one eval + benchmark harness)
├── notebooks/
│   └── colab_train.ipynb
├── tests/
├── results/            # metrics tables, benchmark CSVs, plots (committed)
└── data/               # NOT committed (gitignored)
```

---

## Coding Conventions

- Python 3.10+, current stable PyTorch. Type hints on public functions.
- **Device-agnostic everywhere:** one helper picks `cuda` → `mps` → `cpu`. Never hardcode
  `.cuda()`.
- Remove dependencies that do not run on macOS (e.g., NVIDIA DALI). Use standard
  `torch.utils.data` loaders.
- If an op is unsupported on MPS, fall back explicitly and log a warning; do not silently
  change behavior.
- Configs over magic numbers. All hyperparameters live in `configs/`.
- Fixed seeds and logged configs for every training run, for reproducibility.
- Keep diffs from the original repo understandable: when changing original logic, leave a
  short comment noting what changed and why.
- No large files in git: datasets, checkpoints, and videos go in `data/` or releases.

---

## Commands

Fill these in as they are created:

```bash
# environment (Python 3.12)
# uv venv --python 3.12 venv && source venv/bin/activate && uv pip install -r requirements.txt

# TuSimple: the official links are dead; the Kaggle mirror manideep1108/tusimple is used.
# Full dataset (Colab does this in notebooks/colab_train.ipynb): download, then
python scripts/arrange_tusimple.py --src <extracted download> --dst data/tusimple   # merge train/test clips
python scripts/convert_tusimple.py --root data/tusimple                            # masks, train_gt.txt, test.txt
# Local sample only (rate-limited API; slow, resumable): python scripts/fetch_tusimple_kaggle.py {meta,test,train}

# sanity training on a small subset (local, MPS). python.org builds of Python lack a CA bundle,
# so downloading the ImageNet backbone may need:  export SSL_CERT_FILE=$(python -c 'import certifi;print(certifi.where())')
python scripts/train.py --config configs/tusimple_res18.py --subset 200 --epochs 2

# full training (Colab): a fixed --work-dir makes it resume from <work-dir>/last.pth after a disconnect
python scripts/train.py --config configs/tusimple_res18.py --work-dir /content/drive/MyDrive/ufld/res18_tusimple

# evaluation (official TuSimple Accuracy/FP/FN); --out appends a row to a results CSV
python scripts/eval_tusimple.py --config configs/tusimple_res18.py --ckpt data/checkpoints/tusimple_18.pth

# edge (Phase 3): export, quantize, benchmark, per-backend accuracy, report; see README "Edge deployment"
python scripts/export_onnx.py --config configs/tusimple_res18.py --ckpt <ckpt>
python scripts/export_coreml.py --config configs/tusimple_res18.py --ckpt <ckpt>
python scripts/quantize_onnx.py --config configs/tusimple_res18.py --onnx data/exports/ufld_res18.onnx --out data/exports/ufld_res18_int8.onnx
python scripts/benchmark.py --config configs/tusimple_res18.py --ckpt <ckpt> --onnx ... --coreml ... --backends torch-mps coreml-ne ...
python scripts/eval_backend.py --config configs/tusimple_res18.py --ckpt <ckpt> --backend coreml-ne --coreml ...
python scripts/make_edge_report.py

# demo
# python scripts/demo_video.py --video path/to/dashcam.mp4 --ckpt ...

# tests (golden tests compare against reference/Ultra-Fast-Lane-Detection)
pytest
```

---

## Evaluation & Reporting Rules

- Use the official TuSimple metrics: **Accuracy, FP, FN**. Do not invent alternative
  metrics for the headline result.
- Report reproduced numbers next to the paper's numbers, with honest notes on any gap and
  likely causes (epochs, batch size, augmentation, hardware).
- Benchmarks: report warm-up excluded, mean and p50/p95 latency over a fixed number of
  runs, batch size 1, stated input resolution and backend. Note that M4 numbers are not
  comparable to the paper's desktop-GPU FPS.
- Quantization results always paired with the accuracy change, not speed alone.
- Save all results to `results/` as CSV plus plots, so the README can reference them.

---

## Definition of Done per Phase

- **Phase 1 (reproduce):** trains on Colab; eval script produces TuSimple metrics; results
  table in `results/`.
- **Phase 2 (modernize):** same code runs on CUDA, MPS, CPU; small-subset loss decreases
  on MPS; tests pass.
- **Phase 3 (edge):** benchmark table across backends; ONNX and Core ML outputs match
  PyTorch within a stated tolerance; INT8 accuracy/speed trade-off documented.
- **Phase 4 (ADAS):** offset/curvature functions tested; demo video with overlay and
  departure warning; README explains how outputs would feed an LKA controller.
- **Phase 5 (comparison):** UFLD, CLRNet, and the OpenCV baseline evaluated on the same
  TuSimple test set with the same metric script; CLRNet's reproduced accuracy checked
  against its paper; PyTorch NMS port verified against CUDA NMS; one accuracy-vs-latency
  chart and table in `results/`.

### Comparison fairness rules

- Same device, batch size 1, same warm-up and number of timed runs for every model.
- Report each model's native input resolution, and state whether timing includes
  pre/post-processing (NMS, decoding). Report both if possible.
- Use the authors' pretrained weights unmodified; note the source of every checkpoint.

---

## Working Style

- Before large changes, outline the plan and wait for my go-ahead.
- Prefer small, reviewable steps; run code after changes and fix errors you see.
- Be honest about limitations. If something did not reproduce, say so plainly.
- Credit the original authors and repo clearly in the README.

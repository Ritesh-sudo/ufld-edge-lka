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

## Status (Phase 1-2)

| | Accuracy | FP | FN |
|---|---|---|---|
| Paper, ResNet-18 (Table 5; accuracy and runtime only, no FP/FN) | 95.87% | n/a | n/a |
| Authors' released ResNet-18 checkpoint (their README) | 95.82% | n/a | n/a |
| **This port + the released checkpoint**, full TuSimple test set (2,782 images), Apple M4 (MPS) | **95.82%** (95.8154) | 0.1904 | 0.0391 |
| This port, trained from scratch | *not yet run* (Colab, `notebooks/colab_train.ipynb`) | | |

The released-checkpoint row shows the port (model, data decoding, official metric) reproduces the authors' number. It says nothing yet about training.

**Verification.** `pytest` includes golden tests that run the original code (cloned into `reference/`, gitignored) next to the port and require identical outputs: model with shared weights, all losses, row-anchor labels, converter masks, the TuSimple metric, LR schedules. The tests on real TuSimple frames run on Colab (the local machine has only the test set). Not yet verified: CUDA execution (notebook written, not yet run on Colab).

## Quick start

```bash
uv venv --python 3.12 venv && source venv/bin/activate && uv pip install -r requirements.txt
pytest
python scripts/eval_tusimple.py --config configs/tusimple_res18.py --ckpt data/checkpoints/tusimple_18.pth
```

Data layout, the Kaggle mirror (the official TuSimple links are dead), and training commands are in `CLAUDE.md` (Commands) and `notebooks/colab_train.ipynb`.

## Layout

`ufld/` model, losses, data, evaluation · `scripts/` train, eval, dataset tools · `configs/` all hyperparameters · `tests/` unit and golden tests · `results/` metrics CSVs

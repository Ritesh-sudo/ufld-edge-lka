"""Dashcam demo: lane overlay + lateral offset + curvature + lane-departure warning.

python scripts/demo_video.py --video clip.mp4 --out results/demo.mp4 --backend coreml-ne --coreml data/exports/ufld_res18_fp16.mlpackage
python scripts/demo_video.py --frames data/tusimple/clips/0530/*/20.jpg --fps 10 ...   # image sequence / smoke test

Pipeline per frame: resize to the 1280x720 TuSimple canonical frame -> resize to 288x800 and
normalize -> detector -> lane x at 56 row anchors -> ego lanes -> offset/curvature -> departure
state machine -> draw. Geometry assumptions (focal length, lane width) are in adas/geometry.py;
a camera that differs from TuSimple's (mounting, FOV) will give degraded detections and
less meaningful meters.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ufld  # noqa: F401

import argparse
import glob
import time

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import make_backend  # noqa: E402

from adas.departure import DepartureWarning, State  # noqa: E402
from adas.geometry import NO_LANE, CameraModel, lane_state, select_ego_lanes  # noqa: E402
from ufld.config import load_config  # noqa: E402
from ufld.data import NORM_MEAN, NORM_STD  # noqa: E402
from ufld.evaluation import generate_tusimple_lines  # noqa: E402

CANON_W, CANON_H = 1280, 720
COLORS = [(255, 120, 0), (0, 200, 255), (0, 200, 255), (255, 120, 0)]  # BGR; inner (ego) lanes yellow
BANNER = {State.OK: ((60, 160, 60), 'LANE KEEPING'), State.WARN_LEFT: ((0, 0, 220), 'LANE DEPARTURE LEFT'),
          State.WARN_RIGHT: ((0, 0, 220), 'LANE DEPARTURE RIGHT'), State.LOST: ((90, 90, 90), 'LANE LOST')}


def frames_from(args):
    if args.video:
        cap = cv2.VideoCapture(args.video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        def gen():
            while True:
                ok, f = cap.read()
                if not ok:
                    return
                yield f
        return gen(), fps
    paths = sorted(glob.glob(args.frames))
    return (cv2.imread(p) for p in paths), args.fps


def preprocess(frame_bgr, cfg):
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).resize(
        (cfg.input_size[1], cfg.input_size[0]), Image.BILINEAR)  # same bilinear resize as the eval transform
    x = (np.asarray(img, dtype=np.float32) / 255.0 - np.array(NORM_MEAN, np.float32)) / np.array(NORM_STD, np.float32)
    return x.transpose(2, 0, 1)[None].astype(np.float32)


def draw(frame, lanes, ys, ego, state, st, dep):
    over = frame.copy()
    if ego is not None:  # translucent ego-lane corridor
        ok = (ego.left != NO_LANE) & (ego.right != NO_LANE)
        pts_l = np.stack([ego.left[ok], ys[ok]], 1)
        pts_r = np.stack([ego.right[ok], ys[ok]], 1)
        if ok.sum() >= 2:
            poly = np.concatenate([pts_l, pts_r[::-1]]).astype(np.int32)
            cv2.fillPoly(over, [poly], (0, 160, 0) if dep == State.OK else (0, 0, 200))
    frame = cv2.addWeighted(over, 0.3, frame, 0.7, 0)
    for i, lane in enumerate(lanes):
        for x, y in zip(lane, ys):
            if x != NO_LANE:
                cv2.circle(frame, (int(x), int(y)), 4, COLORS[i], -1)
    color, text = BANNER[dep]
    cv2.rectangle(frame, (0, 0), (CANON_W, 56), color, -1)
    cv2.putText(frame, text, (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)
    if st is not None:
        r = f'R={1 / abs(st.curvature_1pm):.0f} m' if abs(st.curvature_1pm) > 1 / 3000 else 'straight'
        info = f'offset {st.offset_m:+.2f} m   curvature {r}'
    else:
        info = 'offset n/a'
    cv2.putText(frame, info, (16, CANON_H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/tusimple_res18.py')
    ap.add_argument('--ckpt', default='data/checkpoints/tusimple_18.pth')
    ap.add_argument('--backend', default='torch-mps')
    ap.add_argument('--onnx')
    ap.add_argument('--coreml')
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--video')
    src.add_argument('--frames', help='glob of image files, sorted')
    ap.add_argument('--fps', type=float, default=10.0, help='output fps for --frames')
    ap.add_argument('--max-frames', type=int, default=None)
    ap.add_argument('--out', default='results/demo.mp4')
    args = ap.parse_args()

    cfg = load_config(args.config)
    detect = make_backend(args.backend, args, cfg)
    cam, dep = CameraModel(), DepartureWarning()
    ys = np.array(cfg.test_h_samples, dtype=float)
    frames, fps = frames_from(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (CANON_W, CANON_H))

    n, t_det, states = 0, 0.0, {}
    for frame in frames:
        if args.max_frames and n >= args.max_frames:
            break
        frame = cv2.resize(frame, (CANON_W, CANON_H))
        t0 = time.perf_counter()
        logits = torch.from_numpy(detect(preprocess(frame, cfg)))
        lanes = generate_tusimple_lines(logits[0], cfg.griding_num, CANON_W)
        t_det += time.perf_counter() - t0
        ego = select_ego_lanes(lanes, ys, cam)
        st = lane_state(ego, ys, cam) if ego is not None else None
        state = dep.update(st.offset_m if st else None)
        states[state.value] = states.get(state.value, 0) + 1
        writer.write(draw(frame, lanes, ys, ego, state, st, state))
        n += 1
    writer.release()
    print(f'{n} frames -> {args.out}; detector+decode {1000 * t_det / max(n, 1):.1f} ms/frame; states: {states}')


if __name__ == '__main__':
    main()

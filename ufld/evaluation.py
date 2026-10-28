"""Official TuSimple evaluation: Accuracy, FP, FN.

Ported from evaluation/tusimple/lane.py and evaluation/eval_wrapper.py. The scoring follows
the TuSimple benchmark: a predicted lane matches a ground-truth lane if >= 85% of its points
(at the benchmark's 56 y positions) are within 20 px / cos(lane angle).
"""
import json
from pathlib import Path

import numpy as np
import scipy.special
import torch
from tqdm import tqdm


class LaneEval:
    pixel_thresh = 20
    pt_thresh = 0.85

    @staticmethod
    def get_angle(xs, y_samples):
        xs, ys = xs[xs >= 0], y_samples[xs >= 0]
        if len(xs) > 1:
            # changed: np.polyfit slope instead of sklearn LinearRegression (same least-squares fit),
            # which drops the scikit-learn dependency.
            k = np.polyfit(ys, xs, 1)[0]
            return np.arctan(k)
        return 0

    @staticmethod
    def line_accuracy(pred, gt, thresh):
        pred = np.array([p if p >= 0 else -100 for p in pred])
        gt = np.array([g if g >= 0 else -100 for g in gt])
        return np.sum(np.where(np.abs(pred - gt) < thresh, 1., 0.)) / len(gt)

    @staticmethod
    def bench(pred, gt, y_samples, running_time):
        if any(len(p) != len(y_samples) for p in pred):
            raise ValueError('Format of lanes error.')
        if running_time > 200 or len(gt) + 2 < len(pred):
            return 0., 0., 1.
        angles = [LaneEval.get_angle(np.array(x_gts), np.array(y_samples)) for x_gts in gt]
        threshs = [LaneEval.pixel_thresh / np.cos(angle) for angle in angles]
        line_accs = []
        fn = 0.
        matched = 0.
        for x_gts, thresh in zip(gt, threshs):
            accs = [LaneEval.line_accuracy(np.array(x_preds), np.array(x_gts), thresh) for x_preds in pred]
            max_acc = np.max(accs) if len(accs) > 0 else 0.
            if max_acc < LaneEval.pt_thresh:
                fn += 1
            else:
                matched += 1
            line_accs.append(max_acc)
        fp = len(pred) - matched
        if len(gt) > 4 and fn > 0:
            fn -= 1
        s = sum(line_accs)
        if len(gt) > 4:
            s -= min(line_accs)
        return (s / max(min(4.0, len(gt)), 1.),
                fp / len(pred) if len(pred) > 0 else 0.,
                fn / max(min(len(gt), 4.), 1.))

    @staticmethod
    def bench_one_submit(pred_file, gt_file) -> dict[str, float]:
        json_pred = [json.loads(line) for line in open(pred_file).readlines()]
        json_gt = [json.loads(line) for line in open(gt_file).readlines()]
        if len(json_gt) != len(json_pred):
            raise ValueError(f'got predictions for {len(json_pred)} of {len(json_gt)} test tasks')
        gts = {line['raw_file']: line for line in json_gt}
        accuracy, fp, fn = 0., 0., 0.
        for pred in json_pred:
            if pred['raw_file'] not in gts:
                raise ValueError(f'{pred["raw_file"]} is not a test task')
            gt = gts[pred['raw_file']]
            a, p, n = LaneEval.bench(pred['lanes'], gt['lanes'], gt['h_samples'], pred['run_time'])
            accuracy += a
            fp += p
            fn += n
        num = len(gts)
        return {'Accuracy': accuracy / num, 'FP': fp / num, 'FN': fn / num}


def generate_tusimple_lines(out, griding_num: int, orig_width: int = 1280):
    """Turn one image's logits (griding_num+1, rows, lanes) into per-lane x positions.

    For each row anchor and lane: if the argmax class is "no lane" -> -2 (TuSimple's marker);
    otherwise x is the softmax-weighted expected cell index (sub-cell precision), mapped from
    cell units to pixels of the raw frame.
    """
    out = out.detach().cpu().numpy()
    out_loc = np.argmax(out, axis=0)                       # rows, lanes: most likely cell
    prob = scipy.special.softmax(out[:-1, :, :], axis=0)   # over real cells only
    idx = np.arange(griding_num).reshape(-1, 1, 1)
    loc = np.sum(prob * idx, axis=0)                       # expected cell
    loc[out_loc == griding_num] = griding_num              # "no lane" wins the argmax
    lanes = []
    for i in range(loc.shape[1]):
        lanes.append([int(round((c + 0.5) * orig_width / (griding_num - 1))) if c != griding_num else -2
                      for c in loc[:, i]])
    return lanes


def predict_tusimple(net, loader, device, cfg, out_path) -> Path:
    """Run `net` over the test loader and write one JSON line per image (TuSimple submission format)."""
    net.eval()  # BatchNorm must use running statistics
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as fp, torch.no_grad():
        for imgs, names in tqdm(loader, desc='tusimple eval'):
            out = net(imgs.to(device))
            for i, name in enumerate(names):
                fp.write(json.dumps({
                    'lanes': generate_tusimple_lines(out[i], cfg.griding_num, cfg.orig_size[1]),
                    'h_samples': list(cfg.test_h_samples),
                    'raw_file': name,
                    'run_time': 10,  # as in the original: a placeholder under the benchmark's 200 ms cutoff
                }) + '\n')
    return out_path


def evaluate_tusimple(net, loader, device, cfg, work_dir) -> dict[str, float]:
    pred_file = predict_tusimple(net, loader, device, cfg, Path(work_dir) / 'tusimple_pred.json')
    return LaneEval.bench_one_submit(pred_file, Path(cfg.data_root) / 'test_label.json')

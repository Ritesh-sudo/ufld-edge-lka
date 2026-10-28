"""Training-time metrics (not the official TuSimple benchmark; see ufld/evaluation.py)."""
import numpy as np
import torch


def _flat(data):
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    return data.flatten()


class MultiLabelAcc:
    """Top-1: exact cell match."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.cnt = 0
        self.correct = 0

    def update(self, predict, target):
        predict, target = _flat(predict), _flat(target)
        self.cnt += len(predict)
        self.correct += np.sum(predict == target)

    def get(self):
        return self.correct * 1.0 / self.cnt


class AccTopk:
    """Prediction is right if within `k` cells; the "no lane" class must match exactly."""

    def __init__(self, background_classes: int, k: int):
        self.background_classes = background_classes
        self.k = k
        self.reset()

    def reset(self):
        self.cnt = 0
        self.correct = 0

    def update(self, predict, target):
        predict, target = _flat(predict), _flat(target)
        self.cnt += len(predict)
        background = (predict == self.background_classes) | (target == self.background_classes)
        self.correct += np.sum(predict[background] == target[background])
        lane = ~background
        self.correct += np.sum(np.absolute(predict[lane] - target[lane]) < self.k)

    def get(self):
        return self.correct * 1.0 / self.cnt


class MeanIoU:
    """Segmentation mIoU for the auxiliary branch."""

    def __init__(self, class_num: int):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.hist = np.zeros((self.class_num, self.class_num))

    def update(self, predict, target):
        predict, target = _flat(predict), _flat(target)
        hist = np.bincount(self.class_num * target.astype(int) + predict, minlength=self.class_num ** 2)
        self.hist += hist.reshape(self.class_num, self.class_num)

    def get(self):
        h = self.hist
        return np.nanmean(np.diag(h) / (h.sum(axis=1) + h.sum(axis=0) - np.diag(h)))

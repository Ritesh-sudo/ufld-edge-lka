"""Joint image+label augmentations. Ported unchanged in behavior from data/mytransforms.py.

They draw from the global `random` / `numpy.random` generators; `ufld.data.seed_worker`
and `ufld.utils.set_seed` make those reproducible.
"""
import random

import numpy as np
import torch
from PIL import Image


class Compose2:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class FreeScaleMask:
    """Nearest-neighbor resize of a label mask to `size` = (height, width)."""

    def __init__(self, size):
        self.size = size

    def __call__(self, mask):
        return mask.resize((self.size[1], self.size[0]), Image.NEAREST)


class MaskToTensor:
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


class RandomRotate:
    """Rotate image and mask by the same random angle in [-angle, angle] degrees."""

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, image, label):
        assert image.size == label.size
        angle = random.randint(0, self.angle * 2) - self.angle
        label = label.rotate(angle, resample=Image.NEAREST)
        image = image.rotate(angle, resample=Image.BILINEAR)
        return image, label


class RandomLROffsetLABEL:
    """Shift image and mask horizontally by a random offset, zero-filling the gap."""

    def __init__(self, max_offset):
        self.max_offset = max_offset

    def __call__(self, img, label):
        offset = np.random.randint(-self.max_offset, self.max_offset)
        w, h = img.size

        img = np.array(img)
        if offset > 0:
            img[:, offset:, :] = img[:, 0:w - offset, :]
            img[:, :offset, :] = 0
        if offset < 0:
            real_offset = -offset
            img[:, 0:w - real_offset, :] = img[:, real_offset:, :]
            img[:, w - real_offset:, :] = 0

        label = np.array(label)
        if offset > 0:
            label[:, offset:] = label[:, 0:w - offset]
            label[:, :offset] = 0
        if offset < 0:
            offset = -offset
            label[:, 0:w - offset] = label[:, offset:]
            label[:, w - offset:] = 0
        return Image.fromarray(img), Image.fromarray(label)


class RandomUDoffsetLABEL:
    """Shift image and mask vertically by a random offset, zero-filling the gap."""

    def __init__(self, max_offset):
        self.max_offset = max_offset

    def __call__(self, img, label):
        offset = np.random.randint(-self.max_offset, self.max_offset)
        w, h = img.size

        img = np.array(img)
        if offset > 0:
            img[offset:, :, :] = img[0:h - offset, :, :]
            img[:offset, :, :] = 0
        if offset < 0:
            real_offset = -offset
            img[0:h - real_offset, :, :] = img[real_offset:, :, :]
            img[h - real_offset:, :, :] = 0

        label = np.array(label)
        if offset > 0:
            label[offset:, :] = label[0:h - offset, :]
            label[:offset, :] = 0
        if offset < 0:
            offset = -offset
            label[0:h - offset, :] = label[offset:, :]
            label[h - offset:, :] = 0
        return Image.fromarray(img), Image.fromarray(label)


def find_start_pos(row_sample, start_line):
    """Index of the first row anchor at/after `start_line` (binary search over sorted anchors)."""
    l, r = 0, len(row_sample) - 1
    while True:
        mid = int((l + r) / 2)
        if r - l == 1:
            return r
        if row_sample[mid] < start_line:
            l = mid
        if row_sample[mid] > start_line:
            r = mid
        if row_sample[mid] == start_line:
            return mid

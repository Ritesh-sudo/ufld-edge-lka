"""TuSimple data loading: masks -> row-anchor classification labels.

Ported from data/dataset.py and data/dataloader.py. A training sample is
(image, cls_label[, seg_label]):
  * cls_label: (num_row_anchors, num_lanes) ints in [0, griding_num]; the column cell the lane
    passes through on that row, or `griding_num` for "no lane here".
  * seg_label: (aux_h, aux_w) lane-id mask for the auxiliary branch (0 = background).
"""
import os
import random

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

import ufld.transforms as mt

NORM_MEAN, NORM_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def _load(path):
    img = Image.open(path)
    img.load()
    return img


def _strip_slash(name: str) -> str:
    return name[1:] if name[0] == '/' else name


class LaneTestDataset(torch.utils.data.Dataset):
    """Images listed in a text file (first token of each line), no labels."""

    def __init__(self, path, list_path, img_transform=None):
        self.path = path
        self.img_transform = img_transform
        with open(list_path) as f:
            self.list = [_strip_slash(line) for line in f.readlines()]

    def __getitem__(self, index):
        name = self.list[index].split()[0]
        img = _load(os.path.join(self.path, name))
        if self.img_transform is not None:
            img = self.img_transform(img)
        return img, name

    def __len__(self):
        return len(self.list)


class LaneClsDataset(torch.utils.data.Dataset):
    def __init__(self, path, list_path, row_anchor, griding_num=100, num_lanes=4, use_aux=False,
                 img_transform=None, simu_transform=None, segment_transform=None):
        self.path = path
        self.griding_num = griding_num
        self.num_lanes = num_lanes
        self.use_aux = use_aux
        self.img_transform = img_transform
        self.simu_transform = simu_transform
        self.segment_transform = segment_transform
        # changed: copy instead of sorting the caller's list in place.
        self.row_anchor = sorted(row_anchor)
        # changed: replaces a `pdb.set_trace()` that fired if a scaled anchor row was -1.
        assert self.row_anchor[0] >= 0, 'row anchors must be non-negative'
        if use_aux:
            assert segment_transform is not None
        with open(list_path) as f:
            self.list = f.readlines()

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        info = self.list[index].split()
        img_name, label_name = info[0], info[1]
        if img_name[0] == '/':
            img_name, label_name = img_name[1:], label_name[1:]

        label = _load(os.path.join(self.path, label_name))
        img = _load(os.path.join(self.path, img_name))

        if self.simu_transform is not None:
            img, label = self.simu_transform(img, label)
        lane_pts = self._get_index(label)  # lane x-coordinates at the row anchors

        w, h = img.size
        cls_label = self._grid_pts(lane_pts, self.griding_num, w)  # coordinates -> class labels
        if self.use_aux:
            seg_label = self.segment_transform(label)

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.use_aux:
            return img, cls_label, seg_label
        return img, cls_label

    def _grid_pts(self, pts, num_cols, w):
        # pts: num_lane, n, 2 -> labels: n, num_lane
        num_lane, n, n2 = pts.shape
        col_sample = np.linspace(0, w - 1, num_cols)
        assert n2 == 2
        to_pts = np.zeros((n, num_lane))
        for i in range(num_lane):
            pti = pts[i, :, 1]
            to_pts[:, i] = np.asarray(
                [int(pt // (col_sample[1] - col_sample[0])) if pt != -1 else num_cols for pt in pti])
        return to_pts.astype(int)

    def _get_index(self, label):
        w, h = label.size

        # Anchors are defined for a 288-high image; the mask has its own height (720 for
        # TuSimple, 590 for CULane), so they are rescaled to it.
        # changed: the original only assigned `sample_tmp` inside `if h != 288`, so a 288-high
        # mask raised UnboundLocalError. Real TuSimple/CULane masks never hit that case, so
        # for them this is identical to the original; at h == 288 the anchors are used as-is.
        if h != 288:
            sample_tmp = [int((x * 1.0 / 288) * h) for x in self.row_anchor]
        else:
            sample_tmp = list(self.row_anchor)

        all_idx = np.zeros((self.num_lanes, len(sample_tmp), 2))
        mask = np.asarray(label)
        for i, r in enumerate(sample_tmp):
            label_r = mask[int(round(r))]
            for lane_idx in range(1, self.num_lanes + 1):
                pos = np.where(label_r == lane_idx)[0]
                if len(pos) == 0:
                    all_idx[lane_idx - 1, i, 0] = r
                    all_idx[lane_idx - 1, i, 1] = -1
                    continue
                all_idx[lane_idx - 1, i, 0] = r
                all_idx[lane_idx - 1, i, 1] = np.mean(pos)

        # Augmentation: extend each lane down to the image border with a linear fit of its lower half.
        all_idx_cp = all_idx.copy()
        for i in range(self.num_lanes):
            if np.all(all_idx_cp[i, :, 1] == -1):
                continue  # no lane

            valid = all_idx_cp[i, :, 1] != -1
            valid_idx = all_idx_cp[i, valid, :]
            if valid_idx[-1, 0] == all_idx_cp[0, -1, 0]:
                continue  # lane already reaches the last (bottom) row anchor
            if len(valid_idx) < 6:
                continue  # too short to fit

            valid_idx_half = valid_idx[len(valid_idx) // 2:, :]
            p = np.polyfit(valid_idx_half[:, 0], valid_idx_half[:, 1], deg=1)
            start_line = valid_idx_half[-1, 0]
            pos = mt.find_start_pos(all_idx_cp[i, :, 0], start_line) + 1

            fitted = np.polyval(p, all_idx_cp[i, pos:, 0])
            fitted = np.array([-1 if y < 0 or y > w - 1 else y for y in fitted])

            assert np.all(all_idx_cp[i, pos:, 1] == -1)
            all_idx_cp[i, pos:, 1] = fitted
        return all_idx_cp


def seed_worker(worker_id: int) -> None:
    """Seed `random`/`numpy` in each DataLoader worker from the (seeded) torch generator."""
    seed = torch.initial_seed() % 2 ** 32
    random.seed(seed)
    np.random.seed(seed)


def _img_transform(cfg):
    return T.Compose([T.Resize(cfg.input_size), T.ToTensor(), T.Normalize(NORM_MEAN, NORM_STD)])


def build_train_dataset(cfg) -> LaneClsDataset:
    # changed: sizes and augmentation strengths come from the config, not literals.
    return LaneClsDataset(
        cfg.data_root, os.path.join(cfg.data_root, 'train_gt.txt'),
        row_anchor=cfg.row_anchor, griding_num=cfg.griding_num, num_lanes=cfg.num_lanes, use_aux=cfg.use_aux,
        img_transform=_img_transform(cfg),
        simu_transform=mt.Compose2([
            mt.RandomRotate(cfg.rotate_deg),
            mt.RandomUDoffsetLABEL(cfg.ud_offset),
            mt.RandomLROffsetLABEL(cfg.lr_offset),
        ]),
        segment_transform=T.Compose([mt.FreeScaleMask(cfg.aux_size), mt.MaskToTensor()]),
    )


def build_train_loader(cfg, device, subset: int | None = None):
    dataset = build_train_dataset(cfg)
    if subset:
        dataset = torch.utils.data.Subset(dataset, range(min(subset, len(dataset))))
    gen = torch.Generator().manual_seed(cfg.seed)
    return torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, generator=gen,
        num_workers=cfg.num_workers, worker_init_fn=seed_worker,
        persistent_workers=cfg.num_workers > 0, pin_memory=device.type == 'cuda')


def build_test_loader(cfg, device, batch_size: int = 8, subset: int | None = None):
    dataset = LaneTestDataset(cfg.data_root, os.path.join(cfg.data_root, 'test.txt'), img_transform=_img_transform(cfg))
    if subset:
        dataset = torch.utils.data.Subset(dataset, range(min(subset, len(dataset))))
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=cfg.num_workers,
        pin_memory=device.type == 'cuda')

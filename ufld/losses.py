"""Training losses: classification, structural (similarity + shape) and auxiliary segmentation.

Ported from utils/loss.py and utils/factory.py. Logits have shape (N, C, R, L):
C = grid cells + 1 ("no lane"), R = row anchors, L = lanes.
"""
import torch
import torch.nn.functional as F
from torch import nn


class SoftmaxFocalLoss(nn.Module):
    """Focal loss over the column-cell classes: down-weights rows that are already easy."""

    def __init__(self, gamma: float, ignore_lb: int = 255):
        super().__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1. - scores, self.gamma)
        log_score = factor * F.log_softmax(logits, dim=1)
        return self.nll(log_score, labels)


class ParsingRelationLoss(nn.Module):
    """Similarity loss: logits of neighboring row anchors should be close (lanes are continuous)."""

    def forward(self, logits):
        # changed: the original looped over rows and concatenated; this takes all neighboring
        # row differences at once. The value is identical (same elements, same mean).
        diff = logits[:, :, :-1, :] - logits[:, :, 1:, :]
        return F.smooth_l1_loss(diff, torch.zeros_like(diff))


class ParsingRelationDis(nn.Module):
    """Shape loss: the lane's expected column should change at a constant rate row to row.

    Takes the soft-argmax column position per row, the first difference between rows (the
    local slope), and penalizes the change of that slope (second difference), i.e. lanes
    stay locally straight.
    """

    def forward(self, x):
        n, dim, num_rows, num_cols = x.shape
        prob = F.softmax(x[:, :dim - 1, :, :], dim=1)  # drop the "no lane" class
        embedding = torch.arange(dim - 1, dtype=prob.dtype, device=prob.device).view(1, -1, 1, 1)
        pos = (prob * embedding).sum(dim=1)  # n, rows, lanes: expected column index

        # Like the original, only the first half of the rows is used.
        k = num_rows // 2
        slope = pos[:, :k, :] - pos[:, 1:k + 1, :]
        # changed: vectorized form of the original per-row L1 loop; identical value.
        return (slope[:, :-1, :] - slope[:, 1:, :]).abs().mean()


class LaneLoss(nn.Module):
    """Weighted sum of the UFLD losses, built from a config.

    forward() takes the model outputs and labels, and returns (total, {name: value}).
    """

    def __init__(self, cfg):
        super().__init__()
        self.use_aux = cfg.use_aux
        self.sim_w = cfg.sim_loss_w
        self.shp_w = cfg.shp_loss_w
        self.cls = SoftmaxFocalLoss(2)
        self.sim = ParsingRelationLoss()
        self.shp = ParsingRelationDis()
        self.aux = nn.CrossEntropyLoss()

    def forward(self, cls_out, cls_label, seg_out=None, seg_label=None):
        parts = {'cls_loss': self.cls(cls_out, cls_label)}
        # a zero weight skips the term entirely (shape loss is off on TuSimple)
        if self.sim_w:
            parts['relation_loss'] = self.sim(cls_out)
        if self.shp_w:
            parts['relation_dis'] = self.shp(cls_out)
        if self.use_aux:
            parts['aux_loss'] = self.aux(seg_out, seg_label)
        weights = {'cls_loss': 1.0, 'relation_loss': self.sim_w, 'relation_dis': self.shp_w, 'aux_loss': 1.0}
        total = sum(weights[k] * v for k, v in parts.items())
        return total, parts

"""Golden tests: ported losses and metrics must equal the original's values."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ufld.losses import LaneLoss, ParsingRelationDis, ParsingRelationLoss, SoftmaxFocalLoss
from ufld.metrics import AccTopk, MeanIoU, MultiLabelAcc

N, C, R, L = 3, 101, 56, 4


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(N, C, R, L, generator=g) * 3
    labels = torch.randint(0, C, (N, R, L), generator=g)  # includes class 100 = "no lane"
    seg_logits = torch.randn(N, L + 1, 36, 100, generator=g)
    seg_labels = torch.randint(0, L + 1, (N, 36, 100), generator=g)
    return logits, labels, seg_logits, seg_labels


def test_focal_loss(ref):
    logits, labels, *_ = _batch()
    torch.testing.assert_close(SoftmaxFocalLoss(2)(logits, labels), ref.loss.SoftmaxFocalLoss(2)(logits, labels))


def test_similarity_loss(ref):
    logits, *_ = _batch()
    torch.testing.assert_close(ParsingRelationLoss()(logits), ref.loss.ParsingRelationLoss()(logits))


@pytest.mark.parametrize('rows', [56, 18, 7])
def test_shape_loss(ref, rows):
    logits = torch.randn(2, C, rows, L, generator=torch.Generator().manual_seed(3)) * 3
    torch.testing.assert_close(ParsingRelationDis()(logits), ref.loss.ParsingRelationDis()(logits))


@pytest.mark.parametrize('shp_w', [0.0, 0.5])
def test_total_loss_matches_reference_weighting(ref, shp_w):
    logits, labels, seg_logits, seg_labels = _batch()
    cfg = SimpleNamespace(use_aux=True, sim_loss_w=1.0, shp_loss_w=shp_w)
    total, parts = LaneLoss(cfg)(logits, labels, seg_logits, seg_labels)
    want = (ref.loss.SoftmaxFocalLoss(2)(logits, labels)
            + 1.0 * ref.loss.ParsingRelationLoss()(logits)
            + torch.nn.CrossEntropyLoss()(seg_logits, seg_labels)
            + shp_w * ref.loss.ParsingRelationDis()(logits))
    torch.testing.assert_close(total, want)
    assert ('relation_dis' in parts) == (shp_w != 0)


def test_losses_backward_and_cross_device():
    from ufld.device import get_device
    device = get_device()
    logits, labels, *_ = _batch()
    cpu = ParsingRelationDis()(logits) + ParsingRelationLoss()(logits) + SoftmaxFocalLoss(2)(logits, labels)
    x = logits.to(device).requires_grad_()
    on_dev = ParsingRelationDis()(x) + ParsingRelationLoss()(x) + SoftmaxFocalLoss(2)(x, labels.to(device))
    on_dev.backward()
    assert torch.isfinite(x.grad).all()
    torch.testing.assert_close(on_dev.cpu(), cpu, rtol=1e-4, atol=1e-5)


def test_metrics_match_reference(ref):
    rng = np.random.default_rng(0)
    pred, tgt = rng.integers(0, 101, 5000), rng.integers(0, 101, 5000)
    pred[::7] = 100
    seg_p, seg_t = rng.integers(0, 5, 4000), rng.integers(0, 5, 4000)
    for ours, theirs, a, b in [
        (MultiLabelAcc(), ref.metrics.MultiLabelAcc(), pred, tgt),
        (AccTopk(100, 3), ref.metrics.AccTopk(100, 3), pred, tgt),
        (MeanIoU(5), ref.metrics.Metric_mIoU(5), seg_p, seg_t),
    ]:
        ours.update(a, b)
        theirs.update(a, b)
        assert ours.get() == pytest.approx(theirs.get())

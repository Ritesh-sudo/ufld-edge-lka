"""CLRNet (ResNet-18 + FPN + CLRHead), inference only. Ported from clrnet/models/*.

Idea, contrasted with UFLD's row anchors: UFLD classifies, at ~56 fixed image rows, which of 100 column
cells holds a lane, from ONE global feature vector. CLRNet instead keeps 192 *lane priors* (each a line
defined by start point and angle) and *refines* them over three feature-pyramid levels: it samples features along
each prior's line, predicts offsets that bend the line toward the real lane, then re-samples on the finer
level with the improved line (cross-layer refinement). Deep, coarse features say "a lane is roughly here";
shallow, fine ones place it precisely. A final per-prior confidence plus lane-NMS yields the lanes. More compute
than UFLD, much better localization.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


@dataclass(frozen=True)
class CLRConfig:
    """Values from upstream configs/clrnet/clr_resnet18_tusimple.py."""
    img_w: int = 800
    img_h: int = 320
    ori_img_w: int = 1280
    ori_img_h: int = 720
    cut_height: int = 160          # top rows of the raw frame are cropped away before resizing
    num_points: int = 72
    num_priors: int = 192
    refine_layers: int = 3
    sample_points: int = 36
    fc_hidden_dim: int = 64
    num_fc: int = 2
    conf_threshold: float = 0.40
    nms_thres: float = 50
    max_lanes: int = 5


class ConvBNReLU(nn.Module):
    """What mmcv's ConvModule(norm_cfg=BN) is: conv (no bias) -> BN -> ReLU. Submodule names `conv`, `bn` match the checkpoint."""

    def __init__(self, cin, cout, k, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class ConvPlain(nn.Module):
    """mmcv ConvModule with no norm and no activation (used by the FPN): conv with bias."""

    def __init__(self, cin, cout, k, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, padding=padding)

    def forward(self, x):
        return self.conv(x)


class Backbone(nn.Module):
    """ResNet-18 trunk; returns the outputs of layer1..layer4. Key names (`model.conv1`, `model.layer1.0...`) match upstream."""

    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(weights=None)
        self.model.fc = nn.Identity()  # upstream has no classifier head; keeps the state dict key-compatible

    def forward(self, x):
        m = self.model
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        outs = []
        for layer in (m.layer1, m.layer2, m.layer3, m.layer4):
            x = layer(x)
            outs.append(x)
        return outs


class FPN(nn.Module):
    """Feature pyramid: 1x1 lateral convs, nearest-upsample top-down sums, 3x3 smoothing convs. Uses layer2..4 (stride 8/16/32)."""

    def __init__(self, in_channels=(128, 256, 512), out_channels=64):
        super().__init__()
        self.lateral_convs = nn.ModuleList(ConvPlain(c, out_channels, 1) for c in in_channels)
        self.fpn_convs = nn.ModuleList(ConvPlain(out_channels, out_channels, 3, padding=1) for _ in in_channels)

    def forward(self, feats):
        feats = feats[-len(self.lateral_convs):]                 # drop layer1
        lat = [l(f) for l, f in zip(self.lateral_convs, feats)]
        for i in range(len(lat) - 1, 0, -1):
            lat[i - 1] = lat[i - 1] + F.interpolate(lat[i], size=lat[i - 1].shape[2:], mode='nearest')
        return [c(l) for c, l in zip(self.fpn_convs, lat)]


class ROIGather(nn.Module):
    """Turns each prior's sampled line features into one vector, then adds global context via attention over the feature map."""

    def __init__(self, c, num_priors, sample_points, fc_hidden_dim, refine_layers, mid=48):
        super().__init__()
        self.c, self.num_priors = c, num_priors
        self.f_key = ConvBNReLU(c, c, 1)
        self.f_query = nn.Sequential(nn.Conv1d(num_priors, num_priors, 1, groups=num_priors), nn.ReLU())
        self.f_value = nn.Conv2d(c, c, 1)
        self.W = nn.Conv1d(num_priors, num_priors, 1, groups=num_priors)
        self.convs = nn.ModuleList(ConvBNReLU(c, mid, (9, 1), padding=(4, 0)) for _ in range(refine_layers))
        self.catconv = nn.ModuleList(ConvBNReLU(mid * (i + 1), c, (9, 1), padding=(4, 0)) for i in range(refine_layers))
        self.fc = nn.Linear(sample_points * fc_hidden_dim, fc_hidden_dim)
        self.fc_norm = nn.LayerNorm(fc_hidden_dim)

    @staticmethod
    def _resize(x):  # (B, C, H, W) -> (B, C, 10*25)
        return F.interpolate(x, (10, 25)).flatten(2)

    def forward(self, roi_features, x, layer_index):
        # roi_features: list over stages so far of (B*P, C, sample_points, 1); stage i uses convs[i], fused by catconv[layer_index]
        cat = self.catconv[layer_index](torch.cat([self.convs[i](f) for i, f in enumerate(roi_features)], dim=1))
        bs = x.size(0)
        roi = F.relu(self.fc_norm(self.fc(cat.contiguous().view(bs * self.num_priors, -1)))).view(bs, self.num_priors, -1)
        query = self.f_query(roi)
        key = self._resize(self.f_key(x))
        value = self._resize(self.f_value(x)).permute(0, 2, 1)
        sim = F.softmax((self.c ** -.5) * torch.matmul(query, key), dim=-1)
        return roi + self.W(torch.matmul(sim, value))              # dropout (p=0.1) is a no-op at inference


def _linear_relu(hidden, n):
    layers = []
    for _ in range(n):
        layers += [nn.Linear(hidden, hidden), nn.ReLU(inplace=True)]
    return nn.ModuleList(layers)


class CLRHead(nn.Module):
    def __init__(self, cfg: CLRConfig):
        super().__init__()
        self.cfg = cfg
        self.n_strips = cfg.num_points - 1
        self.n_offsets = cfg.num_points
        idx = (torch.linspace(0, 1, steps=cfg.sample_points) * self.n_strips).long()
        # non-persistent: pure functions of constants; the released checkpoint predates them and does not store them
        self.register_buffer('sample_x_indexs', idx, persistent=False)
        self.register_buffer('prior_feat_ys', torch.flip(1 - idx.float() / self.n_strips, dims=[-1]), persistent=False)
        self.register_buffer('prior_ys', torch.linspace(1, 0, steps=self.n_offsets), persistent=False)
        # Learned prior lines; values come from the checkpoint (loaded as buffers, as upstream stores them).
        self.register_buffer('priors', torch.zeros(cfg.num_priors, 6 + self.n_offsets))
        self.register_buffer('priors_on_featmap', torch.zeros(cfg.num_priors, cfg.sample_points))
        self.prior_embeddings = nn.Embedding(cfg.num_priors, 3)    # kept so the checkpoint loads strictly
        self.reg_modules = _linear_relu(cfg.fc_hidden_dim, cfg.num_fc)
        self.cls_modules = _linear_relu(cfg.fc_hidden_dim, cfg.num_fc)
        self.roi_gather = ROIGather(64, cfg.num_priors, cfg.sample_points, cfg.fc_hidden_dim, cfg.refine_layers)
        self.reg_layers = nn.Linear(cfg.fc_hidden_dim, self.n_offsets + 1 + 2 + 1)  # offsets + length + (start_y, start_x, theta)
        self.cls_layers = nn.Linear(cfg.fc_hidden_dim, 2)

    def pool_prior_features(self, feat, num_priors, prior_xs):
        """Sample `sample_points` features along each prior's line with bilinear grid_sample -> (B*P, C, S, 1)."""
        b = feat.shape[0]
        xs = prior_xs.view(b, num_priors, -1, 1) * 2. - 1.
        ys = self.prior_feat_ys.repeat(b * num_priors).view(b, num_priors, -1, 1) * 2. - 1.
        out = F.grid_sample(feat, torch.cat((xs, ys), dim=-1), align_corners=True).permute(0, 2, 1, 3)
        return out.reshape(b * num_priors, feat.shape[1], self.cfg.sample_points, 1)

    def _line_xs(self, p, b, n):
        """Normalized x of a prior at each of 72 rows, from its start (x, y) and angle theta."""
        t = lambda v: v.unsqueeze(2).repeat(1, 1, self.n_offsets)  # noqa: E731
        return (t(p[..., 3]) * (self.cfg.img_w - 1) +
                ((1 - self.prior_ys.repeat(b, n, 1) - t(p[..., 2])) * self.cfg.img_h /
                 torch.tan(t(p[..., 4]) * math.pi + 1e-5))) / (self.cfg.img_w - 1)

    def forward(self, feats):
        cfg = self.cfg
        feats = list(feats[len(feats) - cfg.refine_layers:])
        feats.reverse()                                            # coarse -> fine: stride 32, 16, 8
        b = feats[-1].shape[0]
        priors = self.priors.repeat(b, 1, 1)
        on_feat = self.priors_on_featmap.repeat(b, 1, 1)
        stages = []
        for stage in range(cfg.refine_layers):
            n = on_feat.shape[1]
            stages.append(self.pool_prior_features(feats[stage], n, torch.flip(on_feat, dims=[2])))
            fc = self.roi_gather(stages, feats[stage], stage).reshape(b * n, cfg.fc_hidden_dim)
            # changed: upstream does view(num_priors, batch, -1) first; identical at batch size 1, which is all we run.
            cls_f, reg_f = fc.clone(), fc.clone()
            for l in self.cls_modules:
                cls_f = l(cls_f)
            for l in self.reg_modules:
                reg_f = l(reg_f)
            cls_logits = self.cls_layers(cls_f).reshape(b, -1, 2)
            reg = self.reg_layers(reg_f).reshape(b, -1, self.reg_layers.out_features)

            pred = priors.clone()
            pred[:, :, :2] = cls_logits
            pred[:, :, 2:5] += reg[:, :, :3]                       # start_y, start_x, theta
            pred[:, :, 5] = reg[:, :, 3]                           # length
            pred[..., 6:] = self._line_xs(pred, b, n)
            lines = pred.clone()
            pred[..., 6:] += reg[..., 4:]                          # per-row x offsets refine the straight line
            if stage != cfg.refine_layers - 1:
                priors = lines.detach().clone()                    # next stage re-samples along this refined line
                on_feat = priors[..., 6 + self.sample_x_indexs]
        return pred                                                # last stage: (B, 192, 78)


class CLRNet(nn.Module):
    def __init__(self, cfg: CLRConfig = CLRConfig()):
        super().__init__()
        self.cfg = cfg
        self.backbone = Backbone()
        self.neck = FPN()
        self.heads = CLRHead(cfg)

    def forward(self, x):
        return self.heads(self.neck(self.backbone(x)))


def load_clrnet(ckpt: str, device: torch.device, cfg: CLRConfig = CLRConfig()) -> CLRNet:
    """Load the authors' checkpoint STRICTLY (after dropping the training-only seg decoder and the DataParallel prefix)."""
    sd = torch.load(ckpt, map_location='cpu', weights_only=True)['net']
    sd = {k.removeprefix('module.'): v for k, v in sd.items() if '.seg_decoder.' not in k}
    model = CLRNet(cfg)
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()


def preprocess_bgr(img_bgr: np.ndarray, cfg: CLRConfig = CLRConfig()) -> np.ndarray:
    """Raw BGR frame (as cv2 reads it) -> (1, 3, 320, 800) float32. Crop top rows, cubic resize, /255. No mean/std."""
    img = img_bgr[cfg.cut_height:]
    img = cv2.resize(img, (cfg.img_w, cfg.img_h), interpolation=cv2.INTER_CUBIC)
    return (img.astype(np.float32) / 255.).transpose(2, 0, 1)[None]

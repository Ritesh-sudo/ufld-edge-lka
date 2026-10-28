"""UFLD network: ResNet backbone -> row-anchor classification head (+ training-only aux branch).

Ported from model/model.py and model/backbone.py of the original repo. State-dict keys are
kept identical to the original so the authors' released checkpoints load unchanged.
"""
import torch
import torchvision
from torch import nn
from torchvision.models import ResNet18_Weights, ResNet34_Weights

# channels of the layer2 / layer3 / layer4 feature maps for the supported backbones
_RESNET = {
    '18': (torchvision.models.resnet18, ResNet18_Weights.IMAGENET1K_V1),
    '34': (torchvision.models.resnet34, ResNet34_Weights.IMAGENET1K_V1),
}
_C2, _C3, _C4 = 128, 256, 512
_POOL_CH = 8        # channels after the 1x1 reduction in front of the classifier
_STRIDE = 32        # layer4 downsampling factor of the input


class ResNetBackbone(nn.Module):
    """ResNet trunk returning the layer2, layer3 and layer4 feature maps."""

    def __init__(self, layers: str, pretrained: bool = False):
        super().__init__()
        build, weights = _RESNET[layers]
        # changed: `pretrained=` is deprecated in torchvision; ImageNet weights are now
        # requested via the `weights` enum (IMAGENET1K_V1 is what pretrained=True loaded).
        net = build(weights=weights if pretrained else None)
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x2, x3, x4


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ParsingNet(nn.Module):
    """Row-anchor lane detector.

    The classifier head turns the whole (reduced) feature map into one score vector per
    (row anchor, lane): `cls_dim = (num_grid_cells + 1, num_row_anchors, num_lanes)`, where
    the extra class means "no lane on this row". This is classification over columns at fixed
    rows, instead of per-pixel segmentation, which is why the model is fast.

    With `use_aux=True` a segmentation branch is also built from the three backbone stages.
    It only supplies an extra training loss; build the model with `use_aux=False` for
    inference and it costs nothing (the branch does not exist).
    """

    def __init__(self, size=(288, 800), backbone='18', pretrained=False,
                 cls_dim=(101, 56, 4), use_aux=False):
        super().__init__()
        assert backbone in _RESNET, f'backbone must be one of {tuple(_RESNET)}'
        assert size[0] % _STRIDE == 0 and size[1] % _STRIDE == 0
        self.size = size
        self.cls_dim = cls_dim  # (num_gridding + 1, num_row_anchors, num_lanes)
        self.use_aux = use_aux
        self.total_dim = cls_dim[0] * cls_dim[1] * cls_dim[2]

        self.model = ResNetBackbone(backbone, pretrained=pretrained)

        if use_aux:
            self.aux_header2 = nn.Sequential(
                ConvBnRelu(_C2, 128, 3, stride=1, padding=1),
                ConvBnRelu(128, 128, 3, padding=1),
                ConvBnRelu(128, 128, 3, padding=1),
                ConvBnRelu(128, 128, 3, padding=1),
            )
            self.aux_header3 = nn.Sequential(
                ConvBnRelu(_C3, 128, 3, stride=1, padding=1),
                ConvBnRelu(128, 128, 3, padding=1),
                ConvBnRelu(128, 128, 3, padding=1),
            )
            self.aux_header4 = nn.Sequential(
                ConvBnRelu(_C4, 128, 3, stride=1, padding=1),
                ConvBnRelu(128, 128, 3, padding=1),
            )
            self.aux_combine = nn.Sequential(
                ConvBnRelu(384, 256, 3, padding=2, dilation=2),
                ConvBnRelu(256, 128, 3, padding=2, dilation=2),
                ConvBnRelu(128, 128, 3, padding=2, dilation=2),
                ConvBnRelu(128, 128, 3, padding=4, dilation=4),
                nn.Conv2d(128, cls_dim[-1] + 1, 1),
                # output: n, num_lanes + 1, h/8, w/8  (36x100 for a 288x800 input)
            )
            _init_weights(self.aux_header2, self.aux_header3, self.aux_header4, self.aux_combine)

        # changed: the original hardcoded Linear(1800, ...) = 8 * 9 * 25 for 288x800.
        fea_dim = _POOL_CH * (size[0] // _STRIDE) * (size[1] // _STRIDE)
        self.fea_dim = fea_dim
        self.cls = nn.Sequential(
            nn.Linear(fea_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.total_dim),
        )
        self.pool = nn.Conv2d(_C4, _POOL_CH, 1)
        _init_weights(self.cls)

    def forward(self, x):
        x2, x3, fea = self.model(x)
        aux_seg = None
        if self.use_aux:
            x2 = self.aux_header2(x2)
            x3 = nn.functional.interpolate(self.aux_header3(x3), scale_factor=2, mode='bilinear')
            x4 = nn.functional.interpolate(self.aux_header4(fea), scale_factor=4, mode='bilinear')
            aux_seg = self.aux_combine(torch.cat([x2, x3, x4], dim=1))

        fea = self.pool(fea).view(-1, self.fea_dim)
        group_cls = self.cls(fea).view(-1, *self.cls_dim)

        if self.use_aux:
            return group_cls, aux_seg
        return group_cls


def _init_weights(*modules):
    for m in modules:
        _init_one(m)


def _init_one(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, std=0.01)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Module):
        for child in m.children():
            _init_one(child)


def build_model(cfg, pretrained: bool = False, use_aux: bool | None = None) -> ParsingNet:
    """Build the network described by a config. `use_aux` overrides cfg.use_aux (e.g. False for eval)."""
    return ParsingNet(
        size=cfg.input_size,
        backbone=cfg.backbone,
        pretrained=pretrained,
        cls_dim=(cfg.griding_num + 1, cfg.cls_num_per_lane, cfg.num_lanes),
        use_aux=cfg.use_aux if use_aux is None else use_aux,
    )

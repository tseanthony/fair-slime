import torch
import torch.nn as nn
import torchvision.models.resnet as resnet

from sslime.utils.utils import Flatten, parse_out_keys_arg
from .resnet50 import BottleneckV2


class BottleneckRev(BottleneckV2):
    def __init__(self, inplanes):
        super(BottleneckRev, self).__init__(inplanes // 2, inplanes // 8)

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, 1)
        y1 = x1 + super(BottleneckRev, self).forward(x2, no_shortcut=True)
        y2 = x2
        return torch.cat([y2, y1], 1)


class RevNet(nn.Module):
    def __init__(self, inplanes=64*4, k=1, layers=[3, 4, 6, 3], strides=[2, 2, 2, 1]):
        inplanes *= k
        super(RevNet, self).__init__()
        self.conv1 = nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        for i in range(len(layers)):
            layer = self._make_layer(inplanes, layers[i], strides[i])
            setattr(self, 'layer{}'.format(i+1), layer)
            inplanes *= 2

        self.avgpool = nn.Sequential(
            nn.BatchNorm2d(inplanes // 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self._feature_blocks = nn.ModuleList(
            [
                self.conv1,
                self.maxpool,
                self.layer1,
                self.layer2,
                self.layer3,
                self.layer4,
                self.avgpool,
                Flatten(1),
            ]
        )

        self.all_feat_names = [
            "conv1",
            "res1",
            "res2",
            "res3",
            "res4",
            "res5",
            "res5avg",
            "flatten",
        ]

        assert len(self.all_feat_names) == len(self._feature_blocks)

    def _make_layer(self, inplanes, nblocks, stride):
        blocks = []
        for _ in range(nblocks):
            blocks.append(BottleneckRev(inplanes))
        if stride > 1:
            # blocks.append(nn.AvgPool2d(stride, stride))
            blocks.append(nn.Conv2d(inplanes, inplanes, kernel_size=1, stride=stride, bias=False))
            blocks.append(nn.BatchNorm2d(inplanes))
            padding = (0, 0, 0, 0, inplanes // 2, inplanes // 2)
            blocks.append(nn.ConstantPad3d(padding, 0))
        return nn.Sequential(*blocks)

    def forward(self, x, out_feat_keys=None):
        out_feat_keys, max_out_feat = parse_out_keys_arg(
            out_feat_keys, self.all_feat_names
        )
        out_feats = [None] * len(out_feat_keys)

        feat = x
        for f in range(max_out_feat + 1):
            feat = self._feature_blocks[f](feat)
            key = self.all_feat_names[f]
            if key in out_feat_keys:
                out_feats[out_feat_keys.index(key)] = feat

        return out_feats

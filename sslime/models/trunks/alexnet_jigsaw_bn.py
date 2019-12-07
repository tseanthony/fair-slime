
# input size are 64 x 64 tiles from a 75 x 75 grid

import torch
import torch.nn as nn

from sslime.core.config import config as cfg
from sslime.utils.utils import Flatten, parse_out_keys_arg

# Referenced https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
class AlexNet_BN(nn.Module):
    def __init__(self):
        super(AlexNet_BN, self).__init__()

        conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        conv2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )
        pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        conv3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )
        conv4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        flatten = Flatten(1)

        self._feature_blocks = nn.ModuleList(
            [conv1, pool1, conv2, pool2, conv3, conv4, conv5, pool5, flatten]
        )
        self.all_feat_names = [
            "conv1",
            "pool1",
            "conv2",
            "pool2",
            "conv3",
            "conv4",
            "conv5",
            "pool5",
            "flatten",
        ]
        assert len(self.all_feat_names) == len(self._feature_blocks)

    def forward(self, x, out_feat_keys=None):
        """Forward an image `x` through the network and return the asked output features.

        Args:
          x: input image.
          out_feat_keys: a list/tuple with the feature names of the features
                that the function should return. By default the last feature of
                the network is returned.

        Return:
            out_feats: If multiple output features were asked then `out_feats`
                is a list with the asked output features placed in the same
                order as in `out_feat_keys`. If a single output feature was
                asked then `out_feats` is that output feature (and not a list).
        """
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

class Jigsaw_BN(AlexNet_BN):
    def __init__(self):
        super(Jigsaw_BN, self).__init__()

        # Pretext Training: stride of the first layer of cfn is 2 instead of 4, change fc dimensions
        if not cfg.MODEL.FEATURE_EVAL_MODE:
            self._feature_blocks[0][0].stride = (2,2)
            self.fc6 = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(512),
            )

    def forward(self, x, out_feat_keys=None):
        """
        Override forward function of standard AlexNet
        """

        feat = x

        if cfg.MODEL.FEATURE_EVAL_MODE:
            # Evaluation of SSL features

            out_feat_keys, max_out_feat = parse_out_keys_arg(
                out_feat_keys, self.all_feat_names
            )
            out_feats = [None] * len(out_feat_keys)

            for f in range(max_out_feat + 1):
                feat = self._feature_blocks[f](feat)
                key = self.all_feat_names[f]
                if key in out_feat_keys:
                    out_feats[out_feat_keys.index(key)] = feat
            return out_feats

        else:
            # Pretext Training: process each puzzle piece through model

            batch_dim, jigsaw_dim, _, _, _ = feat.size()
            feat = feat.transpose(0, 1)

            output = []
            for i in range(9):
                jigsaw_piece = feat[i]
                for layer in self._feature_blocks:
                    jigsaw_piece = layer(jigsaw_piece)
                jigsaw_piece = self.fc6(jigsaw_piece)
                output.append(jigsaw_piece)
            output = torch.cat(output, dim=1)
            return [output]


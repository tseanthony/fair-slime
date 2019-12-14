#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from sslime.models.trunks.alexnet_rotnet import AlexNet_RotNet
from sslime.models.trunks.resnet50 import ResNet50
from sslime.models.trunks.resnet50 import ResNet50V2
from sslime.models.trunks.revnet50 import RevNet
from sslime.models.trunks.vgg_a import VGG_A
from sslime.models.trunks.alexnet_jigsaw import Jigsaw
from sslime.models.trunks.alexnet_jigsaw_bn import Jigsaw_BN
from sslime.models.trunks.cfn import CFN

TRUNKS = {"alexnet": AlexNet_RotNet,
          "resnet50": ResNet50,
          "resnet50v2": ResNet50V2,
          "revnet50": RevNet,
          "vgg_a": VGG_A,
          "cfn": CFN,
          "alexnet_jigsaw": Jigsaw,
          "alexnet_jigsaw_bn": Jigsaw_BN}

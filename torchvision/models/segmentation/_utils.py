from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F


class _SimpleSegmentationModel(nn.Module):
    __constants__ = ['aux_classifier']

    def __init__(self, backbone, classifier, aux_classifier=None):
        super(_SimpleSegmentationModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result["aux"] = x

        return result


class _DeepLabV3PlusModel(nn.Module):

    def __init__(self, backbone, classifier):
        super(_DeepLabV3PlusModel, self).__init__()
        self.dilation = 1
        self.backbone = backbone
        self.classifier = classifier
        self._init_weight()

    def train(self, mode=True):
        super(_DeepLabV3PlusModel, self).train(mode)
        self._fix_dilation(16)
        self.classifier.set_output_stride(16)
        return self

    def eval(self):
        super(_DeepLabV3PlusModel, self).eval()
        self._fix_dilation(8)
        self.classifier.set_output_stride(8)
        return self.train(False)

    def _fix_dilation(self, output_stride=16):
        self._fix_layer_dilation(self.backbone.layer3, stride=2, dilate=(output_stride == 8))
        self._fix_layer_dilation(self.backbone.layer4, stride=2, dilate=True)

    def _fix_layer_dilation(self, layer, stride=1, dilate=False):
        if dilate:
            self.dilation *= stride
        for l in range(1, layer):
            layer[l].conv2.dilation = self.dilation

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)

        result = OrderedDict()
        result["out"] = self.classifier(features, input_shape)

        return result

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

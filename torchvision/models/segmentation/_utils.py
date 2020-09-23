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
        self.backbone = backbone
        self.classifier = classifier

    def train(self):
        super(_DeepLabV3PlusModel, self).train()
        self.classifier.set_output_stride(16)

    def eval(self):
        super(_DeepLabV3PlusModel, self).eval()
        self.classifier.set_output_stride(8)

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)

        result = OrderedDict()
        result["out"] = self.classifier(features, input_shape)

        return result

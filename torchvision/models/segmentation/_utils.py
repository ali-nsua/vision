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

    def __init__(self, backbone, classifier, num_classes, llsize=256):
        super(_DeepLabV3PlusModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.low_level_module = nn.Sequential(nn.Conv2d(llsize, 21, 1, bias=False),
                                              nn.BatchNorm2d(21),
                                              nn.ReLU())
        self.final_emb = nn.Sequential(nn.Conv2d(42, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)

        x, low_level_features = features["out"], features["low_level_features"]
        low_level_features = self.low_level_module(low_level_features)
        x = self.classifier(x)
        x = F.interpolate(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_features), dim=1)
        x = self.final_emb(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)

        return x

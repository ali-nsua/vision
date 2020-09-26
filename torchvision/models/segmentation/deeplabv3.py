import torch
from torch import nn
from torch.nn import functional as F

from ._utils import _SimpleSegmentationModel


__all__ = ["DeepLabV3", "DeepLabV3Plus"]


class DeepLabV3Plus(_SimpleSegmentationModel):
    """
    Implements DeepLabV3+ model from
    `"Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1802.02611>`_.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    def __init__(self, backbone, classifier, aux_classifier=None):
        super(DeepLabV3Plus, self).__init__(backbone, classifier, aux_classifier, True)
        self.dilation = 1

    def train(self, mode=True):
        super(DeepLabV3Plus, self).train(mode)
        self._set_output_stride(16)
        return self

    def eval(self):
        super(DeepLabV3Plus, self).eval()
        self._set_output_stride(8)
        return self.train(False)

    def _set_output_stride(self, output_stride=16):
        dilation_rates = [6, 12, 18] if output_stride == 16 else [12, 24, 36]

        for i in range(len(dilation_rates)):
            self.classifier.aspp.convs[i + 1][0].dilation = (dilation_rates[i], dilation_rates[i])

        self.dilation = 1
        self._set_resnet_layer_dilation(self.backbone.layer3, stride=2, dilate=(output_stride == 8))
        self._set_resnet_layer_dilation(self.backbone.layer4, stride=2, dilate=True)

    def _set_resnet_layer_dilation(self, layer, stride=1, dilate=False):
        if dilate:
            self.dilation *= stride
            stride = 1
        stride = (stride, stride)
        layer[0].conv2.stride = stride
        layer[0].stride = stride
        if layer[0].downsample is not None:
            layer[0].downsample[0].stride = stride
        for i in range(1, len(layer)):
            layer[i].conv2.dilation = (self.dilation, self.dilation)


class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )


class DeepLabPlusHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DeepLabPlusHead, self).__init__()
        # default output stride is 16 (training)
        atrous_rates = [6, 12, 18]  # os = 16
        # atrous_rates = [12, 24, 36] # os = 8

        self.aspp = ASPP(in_channels, atrous_rates)
        # Note: low_level_module takes the level1 output from resnet,
        # which is why the input n_channels is set to 256.
        self.low_level_project = nn.Sequential(nn.Conv2d(256, 48, 1, bias=False),
                                               nn.BatchNorm2d(48),
                                               nn.ReLU())
        self.project = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(),
                                     nn.Dropout(0.1),
                                     nn.Conv2d(256, num_classes, kernel_size=1, stride=1))

    def forward(self, input_features):
        x, x_low = input_features
        low_level_features = self.low_level_project(x_low)
        x = self.aspp(x)
        x = F.interpolate(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_features), dim=1)
        x = self.project(x)

        return x


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

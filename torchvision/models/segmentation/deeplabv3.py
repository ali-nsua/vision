import torch
from torch import nn
from torch.nn import functional as F

from ._utils import _SimpleSegmentationModel, _DeepLabV3PlusModel


__all__ = ["DeepLabV3", "DeepLabV3Plus"]


class DeepLabV3Plus(_DeepLabV3PlusModel):
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
    """
    pass


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
    def __init__(self, in_channels, output_stride=16, num_classes=21, llsize=256):
        super(DeepLabPlusHead, self).__init__()
        atrous_rates = [6, 12, 18]
        if output_stride == 16:
            pass
        elif output_stride == 8:
            atrous_rates = [12, 24, 36]

        self.aspp = ASPP(in_channels, atrous_rates)
        self.low_level_module = nn.Sequential(nn.Conv2d(llsize, 48, 1, bias=False),
                                              nn.BatchNorm2d(48),
                                              nn.ReLU())
        self.final_embedding = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                             nn.BatchNorm2d(256),
                                             nn.ReLU(),
                                             nn.Dropout(0.1),
                                             nn.Conv2d(256, num_classes, kernel_size=1, stride=1))

    def set_output_stride(self, output_stride=16):
        atrous_rates = [6, 12, 18]
        if output_stride == 16:
            pass
        elif output_stride == 8:
            atrous_rates = [12, 24, 36]

        for i in range(1, len(atrous_rates) + 1):
            self.aspp.convs[i][0].dilation = (atrous_rates[i - 1], atrous_rates[i - 1])

    def forward(self, features, input_shape):
        x, low_level_features = features["out"], features["low_level_features"]
        low_level_features = self.low_level_module(low_level_features)
        x = self.aspp(x)
        x = F.interpolate(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_features), dim=1)
        x = self.final_embedding(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)

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

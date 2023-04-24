import torch
import torch.nn as nn
import torch.nn.functional as F
from encoding.nn import SegmentationLosses, SyncBatchNorm

import encoding
from encoding.models import BaseNet

# from .fcn import FCNHead
# from ..nn import PyramidPooling
# from .LadderNetv66_small import Decoder, BasicBlock, LadderBlock
# from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveAvgPool2d, \
#     NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter
# # infer number of classes
from ..datasets import datasets


def get_channels_list(backbone):
    if "fpn" in backbone:
        return [256, 256]
    elif "resnet18" in backbone:
        return [128, 256]
    elif "res" in backbone:
        return [256 * 2, 256 * 2 ** 3]
    elif "mobile_net_v3" in backbone:
        return [40, 160]
    elif "inception" in backbone:
        return [384, 1536]


class MobileNetV3head(BaseNet):
    def __init__(
        self,
        nclass,
        backbone,
        aux=True,
        se_loss=True,
        lateral=False,
        norm_layer=SyncBatchNorm,
        dilated=False,
        **kwargs
    ):
        super(MobileNetV3head, self).__init__(
            nclass,
            backbone,
            aux,
            se_loss,
            norm_layer=norm_layer,
            dilated=dilated,
            **kwargs
        )

        in_channels_2, in_channels_1 = get_channels_list(backbone)

        self.latlayer1a = nn.Conv2d(
            in_channels_1, 128, kernel_size=1, stride=1, padding=0
        )
        self.bn1a = norm_layer(128)

        # self.avg_pool1b = nn.AvgPool2d(49, [16, 20])
        self.latlayer1b = nn.Conv2d(
            in_channels_1, 128, kernel_size=1, stride=1, padding=0
        )
        self.latlayer1c = nn.Conv2d(128, nclass, kernel_size=1, stride=1, padding=0)

        self.latlayer2a = nn.Conv2d(
            in_channels_2, nclass, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        imsize = x.size()[2:]
        features = self.base_forward(x)  # return 4 features from restnet backbone
        out1a = self.latlayer1a(features[-1])
        out1a = self.bn1a(out1a)
        out1a = F.relu(out1a)

        # out1b = self.avg_pool1b(features[-1])
        out1b = F.avg_pool2d(features[-1], kernel_size=9, stride=(5, 5))

        out1b = self.latlayer1b(out1b)
        out1b = F.sigmoid(out1b)
        out1b = F.upsample(
            out1b,
            (imsize[0] // 16, imsize[1] // 16),
            mode="bilinear",
            align_corners=False,
        )

        out1c = out1a * out1b
        out1c = self.latlayer1c(out1c)
        out1c = F.upsample(
            out1c,
            (imsize[0] // 8, imsize[1] // 8),
            mode="bilinear",
            align_corners=False,
        )

        out2a = self.latlayer2a(features[1])
        out2a += out1c

        out2a = F.upsample(out2a, imsize, mode="bilinear", align_corners=False)

        return [out2a]


def get_mobile_net_v3_head(
    dataset="pascal_voc",
    backbone="resnet50",
    pretrained=False,
    root="~/.encoding/models",
    **kwargs
):
    r"""EncNet model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    dataset : str, default pascal_voc
        The dataset that model pretrained on. (pascal_voc, ade20k)
    backbone : str, default resnet50
        The backbone network. (resnet50, 101, 152)
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_encnet(dataset='pascal_voc', backbone='resnet50', pretrained=False)
    >>> print(model)
    """
    acronyms = {"pascal_voc": "voc", "ade20k": "ade", "pcontext": "pcontext"}
    kwargs["lateral"] = True if dataset.lower() == "pcontext" else False

    model = MobileNetV3head(
        datasets[dataset.lower()].NUM_CLASS,
        backbone=backbone,
        root=root,
        dilate_only_last_year=True,
        pretrained=False,
        dilated=True,
        **kwargs
    )
    if pretrained:
        from .model_store import get_model_file

        model.load_state_dict(torch.load(get_model_file(backbone, root=root)))
    return model


if __name__ == "__main__":
    from encoding.parallel import DataParallelModel, DataParallelCriterion

    A = torch.rand(1, 3, 1024, 1024)
    net = DataParallelModel(
        MobileNetV3head(
            10, "resnet18", dilate_only_last_year=True, pretrained=False, dilated=True
        )
    )
    out = net(A)
    for o in out:
        print(o.shape)

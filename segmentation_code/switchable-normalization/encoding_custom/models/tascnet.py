import torch
import torch.nn as nn
import torch.nn.functional as F
from encoding_custom.datasets import datasets
from encoding_custom import backbones
from encoding_custom.models import FPN
from encoding_custom.models.base import BaseNet
from encoding_custom.models.deeplab import ASPP_Module


class TascNet(BaseNet):
    def __init__(
        self,
        num_classes,
        backbone=None,
        aux=False,
        se_loss=True,
        lateral=False,
        norm_layer=None,
        dilated=False,
        use_aspp=False,
        **kwargs
    ):
        super(TascNet, self).__init__(
            num_classes,
            backbone,
            aux,
            se_loss,
            norm_layer=norm_layer,
            dilated=dilated,
            **kwargs
        )

        if backbone == "fpn_resnet18":
            self.pretrained = FPN(
                backbones.resnet18(
                    pretrained=True,
                    norm_layer=norm_layer,
                    dilated=False,
                    add_additional_layers=True,
                ),
                has_block_expansion=False,
            )
        elif backbone == "fpn_resnet50":
            self.pretrained = FPN(
                backbones.resnet50(
                    pretrained=True,
                    norm_layer=norm_layer,
                    dilated=False,
                    add_additional_layers=True,
                )
            )

        elif backbone == "fpn_resnet101":
            self.pretrained = FPN(
                backbones.resnet101(
                    pretrained=True,
                    norm_layer=norm_layer,
                    dilated=False,
                    add_additional_layers=True,
                )
            )

        # self.pretrained = FPN(backbones.resnet50(pretrained=True, dilated=False, add_additional_layers=True))

        self.head = _TascNet(
            self.pretrained.total_outputs,
            num_classes,
            norm_layer,
            self._up_kwargs,
            use_aspp,
        )

    def forward(self, x):
        features = self.pretrained(x)
        return self.head(features)


class _TascNet(nn.Module):
    def __init__(self, total_outputs, num_classes, norm_layer, up_kwargs, use_aspp):
        super(_TascNet, self).__init__()

        self.stuff_heads = []
        for _ in range(total_outputs):
            self.stuff_heads.append(self._make_stuff_head())
        self.stuff_heads = nn.Sequential(
            *self.stuff_heads
        )  # needed to make it cuda compliant. (pytorch wart)

        if use_aspp:
            self.conv = ASPP_Module(
                total_outputs * 128,
                [6, 12, 18],
                norm_layer,
                up_kwargs=up_kwargs,
                out_channels=num_classes,
            )
        else:
            self.conv = nn.Conv2d(total_outputs * 128, num_classes, kernel_size=1)

    def _make_stuff_head(self):
        return nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(16, 128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
        )

    def forward(self, x):
        fpn_outs = x
        for i in range(len(fpn_outs)):
            fpn_outs[i] = self.stuff_heads[i](fpn_outs[i])

        _, _, H, W = fpn_outs[0].size()
        for i in range(1, len(fpn_outs)):
            fpn_outs[i] = F.upsample(fpn_outs[i], size=(H, W), mode="bilinear")

        output = torch.cat(fpn_outs, dim=1)
        output = self.conv(output)
        output = F.upsample(output, scale_factor=4, mode="bilinear")

        return tuple([output])


# todo: clean up this method
def get_tascnet(
    dataset="pascal_voc",
    backbone="resnet50",
    pretrained=False,
    root="~/.encoding/models",
    **kwargs
):
    acronyms = {"pascal_voc": "voc", "pascal_aug": "voc", "ade20k": "ade"}
    # infer number of classes
    model = TascNet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, **kwargs)

    # if pretrained:
    #     from .model_store import get_model_file
    #     model.load_state_dict(torch.load(
    #         get_model_file('deeplab_%s_%s' % (backbone, acronyms[dataset]), root=root)))
    return model


def get_tascnet_v2(
    dataset="pascal_voc",
    backbone="resnet50",
    pretrained=False,
    root="~/.encoding/models",
    **kwargs
):
    acronyms = {"pascal_voc": "voc", "pascal_aug": "voc", "ade20k": "ade"}
    # infer number of classes
    model = TascNet(
        datasets[dataset.lower()].NUM_CLASS, backbone=backbone, use_aspp=True, **kwargs
    )

    # if pretrained:
    #     from .model_store import get_model_file
    #     model.load_state_dict(torch.load(
    #         get_model_file('deeplab_%s_%s' % (backbone, acronyms[dataset]), root=root)))
    return model


if __name__ == "__main__":
    A = torch.rand(2, 3, 1024, 1024).cuda()
    net = TascNet(66).cuda()
    out = net(A)
    print(out.shape)

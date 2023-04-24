import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding_custom.models.base import BaseNet
from encoding_custom.models.LadderNetv66_small import Decoder, LadderBlock
from encoding.nn import SyncBatchNorm
from encoding_custom.models.fcn import FCNHead
from encoding_custom.models.dfanet import fcattention
from encoding_custom.datasets import datasets
from encoding_custom.models.danet import DANetHead


def get_channels_list(backbone):
    # if "fpn" in backbone:
    #     return [256, 256, 256, 256]
    if "resnet_18" in backbone:
        return [64, 128, 256, 512]
    if "resnet" in backbone:
        return [256, 256 * 2, 256 * 2 ** 2, 256 * 2 ** 3]
    elif "inception" in backbone:
        return [192, 384, 1024, 1536]


def get_multinet(
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

    model = MultiNet(
        datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs
    )
    if pretrained:
        from .model_store import get_model_file

        model.load_state_dict(torch.load(get_model_file(backbone, root=root)))
    return model


class MultiNet(BaseNet):
    def __init__(
        self,
        nclass,
        backbone,
        aux=False,
        se_loss=False,
        lateral=False,
        norm_layer=SyncBatchNorm,
        dilated=False,
        use_fpn=False,
        add_danet=False,
        **kwargs
    ):
        super(MultiNet, self).__init__(
            nclass,
            backbone,
            aux,
            se_loss,
            norm_layer=norm_layer,
            dilated=dilated,
            **kwargs
        )

        base_outchannels = 72
        self.use_fpn = use_fpn
        self.add_danet = add_danet

        if self.use_fpn:
            channels_list = [256, 256, 256, 256]
        else:
            channels_list = [256, 256 * 2, 256 * 2 ** 2, 256 * 2 ** 3]

        self.head = nn.ModuleList()

        self.head.append(
            LadderHead(
                base_inchannels=channels_list,
                base_outchannels=base_outchannels,
                out_channels=nclass,
                norm_layer=norm_layer,
            )
        )

        self.head.append(
            TascNetHead(
                total_outputs=self.pretrained.total_outputs
                if hasattr(self.pretrained, "total_outputs")
                else 4,
                num_classes=nclass,
                norm_layer=norm_layer,
            )
        )

        self.aux_indexes = []
        self.num_outputs = 2

        if self.add_danet:
            self.head.append(DANetHead(256 * 2 ** 3, nclass, norm_layer))
            self.aux_indexes.extend([3, 4])
            self.num_outputs += 3
        if aux:
            self.auxlayer = FCNHead(256 * 2 ** 3, nclass, norm_layer=norm_layer)
            self.aux_indexes.append(self.num_outputs)
            self.num_outputs += 1

    def forward(self, x):

        imsize = x.size()[2:]

        base_outs, features = self.pretrained.forward_all(x)

        if self.use_fpn:
            x = self.head[0](features)
        else:
            x = self.head[0](base_outs)
        x2 = self.head[1](features)

        x = F.upsample(x, imsize, mode="bilinear")
        x2 = F.upsample(x2, imsize, mode="bilinear")

        outputs_train = [x]
        outputs_train.append(x2)

        if self.add_danet:
            x3, x4, x5 = self.head[2](base_outs[-1])
            x3 = F.upsample(x3, imsize, mode="bilinear")
            x4 = F.upsample(x4, imsize, mode="bilinear")
            x5 = F.upsample(x5, imsize, mode="bilinear")

            outputs_train.append(x3)
            outputs_train.append(x4)
            outputs_train.append(x5)
        if hasattr(self, "auxlayer"):
            x6 = self.auxlayer(base_outs[-1])
            x6 = F.upsample(x6, imsize, mode="bilinear")
            outputs_train.append(x6)

        if not self.training:
            # tensor_sum = outputs_train[0]
            # for i in outputs_train[1:3]:
            #     tensor_sum += i
            # return tuple([tensor_sum])
            out_nn = []
            for i in outputs_train[0:3]:
                out_nn.append(i.unsqueeze(0))
            out_nn = torch.cat(out_nn)
            out_nn, _ = torch.max(out_nn, dim=0)
            return tuple([out_nn])
        else:
            return tuple(outputs_train)


class LadderHead(nn.Module):
    def __init__(self, base_inchannels, base_outchannels, out_channels, norm_layer):
        super(LadderHead, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=base_inchannels[0],
            out_channels=base_outchannels,
            kernel_size=1,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            in_channels=base_inchannels[1],
            out_channels=base_outchannels * 2,
            kernel_size=1,
            bias=False,
        )
        self.conv3 = nn.Conv2d(
            in_channels=base_inchannels[2],
            out_channels=base_outchannels * 2 ** 2,
            kernel_size=1,
            bias=False,
        )
        self.conv4 = nn.Conv2d(
            in_channels=base_inchannels[3],
            out_channels=base_outchannels * 2 ** 3,
            kernel_size=1,
            bias=False,
        )

        self.bn1 = norm_layer(base_outchannels)
        self.bn2 = norm_layer(base_outchannels * 2)
        self.bn3 = norm_layer(base_outchannels * 2 ** 2)
        self.bn4 = norm_layer(base_outchannels * 2 ** 3)

        self.decoder = Decoder(planes=base_outchannels, layers=4, norm_layer=norm_layer)
        self.ladder = LadderBlock(
            planes=base_outchannels, layers=4, norm_layer=norm_layer
        )
        self.final = nn.Conv2d(base_outchannels, out_channels, 1)

    def forward(self, x):
        x1, x2, x3, x4 = x

        out1 = self.conv1(x1)  # 256 --> 64
        out1 = self.bn1(out1)
        out1 = F.relu(out1)

        out2 = self.conv2(x2)  # 512 --> 128
        out2 = self.bn2(out2)
        out2 = F.relu(out2)

        out3 = self.conv3(x3)  # 1024 --> 256
        out3 = self.bn3(out3)
        out3 = F.relu(out3)

        out4 = self.conv4(x4)  # 2048 --> 512
        out4 = self.bn4(out4)
        out4 = F.relu(out4)

        out = self.decoder([out1, out2, out3, out4])
        out = self.ladder(out)

        pred = self.final(out[-1])
        # if self.se_loss:
        #     enc = F.max_pool2d(out[0], kernel_size=out[0].size()[2:])
        #     enc = torch.squeeze(enc, -1)
        #     enc = torch.squeeze(enc, -1)
        #     se = self.selayer(enc)
        #     pred.append(se)

        return pred


class TascNetHead(nn.Module):
    def __init__(self, total_outputs, num_classes, norm_layer):
        super(TascNetHead, self).__init__()

        self.stuff_heads = []
        for _ in range(total_outputs):
            self.stuff_heads.append(self._make_stuff_head(norm_layer))
        self.stuff_heads = nn.Sequential(
            *self.stuff_heads
        )  # needed to make it cuda compliant. (pytorch wart)

        self.conv = nn.Conv2d(total_outputs * 128, num_classes, kernel_size=1)

    def _make_stuff_head(self, norm_layer):
        return nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(16, 128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            norm_layer(128),
        )

    def forward(self, x):
        fpn_outs = list(x)  # return 4 features from restnet backbone

        for i in range(len(fpn_outs)):
            fpn_outs[i] = self.stuff_heads[i](fpn_outs[i])

        _, _, H, W = fpn_outs[0].size()
        for i in range(1, len(fpn_outs)):
            fpn_outs[i] = F.upsample(fpn_outs[i], size=(H, W), mode="bilinear")

        output = torch.cat(fpn_outs, dim=1)
        output = self.conv(output)

        return output


if __name__ == "__main__":
    A = torch.rand(2, 3, 512, 512).cuda()
    model = MultiNet(
        65, "fpn_dilated_resnet101", use_fpn=True, add_danet=True, aux=True
    ).cuda()
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters: %d" % pytorch_total_params)
    model.eval()
    out = model(A)
    print(model)
    print(len(out))
    print(out[0].shape)

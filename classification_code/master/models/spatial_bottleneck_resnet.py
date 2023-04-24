"""Dilated ResNet"""
import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from modules.spatial_bottleneck_layer import ConvSpatialBottleneck

__all__ = [
    "SpBResNet",
    "sp_b_resnet18",
    "sp_b_resnet34",
    "sp_b_resnet50",
    "sp_b_resnet101",
    "sp_b_resnet152",
    "SpBBasicBlock",
    "SpBBottleneck",
]

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


class SpBBasicBlock(nn.Module):
    """ResNet BasicBlock
    """

    expansion = 1

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            dilation=1,
            downsample=None,
            previous_dilation=1,
            norm_layer=None,
    ):
        super(SpBBasicBlock, self).__init__()
        sampling_scale = 2
        ks = 3
        self.bn2 = norm_layer(inplanes)
        self.conv1 = nn.Conv2d(
            inplanes,
            inplanes // 2,
            kernel_size=ks,
            stride=sampling_scale * stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.ConvTranspose2d(
            inplanes // 2,
            planes,
            kernel_size=ks,
            stride=sampling_scale,
            padding=(ks - 1) // 2,
            dilation=previous_dilation,
            output_padding=sampling_scale - (ks - 1) // 2,
            bias=False,
        )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.bn2(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out


class SpBBottleneck(nn.Module):
    """ResNet Bottleneck
    """

    # pylint: disable=unused-argument
    expansion = 4

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            dilation=1,
            downsample=None,
            previous_dilation=1,
            norm_layer=None,
    ):
        super(SpBBottleneck, self).__init__()
        self.bn3 = norm_layer(inplanes)

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)

        self.conv2 = ConvSpatialBottleneck(
            planes,
            planes // 2,
            planes,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            stride=stride,
            bias=False,
        )
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride
        self.planes = planes
        self.inplanes = inplanes

    def _sum_each(self, x, y):
        assert len(x) == len(y)
        z = []
        for i in range(len(x)):
            z.append(x[i] + y[i])
        return z

    def forward(self, x):
        residual = x
        out = self.bn3(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class SpBResNet(nn.Module):
    """Dilated Pre-trained ResNet Model, which preduces the stride of 8 featuremaps at conv5.
    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    Reference:
        - He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """

    # pylint: disable=unused-variable
    def __init__(
            self,
            block,
            layers,
            num_classes=1000,
            dilated=True,
            norm_layer=nn.BatchNorm2d,
            add_additional_layers=False,
            dilate_only_last_layer=False,
    ):
        self.inplanes = 64
        super(SpBResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = norm_layer(64)
        # self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, norm_layer=norm_layer
        )
        if dilated:
            if dilate_only_last_layer:
                self.layer3 = self._make_layer(
                    block, 256, layers[2], stride=2, norm_layer=norm_layer
                )
                self.layer4 = self._make_layer(
                    block, 256, layers[2], stride=1, dilation=2, norm_layer=norm_layer
                )
            else:
                self.layer3 = self._make_layer(
                    block, 256, layers[2], stride=1, dilation=2, norm_layer=norm_layer
                )
                self.layer4 = self._make_layer(
                    block, 512, layers[3], stride=1, dilation=4, norm_layer=norm_layer
                )
        else:
            self.layer3 = self._make_layer(
                block, 256, layers[2], stride=2, norm_layer=norm_layer
            )
            self.layer4 = self._make_layer(
                block, 512, layers[3], stride=2, norm_layer=norm_layer
            )

        if add_additional_layers:
            self.layer5 = self._make_layer(
                block, 64, layers[3], stride=2, norm_layer=norm_layer
            )
            self.layer6 = self._make_layer(
                block, 64, layers[3], stride=1, norm_layer=norm_layer
            )

            self.layer7 = self._make_layer(
                block, 64, layers[3], stride=2, norm_layer=norm_layer
            )

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                norm_layer(self.inplanes),
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
            )

        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride,
                    dilation=1,
                    downsample=downsample,
                    previous_dilation=dilation,
                    norm_layer=norm_layer,
                )
            )
        elif dilation == 4:
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride,
                    dilation=2,
                    downsample=downsample,
                    previous_dilation=dilation,
                    norm_layer=norm_layer,
                )
            )
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    dilation=dilation,
                    previous_dilation=dilation,
                    norm_layer=norm_layer,
                )
            )
        layers.append(
            norm_layer(planes * block.expansion)
        )
        layers.append(
            nn.ReLU(inplace=True)
        )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        x = self.maxpool(x)

        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        out = self.avgpool(c4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def sp_b_resnet18(pretrained=False, root="", **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SpBResNet(SpBBasicBlock, [2, 2, 2, 2], **kwargs)
    model.base_inchannels = [64, 128, 256, 512]
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet18"]), strict=False)
    return model


def sp_b_resnet34(pretrained=False, root="", **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SpBResNet(SpBBasicBlock, [3, 4, 6, 3], **kwargs)
    model.base_inchannels = [64, 128, 256, 512]
    if pretrained:
        assert False, "Pretrained model not implemented"
        # model.load_state_dict(model_zoo.load_url(model_urls["resnet34"]))
    return model


def sp_b_resnet50(pretrained=False, root="~/.encoding/models", **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SpBResNet(SpBBottleneck, [3, 4, 6, 3], dilated=False, **kwargs)
    model.base_inchannels = [256, 512, 1024, 2048]
    if pretrained:
        assert False, "Pretrained model not implemented"
        # from ..models.model_store import get_model_file

        # model.load_state_dict(
        #    torch.load(get_model_file("resnet50", root=root)), strict=False
        # )
    return model


def sp_b_resnet101(pretrained=False, root="~/.encoding/models", **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SpBResNet(SpBBottleneck, [3, 4, 23, 3], **kwargs)
    model.base_inchannels = [256, 512, 1024, 2048]
    if pretrained:
        assert False, "Pretrained model not implemented"
        # from ..models.model_store import get_model_file

        # model.load_state_dict(
        #    torch.load(get_model_file("resnet101", root=root)), strict=False
        # )
    return model


def sp_b_resnet152(pretrained=False, root="~/.encoding/models", **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SpBResNet(SpBBottleneck, [3, 8, 36, 3], **kwargs)
    model.base_inchannels = [256, 512, 1024, 2048]
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet152"]))
    return model

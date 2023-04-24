import torch
import torch.nn.functional as F
import torch.nn as nn
from encoding.nn import SegmentationLosses, SyncBatchNorm

drop = 0.25


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True
    )


class BasicBlock(nn.Module):  # shared-weights residual block
    expansion = 1

    def __init__(
        self, inplanes, planes, stride=1, rate=1, downsample=None, norm_layer=None
    ):
        super(BasicBlock, self).__init__()
        if inplanes != planes:
            self.conv0 = conv3x3(inplanes, planes, rate)

        self.inplanes = inplanes
        self.planes = planes

        self.conv1 = conv3x3(planes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        # self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.drop = nn.Dropout2d(p=drop)

    def forward(self, x):
        if self.inplanes != self.planes:
            x = self.conv0(x)
            x = F.relu(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.drop(out)

        out1 = self.conv1(out)  # using the same conv
        out1 = self.bn2(out1)
        # out1 = self.relu(out1)

        out2 = out1 + x

        return F.relu(out2)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = SyncBatchNorm(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = SyncBatchNorm(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = SyncBatchNorm(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Initial_LadderBlock(nn.Module):
    def __init__(self, planes, layers, kernel=3, block=BasicBlock, inplanes=3):
        super().__init__()
        self.planes = planes
        self.layers = layers
        self.kernel = kernel

        self.padding = int((kernel - 1) / 2)
        self.inconv = nn.Conv2d(
            in_channels=inplanes,
            out_channels=planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
        self.in_bn = SyncBatchNorm(planes)
        # create module list for down branch
        self.down_module_list = nn.ModuleList()
        for i in range(0, layers):
            self.down_module_list.append(block(planes * (2 ** i), planes * (2 ** i)))

        # use strided conv instead of poooling
        self.down_conv_list = nn.ModuleList()
        for i in range(0, layers):
            self.down_conv_list.append(
                nn.Conv2d(
                    planes * 2 ** i,
                    planes * 2 ** (i + 1),
                    stride=2,
                    kernel_size=kernel,
                    padding=self.padding,
                )
            )

        # create module for bottom block
        self.bottom = block(planes * (2 ** layers), planes * (2 ** layers))

        # create module list for up branch
        self.up_conv_list = nn.ModuleList()
        self.up_dense_list = nn.ModuleList()
        for i in range(0, layers):
            self.up_conv_list.append(
                nn.ConvTranspose2d(
                    in_channels=planes * 2 ** (layers - i),
                    out_channels=planes * 2 ** max(0, layers - i - 1),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=True,
                )
            )
            self.up_dense_list.append(
                block(
                    planes * 2 ** max(0, layers - i - 1),
                    planes * 2 ** max(0, layers - i - 1),
                )
            )

    def forward(self, x):
        out = self.inconv(x)
        out = self.in_bn(out)
        out = F.relu(out)

        down_out = []
        # down branch
        for i in range(0, self.layers):
            out = self.down_module_list[i](out)
            down_out.append(out)
            out = self.down_conv_list[i](out)
            out = F.relu(out)

        # bottom branch
        out = self.bottom(out)
        bottom = out

        # up branch
        up_out = []
        up_out.append(bottom)

        for j in range(0, self.layers):
            out = self.up_conv_list[j](out) + down_out[self.layers - j - 1]
            # out = F.relu(out)
            out = self.up_dense_list[j](out)
            up_out.append(out)

        return up_out


class Decoder(nn.Module):
    def __init__(self, planes, layers, kernel=3, block=BasicBlock, norm_layer=None):
        super().__init__()
        self.planes = planes
        self.layers = layers
        self.kernel = kernel
        self.padding = int((kernel - 1) / 2)
        self.inconv = block(planes, planes, norm_layer=norm_layer)
        # create module for bottom block
        self.bottom = block(
            planes * (2 ** (layers - 1)),
            planes * (2 ** (layers - 1)),
            norm_layer=norm_layer,
        )

        # create module list for up branch
        self.up_conv_list = nn.ModuleList()
        self.up_dense_list = nn.ModuleList()
        for i in range(0, layers - 1):
            self.up_conv_list.append(
                nn.ConvTranspose2d(
                    planes * 2 ** (layers - 1 - i),
                    planes * 2 ** max(0, layers - i - 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=True,
                )
            )
            self.up_dense_list.append(
                block(
                    planes * 2 ** max(0, layers - i - 2),
                    planes * 2 ** max(0, layers - i - 2),
                    norm_layer=norm_layer,
                )
            )
            # 256, 128, 64

        """
        self.up_conv_list
        ModuleList(
  (0): ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
  (1): ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
  (2): ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
)

######################################
        self.up_dense_list
        ModuleList(
  (0): BasicBlock(
    (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn1): SyncBatchNorm(sync=False)
    (relu): ReLU(inplace)
    (bn2): SyncBatchNorm(sync=False)
    (drop): Dropout2d(p=0.25)
  )
  (1): BasicBlock(
    (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn1): SyncBatchNorm(sync=False)
    (relu): ReLU(inplace)
    (bn2): SyncBatchNorm(sync=False)
    (drop): Dropout2d(p=0.25)
  )
  (2): BasicBlock(
    (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bn1): SyncBatchNorm(sync=False)
    (relu): ReLU(inplace)
    (bn2): SyncBatchNorm(sync=False)
    (drop): Dropout2d(p=0.25)
  )  
        """

    def forward(self, x):
        # bottom branch
        out = self.bottom(x[-1])
        bottom = out

        # up branch
        up_out = []
        up_out.append(bottom)

        for j in range(0, self.layers - 1):
            out = self.up_conv_list[j](out) + x[self.layers - j - 2]
            # out = F.relu(out)
            out = self.up_dense_list[j](out)
            up_out.append(out)

        return up_out


class LadderBlock(nn.Module):
    def __init__(self, planes, layers, kernel=3, block=BasicBlock, norm_layer=None):
        super().__init__()
        self.planes = planes
        self.layers = layers
        self.kernel = kernel

        self.padding = int((kernel - 1) / 2)
        self.inconv = block(planes, planes, norm_layer=norm_layer)

        # create module list for down branch
        self.down_module_list = nn.ModuleList()
        for i in range(0, layers - 1):
            self.down_module_list.append(
                block(planes * (2 ** i), planes * (2 ** i), norm_layer=norm_layer)
            )

        # use strided conv instead of pooling
        self.down_conv_list = nn.ModuleList()
        for i in range(0, layers - 1):
            self.down_conv_list.append(
                nn.Conv2d(
                    planes * 2 ** i,
                    planes * 2 ** (i + 1),
                    stride=2,
                    kernel_size=kernel,
                    padding=self.padding,
                )
            )

        # create module for bottom block
        self.bottom = block(
            planes * (2 ** (layers - 1)),
            planes * (2 ** (layers - 1)),
            norm_layer=norm_layer,
        )

        # create module list for up branch
        self.up_conv_list = nn.ModuleList()
        self.up_dense_list = nn.ModuleList()
        for i in range(0, layers - 1):
            self.up_conv_list.append(
                nn.ConvTranspose2d(
                    planes * 2 ** (layers - i - 1),
                    planes * 2 ** max(0, layers - i - 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=True,
                )
            )
            self.up_dense_list.append(
                block(
                    planes * 2 ** max(0, layers - i - 2),
                    planes * 2 ** max(0, layers - i - 2),
                    norm_layer=norm_layer,
                )
            )

        # print(self.up_conv_list, self.up_dense_list)
        # input()

        """
        self.up_conv_list
        ModuleList(
          (0): ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
          (1): ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
          (2): ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
        ) 


       self.up_dense_list
        ModuleList(
          (0): BasicBlock(
            (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (bn1): SyncBatchNorm(sync=False)
            (relu): ReLU(inplace)
            (bn2): SyncBatchNorm(sync=False)
            (drop): Dropout2d(p=0.25)
          )
          (1): BasicBlock(
            (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (bn1): SyncBatchNorm(sync=False)
            (relu): ReLU(inplace)
            (bn2): SyncBatchNorm(sync=False)
            (drop): Dropout2d(p=0.25)
          )
          (2): BasicBlock(
            (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (bn1): SyncBatchNorm(sync=False)
            (relu): ReLU(inplace)
            (bn2): SyncBatchNorm(sync=False)
            (drop): Dropout2d(p=0.25)
          )
        )
        
        """

    def forward(self, x):
        out = self.inconv(x[-1])

        down_out = []
        # down branch
        for i in range(0, self.layers - 1):
            out = out + x[-i - 1]
            out = self.down_module_list[i](out)
            down_out.append(out)

            out = self.down_conv_list[i](out)
            out = F.relu(out)

        # bottom branch
        out = self.bottom(out)
        bottom = out

        # up branch
        up_out = []
        up_out.append(bottom)

        for j in range(0, self.layers - 1):
            out = self.up_conv_list[j](out) + down_out[self.layers - j - 2]
            # out = F.relu(out)
            out = self.up_dense_list[j](out)
            up_out.append(out)

        return up_out


class Final_LadderBlock(nn.Module):
    def __init__(self, planes, layers, kernel=3, block=BasicBlock, inplanes=3):
        super().__init__()
        self.block = LadderBlock(planes, layers, kernel=kernel, block=block)

    def forward(self, x):
        out = self.block(x)
        return out[-1]


class LadderNetv6(nn.Module):
    def __init__(self, layers=4, filters=32, num_classes=21, inplanes=3):
        super().__init__()
        self.initial_block = Initial_LadderBlock(
            planes=filters, layers=layers, inplanes=inplanes
        )
        # self.middle_block = LadderBlock(planes=filters,layers=layers)
        self.final_block = Final_LadderBlock(planes=filters, layers=layers)
        self.final = nn.Conv2d(
            in_channels=filters, out_channels=num_classes, kernel_size=1
        )

    def forward(self, x):
        out = self.initial_block(x)
        # out = self.middle_block(out)
        out = self.final_block(out)
        out = self.final(out)
        # out = F.relu(out)
        # out = F.log_softmax(out,dim=1)
        return out

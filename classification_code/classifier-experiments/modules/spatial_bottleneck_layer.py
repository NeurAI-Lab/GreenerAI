from torch import nn


class ConvSpatialBottleneck(nn.Module):
    # Paper has not mentioned the quantity of inter_channels.
    # It is also assumed that the input size(resolution) are in multiples of 4
    def __init__(self, in_channels, inter_channels, out_channels, kernel_size=3, stride=1, sampling_scale=2, **kwargs):
        super(ConvSpatialBottleneck, self).__init__()
        assert kernel_size % 2 == 1, "conv_spatial_bottleneck is implemented only for odd kernel_size"
        kwargs['padding'] = (kernel_size-1)//2
        self._conv  = nn.Conv2d(in_channels=in_channels, out_channels=inter_channels, kernel_size=kernel_size, stride=sampling_scale*stride, **kwargs)
        self._convT = nn.ConvTranspose2d(in_channels=inter_channels, out_channels=out_channels, kernel_size=kernel_size, stride=sampling_scale, output_padding=max(1, sampling_scale-(kernel_size-1)//2),   **kwargs)

    def forward(self, input):
        input = self._conv(input)
        input = self._convT(input)
        return input

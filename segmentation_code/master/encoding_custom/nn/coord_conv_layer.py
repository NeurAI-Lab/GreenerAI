from torch import nn
import torch


class AddCoords(nn.Module):
    def __init__(self, spatial_dim=None, radius_channel=True, coord_scales=(1.0, 1.0)):
        super(AddCoords, self).__init__()
        self.radius_channel = radius_channel
        self.spatial_dim = spatial_dim
        self.coord_scales = coord_scales
        self.xx_channel = None
        self.xx_channel = None
        self.radius_calc = None
        if self.spatial_dim is not None:
            self.xx_channel, self.yy_channel = self._create_channels(self.spatial_dim)
            if self.radius_channel:
                self.radius_calc = torch.sqrt(self.xx_channel ** 2 + self.yy_channel ** 2)

    def _create_channels(self, spatial_dim):
        xx_ones = torch.ones([spatial_dim[0], 1], dtype=torch.int32)
        yy_ones = torch.ones([spatial_dim[1], 1], dtype=torch.int32)

        xx_range = torch.arange(spatial_dim[1], dtype=torch.int32).unsqueeze(0)
        yy_range = torch.arange(spatial_dim[0], dtype=torch.int32).unsqueeze(0)

        xx_channel = torch.matmul(xx_ones, xx_range)
        yy_channel = torch.matmul(yy_ones, yy_range).t()

        xx_channel = xx_channel.unsqueeze(0)
        yy_channel = yy_channel.unsqueeze(0)

        xx_channel = xx_channel.float() / (spatial_dim[1] - 1)
        yy_channel = yy_channel.float() / (spatial_dim[0] - 1)

        xx_channel = (xx_channel * 2 - 1)*self.coord_scales[0]
        yy_channel = (yy_channel * 2 - 1)*self.coord_scales[1]

        return xx_channel, yy_channel

    def forward(self, in_tensor):
        shape_ = [in_tensor.shape[0], 1, in_tensor.shape[2], in_tensor.shape[3]]
        if self.spatial_dim is None:
            self.xx_channel, self.yy_channel = self._create_channels(shape_[-2:])
            if self.radius_channel:
                self.radius_calc = torch.sqrt(self.xx_channel ** 2 + self.yy_channel ** 2)

        if self.radius_channel:
            out = torch.cat([in_tensor,
                             self.xx_channel.expand(shape_).cuda(),
                             self.yy_channel.expand(shape_).cuda(),
                             self.radius_calc.expand(shape_).cuda()],
                            dim=1)
        else:
            out = torch.cat([in_tensor,
                             self.xx_channel.expand(shape_).cuda(),
                             self.yy_channel.expand(shape_).cuda()],
                            dim=1)
        return out


class CoordConv(nn.Module):
    """ add any additional coordinate channels to the input tensor """
    def __init__(self, in_channels, out_channels, kernel_size, in_spatial_resolution=None, radial_channel=False, **kwargs):
        super(CoordConv, self).__init__()
        self.addcoord = AddCoords(in_spatial_resolution, radius_channel=radial_channel)
        in_channels += 3 if radial_channel else 2

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)

    def forward(self, in_tensor):
        out = self.addcoord(in_tensor)
        out = self.conv(out)
        return out


class CoordConvTranspose(nn.Module):
    """CoordConvTranspose layer for segmentation tasks."""
    def __init__(self, in_channels, out_channels, kernel_size, in_spatial_resolution=None, radial_channel=False, **kwargs):
        super(CoordConvTranspose, self).__init__()
        self.addcoord = AddCoords(in_spatial_resolution, radius_channel=radial_channel)
        in_channels += 3 if radial_channel else 2
        self.convT = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, **kwargs)

    def forward(self, in_tensor):
        out = self.addcoord(in_tensor)
        out = self.convT(out)
        return out
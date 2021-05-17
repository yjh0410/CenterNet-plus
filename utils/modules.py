import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import os
from copy import deepcopy



class Conv(nn.Module):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1, act=True):
        super(Conv, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True) if act else nn.Identity()
        )

    def forward(self, x):
        return self.convs(x)


class DeConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=2, act=True):
        super(DeConv, self).__init__()
        # deconv basic config
        if ksize == 4:
            padding = 1
            output_padding = 0
        elif ksize == 3:
            padding = 1
            output_padding = 1
        elif ksize == 2:
            padding = 0
            output_padding = 0

        self.convs = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, ksize, stride=stride, padding=padding, output_padding=output_padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) if act else nn.Identity()
        )

    def forward(self, x):
        return self.convs(x)


class UpSample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corner=None):
        super(UpSample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corner = align_corner

    def forward(self, x):
        return torch.nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, 
                                                mode=self.mode, align_corners=self.align_corner)


class ResizeConv(nn.Module):
    def __init__(self, in_ch, out_ch, size=None, scale_factor=None, mode='nearest', align_corner=None):
        super(ResizeConv, self).__init__()
        self.upsample = UpSample(size=size, scale_factor=scale_factor, mode=mode, align_corner=align_corner)
        self.conv = Conv(in_ch, out_ch, k=1)

    def forward(self, x):
        x = self.conv(self.upsample(x))
        return x


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, in_ch, dilation=1, e=0.5):
        super(Bottleneck, self).__init__()
        inter_ch = int(in_ch * e)
        self.branch = nn.Sequential(
            Conv(in_ch, inter_ch, k=1),
            Conv(inter_ch, inter_ch, k=3, p=dilation, d=dilation),
            Conv(inter_ch, in_ch, k=1)
        )

    def forward(self, x):
        return x + self.branch(x)


class DilateEncoder(nn.Module):
    """ DilateEncoder """
    def __init__(self, in_ch, out_ch, dilation_list=[2, 4, 6, 8]):
        super(DilateEncoder, self).__init__()
        self.projector = nn.Sequential(
            Conv(in_ch, out_ch, k=1, act=False),
            Conv(out_ch, out_ch, k=3, p=1, act=False)
        )
        encoders = []
        for d in dilation_list:
            encoders.append(Bottleneck(in_ch=out_ch, dilation=d))
        self.encoders = nn.Sequential(*encoders)

    def forward(self, x):
        x = self.projector(x)
        x = self.encoders(x)

        return x


class SPP(nn.Module):
    """
        Spatial Pyramid Pooling
    """
    def __init__(self):
        super(SPP, self).__init__()

    def forward(self, x):
        x_1 = F.max_pool2d(x, 5, stride=1, padding=2)
        x_2 = F.max_pool2d(x, 9, stride=1, padding=4)
        x_3 = F.max_pool2d(x, 13, stride=1, padding=6)
        x = torch.cat([x, x_1, x_2, x_3], dim=1)

        return x


class CoordConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, p=0, s=1, d=1, g=1, act=True):
        super(CoordConv, self).__init__()
        self.conv = Conv(in_ch + 2, out_ch, k=k, p=p, s=s, d=d, g=g, act=act)

    def forward(self, x):
        """CoordConv.

            Input:
                x: [B, C, H, W]
                gridxy: [1, H*W, 2]
        """
        B, _, H, W = x.size()
        device = x.device
        grid_y, grid_x = torch.meshgrid([torch.arange(H), torch.arange(W)])
        # [2, H, W]
        gridxy = torch.stack([grid_x, grid_y], dim=0).float()
        # [H, W, 2] -> [B, H, W, 2]
        gridxy = gridxy.repeat(B, 1, 1, 1).to(device)

        # normalize gridxy -> [-1, 1]
        gridxy[:, 0, :, :] = (gridxy[:, 0, :, :] / (W - 1)) * 2.0 - 1.0
        gridxy[:, 1, :, :] = (gridxy[:, 1, :, :] / (H - 1)) * 2.0 - 1.0

        x_coord = torch.cat([x, gridxy], dim=1)
        y = self.conv(x_coord)

        return y


class ModelEMA(object):
    def __init__(self, model, decay=0.9999, updates=0):
        # create EMA
        self.ema = deepcopy(model).eval()
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000.))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

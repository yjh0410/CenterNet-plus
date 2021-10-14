import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from copy import deepcopy


class Conv(nn.Module):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1, act='relu'):
        super(Conv, self).__init__()
        if act is not None:
            if act == 'relu':
                self.convs = nn.Sequential(
                    nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=False),
                    nn.BatchNorm2d(c2),
                    nn.ReLU(inplace=True) if act else nn.Identity()
                )
            elif act == 'leaky':
                self.convs = nn.Sequential(
                    nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=False),
                    nn.BatchNorm2d(c2),
                    nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()
                )
        else:
            self.convs = nn.Sequential(
                nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=False),
                nn.BatchNorm2d(c2)
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
    def __init__(self, c1, c2, act='relu', size=None, scale_factor=None, mode='nearest', align_corner=None):
        super(ResizeConv, self).__init__()
        self.upsample = UpSample(size=size, scale_factor=scale_factor, mode=mode, align_corner=align_corner)
        self.conv = Conv(c1, c2, k=1, act=act)

    def forward(self, x):
        x = self.conv(self.upsample(x))
        return x


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c, d=1, e=0.5, act='relu'):
        super(Bottleneck, self).__init__()
        c_ = int(c * e)
        self.branch = nn.Sequential(
            Conv(c, c_, k=1, act=act),
            Conv(c_, c_, k=3, p=d, d=d, act=act),
            Conv(c_, c, k=1, act=act)
        )

    def forward(self, x):
        return x + self.branch(x)


class DilateEncoder(nn.Module):
    """ DilateEncoder """
    def __init__(self, c1, c2, act='relu', dilation_list=[4, 8, 12, 16]):
        super(DilateEncoder, self).__init__()
        self.projector = nn.Sequential(
            Conv(c1, c2, k=1, act=None),
            Conv(c2, c2, k=3, p=1, act=None)
        )
        encoders = []
        for d in dilation_list:
            encoders.append(Bottleneck(c=c2, d=d, act=act))
        self.encoders = nn.Sequential(*encoders)

    def forward(self, x):
        x = self.projector(x)
        x = self.encoders(x)

        return x


class SPP(nn.Module):
    """
        Spatial Pyramid Pooling
    """
    def __init__(self, c1, c2, e=0.5, act='relu'):
        super(SPP, self).__init__()
        c_ = int(c1 * e)
        self.cv1 = Conv(c1, c_, k=1, act=act)
        self.cv2 = Conv(c_*4, c2, k=1, act=act)

    def forward(self, x):
        x = self.cv1(x)
        x_1 = F.max_pool2d(x, 5, stride=1, padding=2)
        x_2 = F.max_pool2d(x, 9, stride=1, padding=4)
        x_3 = F.max_pool2d(x, 13, stride=1, padding=6)
        x = torch.cat([x, x_1, x_2, x_3], dim=1)
        x = self.cv2(x)

        return x


class CoordConv(nn.Module):
    def __init__(self, c1, c2, k=1, p=0, s=1, d=1, g=1, act='relu'):
        super(CoordConv, self).__init__()
        self.conv = Conv(c1 + 2, c2, k=k, p=p, s=s, d=d, g=g, act=act)

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

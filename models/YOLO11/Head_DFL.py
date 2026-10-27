import math
import torch

from ...utils import make_anchors # from remaining utils.py
from .Blocks_Module import Conv

class DFL(torch.nn.Module):
    # Generalized Focal Loss
    # https://ieeexplore.ieee.org/document/9792391
    
    def __init__(self, ch=16):
        super().__init__()
        self.ch = ch
        self.conv = torch.nn.Conv2d(ch, out_channels=1, kernel_size=1, bias=False).requires_grad_(False)
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = torch.nn.Parameter(x)

    def forward(self, x):
        b, c, a = x.shape
        x = x.view(b, 4, self.ch, a).transpose(2, 1)
        return self.conv(x.softmax(1)).view(b, 4, a)


class Head(torch.nn.Module):
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, filters=()):
        super().__init__()
        self.ch = 16  # DFL channels
        self.nc = nc  # number of classes
        self.nl = len(filters)  # number of detection layers
        self.no = nc + self.ch * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build

        box = max(64, filters[0] // 4)
        cls = max(80, filters[0], self.nc)

        self.dfl = DFL(self.ch)
        self.box = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, box,torch.nn.SiLU(), k=3, p=1),
                                                           Conv(box, box,torch.nn.SiLU(), k=3, p=1),
                                                           torch.nn.Conv2d(box, out_channels=4 * self.ch,
                                                                           kernel_size=1)) for x in filters)
        self.cls = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, x, torch.nn.SiLU(), k=3, p=1, g=x),
                                                           Conv(x, cls, torch.nn.SiLU()),
                                                           Conv(cls, cls, torch.nn.SiLU(), k=3, p=1, g=cls),
                                                           Conv(cls, cls, torch.nn.SiLU()),
                                                           torch.nn.Conv2d(cls, out_channels=self.nc,
                                                                           kernel_size=1)) for x in filters)

    def forward(self, x):
        for i, (box, cls) in enumerate(zip(self.box, self.cls)):
            x[i] = torch.cat(tensors=(box(x[i]), cls(x[i])), dim=1)
        if self.training:
            return x

        self.anchors, self.strides = (i.transpose(0, 1) for i in make_anchors(x, self.stride))          # Cơ chế make anchor này - có phải fixed anchor || dynamic anchor 6/Nov/2025
        x = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], dim=2)
        box, cls = x.split(split_size=(4 * self.ch, self.nc), dim=1)

        a, b = self.dfl(box).chunk(2, 1)
        a = self.anchors.unsqueeze(0) - a
        b = self.anchors.unsqueeze(0) + b
        box = torch.cat(tensors=((a + b) / 2, b - a), dim=1)

        return torch.cat(tensors=(box * self.strides, cls.sigmoid()), dim=1)

    def initialize_biases(self):
        # Initialize biases
        # WARNING: requires stride availability
        for box, cls, s in zip(self.box, self.cls, self.stride):
            # box
            box[-1].bias.data[:] = 1.0
            # cls (.01 objects, 80 classes, 640 image)
            cls[-1].bias.data[:self.nc] = math.log(5 / self.nc / (640 / s) ** 2)
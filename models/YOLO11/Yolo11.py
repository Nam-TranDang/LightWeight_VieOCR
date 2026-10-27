import torch
from .DarkNet import DarkNet
from .Neck_DarkFPN import DarkFPN
from .Head_DFL import Head
from .Blocks_Module import fuse_conv, Conv

class Yolo11(torch.nn.Module):
    def __init__(self, width, depth, csp, num_classes):
        super().__init__()
        self.net = DarkNet(width, depth, csp)
        self.fpn = DarkFPN(width, depth, csp)

        img_dummy = torch.zeros(1, width[0], 256, 256)
        self.head = Head(num_classes, (width[3], width[4], width[5]))
        self.head.stride = torch.tensor([256 / x.shape[-2] for x in self.forward(img_dummy)])
        self.stride = self.head.stride
        self.head.initialize_biases()

    def forward(self, x):
        x = self.net(x)
        x = self.fpn(x)
        return self.head(list(x))

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self
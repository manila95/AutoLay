""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *
import torch.optim as optim

class PseudoLidar_UNet(nn.Module):
    def __init__(self, opt, num_ch_dec=2, bilinear=True):
        super(PseudoLidar_UNet, self).__init__()
        self.opt = opt
        n_channels = 10
        n_classes = num_ch_dec
        self.bilinear = bilinear
        
        # Network Architecture
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        self.parameters = list(self.parameters())
        self.model_optimizer = optim.Adam(self.parameters, opt.lr)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
                self.model_optimizer, opt.scheduler_step_size, 0.1)

    def forward(self, x):
        outputs = {}
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        outputs["topview"] = self.outc(x)
        return outputs

    def step(self, inputs, outputs, losses, epoch):
        self.model_optimizer.zero_grad()
        losses["loss"].backward()
        self.model_optimizer.step()

        return losses


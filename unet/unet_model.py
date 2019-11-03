from unet.unet_parts import *
import torch.nn as nn


class Unet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Unet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = U_down(64, 128)
        self.down2 = U_down(128, 256)
        self.down3 = U_down(256, 512)
        self.down4 = U_down(512, 512)
        self.up1 = U_up(1024, 256)
        self.up2 = U_up(512, 128)
        self.up3 = U_up(256, 64)
        self.up4 = U_up(128, 64)
        self.out = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)
        return x


class Res_Unet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Res_Unet, self).__init__()
        self.down = RU_first_down(n_channels, 32)
        self.down1 = RU_down(32, 64)
        self.down2 = RU_down(64, 128)
        self.down3 = RU_down(128, 256)
        self.down4 = RU_down(256, 256)
        self.up1 = RU_up(512, 128)
        self.up2 = RU_up(256, 64)
        self.up3 = RU_up(128, 32)
        self.up4 = RU_up(64, 32)
        self.out = outconv(32, n_classes)

    def forward(self, x):
        x1 = self.down(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)
        return x


class Ringed_Res_Unet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(Ringed_Res_Unet, self).__init__()
        self.down = RRU_first_down(n_channels, 32)
        self.down1 = RRU_down(32, 64)
        self.down2 = RRU_down(64, 128)
        self.down3 = RRU_down(128, 256)
        self.down4 = RRU_down(256, 256)
        self.up1 = RRU_up(512, 128)
        self.up2 = RRU_up(256, 64)
        self.up3 = RRU_up(128, 32)
        self.up4 = RRU_up(64, 32)
        self.out = outconv(32, n_classes)

    def forward(self, x):
        x1 = self.down(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)
        return x
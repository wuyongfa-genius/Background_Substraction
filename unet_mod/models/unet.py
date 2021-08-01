""" Full assembly of the parts to form the complete network """
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, upsample="bilinear"):
        super().__init__()

        # if upsample is bilinear or trilinear, use the normal convolutions to reduce the number of channels
        if upsample is not None:
            self.up = nn.Upsample(scale_factor=2, mode=upsample, align_corners=False)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.shape[2] - x1.shape[2]
        diffX = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, upsample="bilinear"):
        super(UNet, self).__init__()
        assert upsample in [None, "bilinear", "trilinear", "bicubic", "nearest"]
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.upsample = upsample

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if upsample is not None else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, upsample)
        self.up2 = Up(512, 256 // factor, upsample)
        self.up3 = Up(256, 128 // factor, upsample)
        self.up4 = Up(128, 64, upsample)
        self.outc = OutConv(64, n_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0.)

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
        logits = self.outc(x)
        return logits


class TinyUNet(nn.Module):
    def __init__(self, n_channels, n_classes, upsample="bilinear"):
        super().__init__()
        self.inc = DoubleConv(n_channels, 128, mid_channels=64)
        factor = 2 if upsample else 1
        self.down1 = Down(128, 256//factor)
        
        self.up1 = Up(256, 64, upsample)
        # C = 1 if n_classes==2 else n_classes
        self.outc = OutConv(64, n_classes)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.constant_(m.weight, 1.)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x = self.up1(x2, x1)
        logits = self.outc(x)
        return logits


class TinyUNet3d(nn.Module):
    def __init__(self, n_channels, n_classes, upsample="trilinear"):
        super().__init__()
        self.upsample = upsample
        self.inc = nn.Sequential(
            nn.Conv3d(n_channels, 32, kernel_size=(3, 5, 5), padding=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        self.down1 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        self.up1 = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        self.outc = nn.Conv3d(32, n_classes, kernel_size=1)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
            nn.init.constant_(m.weight, 1.)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = F.interpolate(x2, scale_factor=2, mode=self.upsample, align_corners=False)
        assert x2.shape[-2:]==x1.shape[-2:]
        x = torch.cat((x2, x1), dim=1)
        x = self.up1(x)
        logits = self.outc(x)
        return logits

def replace2dwith3d(model, inflated=True):
    """Replace all 2d modules with their 3d counterpart.
    Args:
        model(nn.Module): the model to be replaced.
        inflated(bool): whether to inflate 2d pretrained weights
            to 3d as well.
    """
    modules = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            kernel_size = module.kernel_size[0]
            stride = module.stride[0]
            padding = module.padding[0]
            if inflated:
                weight = module.weight.unsqueeze(2) / kernel_size
                weight = torch.cat([weight for _ in range(0, kernel_size)], dim=2)
            bias = module.bias

            if(bias is None):
                modules[name] = nn.Conv3d(in_channels=module.weight.shape[1], out_channels=module.weight.shape[0],
                                kernel_size=kernel_size, padding=padding, stride=stride, bias=False)
            else:
                modules[name] = nn.Conv3d(in_channels=module.weight.shape[1], out_channels=module.weight.shape[0],
                                kernel_size=kernel_size, padding=padding, stride=stride, bias=True)
                modules[name].bias = bias
            if inflated:
                modules[name].weight.data = weight

        elif isinstance(module, nn.BatchNorm2d):
            weight = module.weight
            bias = module.bias
            modules[name] = nn.BatchNorm3d(weight.shape[0])
            modules[name].weight = weight
            modules[name].bias = bias
        elif isinstance(module, nn.MaxPool2d):
            kernel_size = module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0]
            stride = module.stride if isinstance(module.stride, int) else module.stride[0]
            padding = module.padding if isinstance(module.padding, int) else module.padding[0]
            modules[name] = nn.MaxPool3d(kernel_size, stride, padding)
        elif isinstance(module, nn.Upsample):
            scale_factor = module.scale_factor
            align_corners = module.align_corners
            modules[name] = nn.Upsample(scale_factor=scale_factor, mode="trilinear", align_corners=align_corners)

    for name in modules:
        parent_module = model
        objs = name.split(".")
        if len(objs) == 1:
            model.__setattr__(name, modules[name])
            continue

        for obj in objs[:-1]:
            parent_module = parent_module.__getattr__(obj)

        parent_module.__setattr__(objs[-1], modules[name])


# if __name__=="__main__":
#     # unet = TinyUNet(n_channels=3, n_classes=2, upsample="bilinear")
#     # replace2dwith3d(unet, inflated=True)
#     unet3d = TinyUNet3d(1, 2)
#     print(unet3d)
#     unet3d = unet3d.cuda()
#     x = torch.randn(2, 1, 4, 256, 256)
#     print(unet3d(x.cuda()).shape)
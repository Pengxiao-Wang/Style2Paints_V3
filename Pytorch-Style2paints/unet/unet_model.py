# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F
import pretrainedmodels

import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
 
    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2, ceil_mode=False),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class remain(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(remain, self).__init__()
        self.mpconv = nn.Sequential(
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        # import pdb;pdb.set_trace()
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        from googlenet import get_googlenet
        self.inc = inconv(in_channels, 96)
        self.remain0 = remain(96, 96)
        self.remain6 = remain(96, 96)
        self.down1 = down(96, 192)
        self.remain1 = remain(192, 192)
        self.remain5 = remain(192, 192)
        self.down2 = down(192, 384)
        self.remain2 = remain(384, 384)
        self.remain4 = remain(384, 384)
        self.down3 = down(384, 768)
        self.down4 = down(768, 1024)
        # self.down5 = down(1024, 2048)
        self.remain3 = remain(1024, 1024)
        # self.remain3 = down(512, 512)
        self.inceptionv1 = get_googlenet(pretrain=True, pth_path='./weights/google_net.pth')
        # self.inceptionv3.load_state_dict(model_zoo.load_url(model_urls['inception_v3_google']))
        # self.down4 = down(512, 512)
        # self.up0 = up(4096, 1024, bilinear=False)
        self.up1 = up(2048, 768, bilinear=False)
        self.up2 = up(1536, 384, bilinear=False)
        self.up3 = up(768, 192, bilinear=False)
        self.up4 = up(384, 96, bilinear=False)
        self.up5 = up(192, 48, bilinear=False)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))#nn.AdaptiveAvgPool2d((1,1))#自适应最大还是平均池化
        # self.up2 = up(512, 128)
        # self.up3 = up(256, 64)
        # self.up4 = up(128, 64)
        self.outc = outconv(48, out_channels) 
        

    def forward(self, x, draft):
        #import pdb;pdb.set_trace() 
        # self._ini_inception()
        
        x1 = self.inc(x)
        x = self.remain0(x1)
        x3 = self.down1(x) 
        x = self.remain1(x3)
        x5 = self.down2(x)
        x = self.remain2(x5)
        x7 = self.down3(x)
        x8 = self.down4(x7)
        # x9 = self.down5(x8)    
        x9 = self.remain3(x8) #input是256时刚好是2048，1，1
        # x10 = self.avg_pool(x10)
        #add 
        x_inception = self.inceptionv1(draft)
        #x_inception = self.avg_pool(x_inception)       
        x10 = x9 + x_inception 
        # x11 = x10
        #import pdb;pdb.set_trace()
        # x = self.up0(x11, x9)
        x = self.up1(x10, x8)
        x = self.up2(x, x7)
        x = self.remain4(x)
        x = self.up3(x, x5)
        x = self.remain5(x)
        x = self.up4(x, x3)
        x = self.remain6(x)
        x = self.up5(x, x1)
        x = self.outc(x)
        return x




class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Discriminator, self).__init__()
        self.inc = inconv(in_channels, 64)
        self.remain0 = remain(64, 64)
        self.down1 = down(64, 192)
        self.remain1 = remain(192, 192)
        self.down2 = down(192, 384)
        self.remain2 = remain(384, 384)
        self.down3 = down(384, 768)
        self.outc = outconv(768, out_channels)

    def forward(self, x):
        #import pdb;pdb.set_trace()
        x = self.inc(x)
        x = self.remain0(x)
        x = self.down1(x)
        x = self.remain1(x)
        x = self.down2(x)
        x = self.remain2(x)
        x = self.down3(x)
        x = self.outc(x)

        return x

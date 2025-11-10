import torch.nn as nn
from einops.einops import rearrange
from .pos_enc import ImgPosEnc
# from .mlp_block import MlpBlock
import copy
from torch import FloatTensor, LongTensor

import math
import torch

from pytorch_wavelets import DWTForward

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, embed_dim: int, patch_size=8, in_chans=1):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H % 8 == 0 and W % 8 == 0, \
            f"Input image size ({H}、{W}) doesn't match model."
        x = self.proj(x)  # [4,256,h/8,w/8]
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DWTNet(nn.Module):
    def __init__(self, d_model):
        super(DWTNet, self).__init__()
        # self.patch_embed = PatchEmbed(embed_dim=d_model, patch_size=8, in_chans=4)
        self.out_channel = d_model
        self.pos_enc_2d = ImgPosEnc(256, normalize=True)
        self.norm = nn.LayerNorm(256)
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.wat = Waveletatt(256)
        # self.wata = Waveletattspace(256)
        # self.channel_att = ChannelAtt(256, 16)
        self.dwt = DWT2DDirect(J=1, wave='haar', mode='zero')
        self.bot1 = Bottleneck(4, 32, 64)
        self.bot2 = Bottleneck(64, 64, 128)
        self.bot3 = Bottleneck(128, 128, 256)

    def forward(self, x: FloatTensor, mask: LongTensor) -> FloatTensor:
        """ Frequency Feature Extractor in MFH

        @param x: FloatTensor [b, 1, h, w]
        @param mask: LongTensor [b, h, w]
        @return: FloatTensor [b, h, w, d]
        """
        x = self.padding(x)
        LL, LH, HL, HH = self.dwt(x)  # LL: [B, C, H/2, W/2]
        dwt_fea = torch.cat([LL, LH, HL, HH], dim=1)  # [B, 4C, H/2, W/2]
        # x = self.patch_embed(dwt_fea)  # [b,256,H//8,W//8]
        x = self.bot1(dwt_fea)
        x = self.bot2(x)
        x = self.bot3(x)
        x = self.wat(x)
        # x = self.wata(x)
        # x = self.pool(x)
        # x = self.channel_att(x)
        x = rearrange(x, "b d h w -> b h w d")
        x = self.pos_enc_2d(x, mask)
        x = self.norm(x) 

        return x
    
    def padding(self, x):
        """
        Pads the input tensor so that height and width are divisible by 16.
        Padding is applied symmetrically to keep the image centered.
        
        @param x: Tensor of shape [B, C, H, W]
        @return: Padded tensor [B, C, H', W'] where H' and W' are divisible by 16
        """
        B, C, H, W = x.size()
        
        # 计算目标尺寸（向上取整到最近的 16 倍数）
        target_H = math.ceil(H / 16) * 16
        target_W = math.ceil(W / 16) * 16

        # 计算各方向填充量
        pad_top = (target_H - H) // 2
        pad_bottom = target_H - H - pad_top
        pad_left = (target_W - W) // 2
        pad_right = target_W - W - pad_left

        # 应用 padding
        pad = nn.ZeroPad2d((pad_left, pad_right, pad_top, pad_bottom))
        return pad(x)


class ChannelAtt(nn.Module):
    def __init__(self, channel, reduction):
        super(ChannelAtt, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
    

class Waveletatt(nn.Module):
    def __init__(self, in_planes=3, norm_layer=nn.LayerNorm):
        super().__init__()
        wavename = 'haar'
        self.dwt = DWT2DDirect(J=1, wave=wavename) 
        
        # 使用Sequential包装卷积层实现通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_attention = nn.Sequential(
            nn.Linear(in_planes, in_planes // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // 16, in_planes, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: B, C, H, W
        """
        B, C, H, W = x.shape  
        LL, LH, HL, HH = self.dwt(x)   
        
        ll_mean = torch.mean(LL, dim=[2, 3])  # [B, C]
        hl_mean = torch.mean(HL, dim=[2, 3])
        lh_mean = torch.mean(LH, dim=[2, 3])
        hh_mean = torch.mean(HH, dim=[2, 3])
        
        # 应用平均池化
        y = ll_mean + hl_mean + lh_mean + hh_mean  # [B, C, 1, 1]
        
        # 使用Sequential包装的卷积实现通道注意力
        y = self.conv_attention(y).view(B, C, 1, 1)
        
        return x * y

class Waveletattspace(nn.Module):
    def __init__(self, in_planes=3, norm_layer=nn.LayerNorm):
        super().__init__()
        wavename = 'haar'

        self.dwt = DWT2DDirect(J=1, wave=wavename)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // 16),
            nn.ReLU(),
            nn.Linear(in_planes // 16, in_planes),
            nn.Sigmoid())

    def forward(self, x):
        """
        x: B, H*W, C
        """
        B, C, _, _= x.shape
        LL, LH, HL, HH = self.dwt(x)
        H_sum = HL + LH + HH
        y = torch.cat([LL, H_sum], dim=1)

        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return y * x

class DWT2DDirect(nn.Module):  # 继承nn.Module以自动管理设备
    def __init__(self, J=1, wave='haar', mode='zero'):
        super().__init__()
        self.dwt = DWTForward(J=J, wave=wave, mode=mode)
        
        # 显式注册子模块（如果DWTForward是nn.Module的子类）
        # 如果DWTForward是第三方模块，确保它本身能正确处理设备

    def forward(self, tensor):  # 使用forward而不是__call__
        # 输入tensor会自动继承模块所在设备
        Yl, Yh = self.dwt(tensor)
        LL = Yl
        LH = Yh[0][:, :, 0, :, :]
        HL = Yh[0][:, :, 1, :, :]
        HH = Yh[0][:, :, 2, :, :]
        return LL, LH, HL, HH


class Bottleneck(nn.Module):
    def __init__(self,in_channel,c1,c2):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=c1, kernel_size=3, padding=1)  # squeeze channels

        self.conv2 = nn.Conv2d(in_channels=c1*4, out_channels=c2, kernel_size=3, stride=1, padding=1)  # unsqueeze channels

        self.conv3 = nn.Conv2d(in_channels=c2, out_channels=c2, kernel_size=3, stride=1, padding=1)  # unsqueeze channels
        
        self.res_conv = nn.Conv2d(in_channels=in_channel, out_channels=c2, kernel_size=3, stride=2, padding=1)
                                
        self.dwt = DWT2DDirect(J=1, wave='haar', mode='zero')

        self.Leaky_relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        identity = self.Leaky_relu(self.res_conv(x))
        x = self.Leaky_relu(self.conv1(x))
        ll, lh, hl, hh = self.dwt(x)
        x = torch.cat((ll, lh, hl, hh), dim=1)
        x = self.Leaky_relu(self.conv2(x))
        x = self.Leaky_relu(self.conv3(x))
        x = x + identity
        return x

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class ChannelBranch(nn.Module):
    def __init__(self, in_channel, r=16):
        super(ChannelBranch, self).__init__()
        self.ch = nn.Sequential(
            Pooling(),
            nn.Linear(in_channel, in_channel // r),
            nn.BatchNorm1d(in_channel // r),
            nn.ReLU(),
            nn.Linear(in_channel // r, in_channel)
        )

    def forward(self, x):
        ch = self.ch(x) 
        ch_scale = ch.unsqueeze(2).unsqueeze(3)
        ch_scale_expanded = ch_scale.expand_as(x)
        return ch_scale_expanded

class SpatialBranch(nn.Module):
    def __init__(self, in_channel, r=16, dilation_val=4):
        super(SpatialBranch, self).__init__()

        self.spatial = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // r, kernel_size=1),
            nn.BatchNorm2d(in_channel // r),
            nn.ReLU(),

            nn.Conv2d(in_channel // r, in_channel // r, kernel_size=3,
                      padding=dilation_val, dilation=dilation_val),
            nn.BatchNorm2d(in_channel // r),
            nn.ReLU(),

            nn.Conv2d(in_channel // r, in_channel // r, kernel_size=3,
                      padding=dilation_val, dilation=dilation_val),
            nn.BatchNorm2d(in_channel // r),
            nn.ReLU(),

            nn.Conv2d(in_channel // r, 1, kernel_size=1)
        )

    def forward(self, x):
        output = self.spatial(x) 
        output = output.expand_as(x)
        return output

class Pooling(nn.Module):
    def __init__(self):
        super(Pooling, self).__init__()
        self.pooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch, channel, _, _ = x.size()
        pooled = self.pooling(x)
        pooled = pooled.squeeze(3).squeeze(2)
        return pooled


class BAM(nn.Module):
    def __init__(self, in_channel, r=16):
        super(BAM, self).__init__()
        self.channel_attention = ChannelBranch(in_channel, r)
        self.spatial_attention = SpatialBranch(in_channel, r)

    def forward(self, x):
        full_attention = 1 + F.sigmoid(self.channel_attention(x) + self.spatial_attention(x))
        return x * full_attention

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

class Pooling(nn.Module):
    def __init__(self):
        super(Pooling, self).__init__()
        self.pooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch, channel, _, _ = x.size()
        pooled = self.pooling(x)
        pooled = pooled.squeeze(3).squeeze(2)
        return pooled


class Channel_Attention_Block(nn.Module):
    def __init__(self, in_channel, r=16):
        super(Channel_Attention_Block, self).__init__()
        self.channel_attention = ChannelBranch(in_channel, r)

    def forward(self, x):
        full_attention = x * F.sigmoid(self.channel_attention(x))
        return full_attention

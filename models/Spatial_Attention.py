import torch
import math
import torch.nn as nn
import torch.nn.functional as F

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

class Spatial_Attention_Block(nn.Module):
    def __init__(self, in_channel, r=16):
        super(Spatial_Attention_Block, self).__init__()
        self.spatial_attention = SpatialBranch(in_channel, r)

    def forward(self, x):
        full_attention = x * F.sigmoid(self.spatial_attention(x))
        return full_attention

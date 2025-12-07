import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class PositionwiseAttention(nn.Module):
    def __init__(self, in_channels, key_channels, value_channels):
        super(PositionwiseAttention, self).__init__()
        
        self.query_conv = nn.Conv2d(in_channels, key_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, key_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, value_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        query = self.query_conv(x)
        key = self.key_conv(x)
        value = self.value_conv(x)
        
        query = query.view(query.size(0), -1, query.size(2) * query.size(3))
        key = key.view(key.size(0), -1, key.size(2) * key.size(3))
        value = value.view(value.size(0), -1, value.size(2) * value.size(3))
        
        key = key.permute(0, 2, 1)
        
        attention = torch.bmm(query, key)
        attention = torch.softmax(attention, dim=-1)
        
        out = torch.bmm(value, attention)
        out = out.view(x.size(0), -1, x.size(2), x.size(3))
        out = self.gamma * out + x
        
        return out

class Pooling(nn.Module):
    def __init__(self):
        super(Pooling, self).__init__()
        self.pooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # getting size of the input
        batch, channel, _, _ = x.size()
        pooled = self.pooling(x)
        pooled = pooled.squeeze(3).squeeze(2)
        return pooled

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

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) 
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = x + self.SpatialGate(x_out)
        return x_out

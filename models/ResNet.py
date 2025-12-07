from __future__ import absolute_import
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from .cbam import *
from .bam import *
from .Spatial_Attention import *
from .Channel_Attention import *
from .ECA_Net import *
from .Coordinate_Attention import *
from .HYP_ECA_Block import *


from aligned.HorizontalMaxPool2D import HorizontalMaxPool2d

__all__ = ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']

class ResNet18(nn.Module):
    def __init__(self, num_classes, loss={'softmax'}, aligned=False, **kwargs):
        super(ResNet18, self).__init__()
        self.loss = loss
        resnet18 = torchvision.models.resnet18(weights='ResNet18_Weights.DEFAULT')   # equivalent to imagenet weights
        self.base = nn.Sequential(*list(resnet18.children())[:-2])
       
        self.base[7][0].conv1.stride = (1, 1)
        self.base[7][0].downsample = nn.Sequential(
           nn.Conv2d(256, 512, kernel_size=1, stride=1, bias=False),
           nn.BatchNorm2d(512))
        self.base[7][0].downsample[0].stride = (1, 1)
               
        for name, module in self.base.named_modules():
            if isinstance( module, torchvision.models.resnet.BasicBlock):
                bottleneck = module
                num_features = bottleneck.conv2.out_channels
                if num_features == 256 or num_features == 512:
                    bottleneck.conv2 = nn.Sequential(
                            bottleneck.conv2,
                            Channel_Attention_Block(num_features))
        
        self.dropout = nn.Dropout(p=0.5)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(512, num_classes)
        self.feat_dim = 512 # feature dimension
        self.aligned = aligned
        self.horizon_pool = HorizontalMaxPool2d()
        if self.aligned:
            self.bn = nn.BatchNorm2d(512)
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=True)
        

    def forward(self, x):
        x = self.base(x)
        if not self.training:
            lf = self.horizon_pool(x)
        if self.aligned and self.training:
            lf = self.bn(x)
            lf = self.relu(lf)
            lf = self.horizon_pool(lf)
            lf = self.conv1(lf)
            x = self.bn1(x)
            x = self.relu1(x)
        if self.aligned or not self.training:
            lf = lf.view(lf.size()[0:3])
            lf = lf / torch.pow(lf,2).sum(dim=1, keepdim=True).clamp(min=1e-12).sqrt()
        
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        if not self.training:
            return f,lf
        f = self.dropout(f)
        y = self.classifier(f)
        if self.loss == {'softmax'}:
            return y
        elif self.loss == {'metric'}:
            if self.aligned: return  f, lf
            return f
        elif self.loss == {'softmax', 'metric'}:
            if self.aligned: return y, f, lf
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet34(nn.Module):
    def __init__(self, num_classes, loss={'softmax'}, aligned=False, **kwargs):
        super(ResNet34, self).__init__()
        self.loss = loss
        resnet34 = torchvision.models.resnet34(weights='ResNet34_Weights.DEFAULT')   # equivalent to imagenet weights
        self.base = nn.Sequential(*list(resnet34.children())[:-2])
         
        self.base[7][0].conv1.stride = (1, 1)
        self.base[7][0].downsample = nn.Sequential(
           nn.Conv2d(256, 512, kernel_size=1, stride=1, bias=False),
           nn.BatchNorm2d(512))
        self.base[7][0].downsample[0].stride = (1, 1)
               
        for name, module in self.base.named_modules():
            if isinstance( module, torchvision.models.resnet.BasicBlock):
                bottleneck = module
                num_features = bottleneck.conv2.out_channels
                if num_features == 256 or num_features == 512:
                    bottleneck.conv2 = nn.Sequential(
                            bottleneck.conv2,
                            Channel_Attention_Block(num_features))
        
        self.dropout = nn.Dropout(p=0.5)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(512, num_classes)
        self.feat_dim = 512 # feature dimension
        self.aligned = aligned
        self.horizon_pool = HorizontalMaxPool2d()
        if self.aligned:
            self.bn = nn.BatchNorm2d(512)
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=True)
        

    def forward(self, x):
        x = self.base(x)
        if not self.training:
            lf = self.horizon_pool(x)
        if self.aligned and self.training:
            lf = self.bn(x)
            lf = self.relu(lf)
            lf = self.horizon_pool(lf)
            lf = self.conv1(lf)

            x = self.bn1(x)
            x = self.relu1(x)
        if self.aligned or not self.training:
            lf = lf.view(lf.size()[0:3])
            lf = lf / torch.pow(lf,2).sum(dim=1, keepdim=True).clamp(min=1e-12).sqrt()
        
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        if not self.training:
            return f,lf
        f = self.dropout(f)
        y = self.classifier(f)
        if self.loss == {'softmax'}:
            return y
        elif self.loss == {'metric'}:
            if self.aligned: return  f, lf
            return f
        elif self.loss == {'softmax', 'metric'}:
            if self.aligned: return y, f, lf
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))



class ResNet50(nn.Module):
    def __init__(self, num_classes, loss={'softmax'}, aligned=False, **kwargs):
        super(ResNet50, self).__init__()
        self.loss = loss
        resnet50 = torchvision.models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V2')   # equivalent to imagenet weights
        self.base = nn.Sequential(*list(resnet50.children())[:-2])

        self.base[7][0].conv2.stride = (1, 1)
        self.base[7][0].downsample = nn.Sequential(
           nn.Conv2d(1024, 2048, kernel_size=1, stride=1, bias=False),
           nn.BatchNorm2d(2048))
        self.base[7][0].downsample[0].stride = (1, 1)
       
        for name, module in self.base.named_modules():
            if isinstance(module, torchvision.models.resnet.Bottleneck):
                bottleneck = module
                num_features = bottleneck.conv3.out_channels
                
                # block for CWA
                if num_features == 1024 or num_features == 2048:
                    bottleneck.conv3 = nn.Sequential(
                            bottleneck.conv3,
                            Channel_Attention_Block(num_features))

    
        self.dropout = nn.Dropout(p=0.5)
        self.bn1 = nn.BatchNorm2d(2048)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.classifier = nn.Linear(2048, num_classes)
        self.feat_dim = 2048 # feature dimension
        self.aligned = aligned
        self.horizon_pool = HorizontalMaxPool2d()
        if self.aligned:
            self.bn = nn.BatchNorm2d(2048)
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(2048, 128, kernel_size=1, stride=1, padding=0, bias=True)
        

    def forward(self, x):
        x = self.base(x)
        if not self.training:
            lf = self.horizon_pool(x)
        if self.aligned and self.training:
            lf = self.bn(x)
            lf = self.relu(lf)
            lf = self.horizon_pool(lf)
            lf = self.conv1(lf)

            x = self.bn1(x)
            x = self.relu1(x)
        if self.aligned or not self.training:
            lf = lf.view(lf.size()[0:3])
            lf = lf / torch.pow(lf,2).sum(dim=1, keepdim=True).clamp(min=1e-12).sqrt()
        
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        if not self.training:
            return f,lf
        f = self.dropout(f)
        y = self.classifier(f)
        if self.loss == {'softmax'}:
            return y
        elif self.loss == {'metric'}:
            if self.aligned: return  f, lf
            return f
        elif self.loss == {'softmax', 'metric'}:
            if self.aligned: return y, f, lf
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

class ResNet101(nn.Module):
    def __init__(self, num_classes, loss={'softmax'}, aligned=False, **kwargs):
        super(ResNet101, self).__init__()
        self.loss = loss
        resnet101 = torchvision.models.resnet101(weights='ResNet101_Weights.DEFAULT')
        self.base = nn.Sequential(*list(resnet101.children())[:-2])

        self.base[7][0].conv2.stride = (1, 1)
        self.base[7][0].downsample = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(2048))
        self.base[7][0].downsample[0].stride = (1, 1)
       
        for name, module in self.base.named_modules():
            if isinstance( module, torchvision.models.resnet.Bottleneck):
                bottleneck = module
                num_features = bottleneck.conv3.out_channels

                if num_features == 1024 or num_features == 2048:
                    bottleneck.conv3 = nn.Sequential(
                            bottleneck.conv3,
                           Channel_Attention_Block(num_features))


        self.dropout = nn.Dropout(p=0.5)
        self.bn1 = nn.BatchNorm2d(2048)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)


        self.classifier = nn.Linear(2048, num_classes)
        self.feat_dim = 2048 # feature dimension
        self.aligned = aligned
        self.horizon_pool = HorizontalMaxPool2d()
        if self.aligned:
            self.bn = nn.BatchNorm2d(2048)
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(2048, 128, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.base(x)
        if not self.training:
            lf = self.horizon_pool(x)
        if self.aligned and self.training:
            lf = self.bn(x)
            lf = self.relu(lf)
            lf = self.horizon_pool(lf)
            lf = self.conv1(lf)

            x = self.bn1(x)
            x = self.relu1(x)
        if self.aligned or not self.training:
            lf = lf.view(lf.size()[0:3])
            lf = lf / torch.pow(lf, 2).sum(dim=1, keepdim=True).clamp(min=1e-12).sqrt()
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        
        if not self.training:
            return f, lf
        f = self.dropout(f)
        y = self.classifier(f)
        if self.loss == {'softmax'}:
            return y
        elif self.loss == {'metric'}:
            if self.aligned: return f, lf
            return f
        elif self.loss == {'softmax', 'metric'}:
            if self.aligned: return y, f, lf
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet152(nn.Module):
    def __init__(self, num_classes, loss={'softmax'}, aligned=False, **kwargs):
        super(ResNet152, self).__init__()
        self.loss = loss
        resnet152 = torchvision.models.resnet152(weights='ResNet152_Weights.DEFAULT')
        self.base = nn.Sequential(*list(resnet152.children())[:-2])

        self.base[7][0].conv2.stride = (1, 1)
        self.base[7][0].downsample = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(2048))
        self.base[7][0].downsample[0].stride = (1, 1)
       
        for name, module in self.base.named_modules():
            if isinstance( module, torchvision.models.resnet.Bottleneck):
                bottleneck = module
                num_features = bottleneck.conv3.out_channels

                if num_features == 1024 or num_features == 2048:
                    bottleneck.conv3 = nn.Sequential(
                            bottleneck.conv3,
                            Channel_Attention_Block(num_features))

        self.dropout = nn.Dropout(p=0.5)
        self.bn1 = nn.BatchNorm2d(2048)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.classifier = nn.Linear(2048, num_classes)
        self.feat_dim = 2048 # feature dimension
        self.aligned = aligned
        self.horizon_pool = HorizontalMaxPool2d()
        if self.aligned:
            self.bn = nn.BatchNorm2d(2048)
            self.relu = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(2048, 128, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.base(x)
        if not self.training:
            lf = self.horizon_pool(x)
        if self.aligned and self.training:
            lf = self.bn(x)
            lf = self.relu(lf)
            lf = self.horizon_pool(lf)
            lf = self.conv1(lf)

            x = self.bn1(x)
            x = self.relu1(x)
        if self.aligned or not self.training:
            lf = lf.view(lf.size()[0:3])
            lf = lf / torch.pow(lf, 2).sum(dim=1, keepdim=True).clamp(min=1e-12).sqrt()
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)
        
        if not self.training:
            return f, lf
        f = self.dropout(f)
        y = self.classifier(f)
        if self.loss == {'softmax'}:
            return y
        elif self.loss == {'metric'}:
            if self.aligned: return f, lf
            return f
        elif self.loss == {'softmax', 'metric'}:
            if self.aligned: return y, f, lf
            return y, f
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))

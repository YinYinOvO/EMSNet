#!/usr/bin/python3
#coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
import torch
import numpy as np
import timm
import torch.nn.functional as F
def weight_init(module):
    for n, m in module.named_children():
        # print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        elif isinstance(m, nn.ReLU6):
            pass
        elif isinstance(m, nn.AdaptiveAvgPool2d):
            pass
        elif isinstance(m, nn.AvgPool2d):
            pass
        elif isinstance(m, nn.ModuleList):
            pass
        elif isinstance(m, nn.MaxPool2d):
            pass
        else:
            m.initialize()

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.resnet = timm.create_model(model_name="resnet50", pretrained=False, in_chans=3, features_only=True)
    def forward(self,x):
        out1, out2, out3, out4, out5 = self.resnet(x)
        return out1, out2, out3, out4, out5
    def initialize(self):
        pass

class ScoreLayer(nn.Module):
    def __init__(self, k):
        super(ScoreLayer, self).__init__()
        self.score = nn.Conv2d(k, 1, 1, 1)
    def forward(self, x):
        x = self.score(x)
        return x
    def initialize(self):
        weight_init(self)

class Conv2dBlock3x3(nn.Module):
    def __init__(self, left, right):
        super(Conv2dBlock3x3, self).__init__()
        self.conv = nn.Conv2d(left, right, 3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(right)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)

    def initialize(self):
        weight_init(self)

class Conv2dBlock1x1(nn.Module):
    def __init__(self, left, right):
        super(Conv2dBlock1x1, self).__init__()
        self.conv = nn.Conv2d(left, right, 1)
        self.bn = nn.BatchNorm2d(right)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)

    def initialize(self):
        weight_init(self)

class PEM(nn.Module):
    def __init__(self):
        super(PEM, self).__init__()
        # self.conv = nn.Conv2d(512,512,1)
        self.conv0 = nn.Conv2d(128, 64, 1, stride=1)
        self.conv1 = nn.Conv2d(64, 64, 3, stride=2)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=4)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=6)
        self.mix = nn.Conv2d(4*64, 128, 1)
        self.mix_ = nn.Conv2d(2*128, 64, 1)
        

    def forward(self, x):
        # x = self.conv(x)
        out0 = self.conv0(x)
        out1 = F.interpolate(self.conv1(out0), x.size()[2:], mode='bilinear')
        out2 = F.interpolate(self.conv2(out0+out1), x.size()[2:], mode='bilinear')
        out3 = F.interpolate(self.conv3(out0+out2), x.size()[2:], mode='bilinear')
        x = self.mix_(torch.cat((x, self.mix(torch.cat((out0, out1, out2, out3), dim=1))), dim=1))
        return x

    def initialize(self):
        weight_init(self)

class Conv2dBlock(nn.Module):
    def __init__(self, left, right):
        super(Conv2dBlock, self).__init__()
        self.conv1 = Conv2dBlock3x3(left,right)
        self.conv2 = Conv2dBlock1x1(right,right)

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        return x
    def initialize(self):
        weight_init(self)

class MultipleUNetModule(nn.Module):
    def __init__(self,in_ch, out_ch):
        super(MultipleUNetModule,self).__init__()

        self.conv_b = Conv2dBlock1x1(in_ch, in_ch)
        self.conv_c = Conv2dBlock1x1(in_ch, in_ch)
        self.conv_e = Conv2dBlock(in_ch, out_ch)
        self.superconvs = nn.ModuleList([InbuiltUNet(i+1, in_ch//4) for i in range(4)])

    def forward(self,x):
        xout = self.conv_b(x)
        xouts = list(torch.split(xout, x.size(1)//4, dim=1))
        for index,unet in enumerate(self.superconvs):
            xouts[index] = unet(xouts[index])
            if index!=  len(self.superconvs)-1:
                xouts[index+1] = xouts[index+1] + xouts[index]
        xout = self.conv_e((self.conv_c(torch.cat(xouts, dim=1))+x))
        return xout
    def initialize(self):
        weight_init(self)

class InbuiltUNet(nn.Module):
    def __init__(self, num, channel):
        super(InbuiltUNet,self).__init__()
        self.enconv = nn.ModuleList([Conv2dBlock(channel, channel)  for _ in range(num)])
        self.deconv = nn.ModuleList([Conv2dBlock(channel, channel)  for _ in range(num)])
        self.maxpool = nn.AvgPool2d(2,stride=2)
    def forward(self,x):
        xout = []
        tempout = x
        for idx,conv in enumerate(self.enconv):
            xout.append(tempout)
            if idx != len(self.enconv) - 1:
                tempout = self.maxpool(tempout)
            tempout = conv(tempout)
            
        for index,conv in enumerate(self.deconv):
            tempout = conv((xout[len(self.deconv)-1-index]+F.interpolate(tempout, xout[len(self.deconv)-1-index].size()[2:], mode='bilinear')))
        return tempout
    def initialize(self):
        weight_init(self)

class CompressModule(nn.Module):
    def __init__(self):
        super(CompressModule, self).__init__()
        self.compress1 = Conv2dBlock(64, 64)
        self.compress2 = Conv2dBlock(256, 64)
        self.compress3 = Conv2dBlock(512, 64)
        self.compress4 = Conv2dBlock(1024, 64)
        self.compress5 = Conv2dBlock(2048, 128)
        self.MUM0 = MultipleUNetModule(64,64)
        self.MUM1 = MultipleUNetModule(64,64)
        self.MUM2 = MultipleUNetModule(64,64)
        self.MUM3 = MultipleUNetModule(64,64)
        self.MUM4 = MultipleUNetModule(128,64)
        self.PEM = PEM()

    def forward(self, out1, out2, out3, out4, out5):
        out1 = self.MUM0(self.compress1(out1))
        out2 = self.MUM1(self.compress2(out2))
        out3 = self.MUM2(self.compress3(out3))
        out4 = self.MUM3(self.compress4(out4))
        temp = self.compress5(out5)
        out5_1 = self.MUM4(temp)
        out5_2 = self.PEM(temp)
        
        return out1, out2, out3, out4, out5_1, out5_2

    def initialize(self):
        weight_init(self)


class Decoder1(nn.Module):
    def __init__(self):
        super(Decoder1, self).__init__()
        self.MUM0 = MultipleUNetModule(128,64)
        self.MUM1 = MultipleUNetModule(128,64)
        self.MUM2 = MultipleUNetModule(128,64)
        self.MUM3 = MultipleUNetModule(128,64)
        self.MUM4 = MultipleUNetModule(128,64)

    def forward(self, out1, out2, out3, out4, out5_1, out5_2):
        resinfo = self.MUM4(torch.cat((out5_1, out5_2),dim=1))
        resinfo4 = self.MUM3(torch.cat((F.interpolate(resinfo, out4.size()[2:], mode='bilinear'), out4), dim=1) )
        resinfo3 = self.MUM2(torch.cat((F.interpolate(resinfo4, out3.size()[2:], mode='bilinear'), out3), dim=1))
        resinfo2 = self.MUM1(torch.cat((F.interpolate(resinfo3, out2.size()[2:], mode='bilinear'), out2), dim=1) )
        resinfo1 = self.MUM0(torch.cat((F.interpolate(resinfo2, out1.size()[2:], mode='bilinear'), out1),dim=1))
        
        return resinfo1,resinfo2, resinfo3, resinfo4
    def initialize(self):
        weight_init(self)

class FocusNet(nn.Module):
    def __init__(self, cfg):
        super(FocusNet, self).__init__()
        self.cfg = cfg
        self.resnet = ResNet()
        self.CompressModule = CompressModule()
        self.decoder = Decoder1()
        self.predscore = ScoreLayer(64)
        self.initialize()
    def forward(self, x):
        out1, out2, out3, out4, out5 = self.resnet(x)
        out1, out2, out3, out4, out5_1, out5_2 = self.CompressModule(out1, out2, out3, out4, out5)
        resinfo1, resinfo2, resinfo3, resinfo4 = self.decoder(out1, out2, out3, out4, out5_1, out5_2)
        pred1 = torch.sigmoid(self.predscore(F.interpolate(resinfo1, x.size()[2:], mode='bilinear')))
        pred2 = torch.sigmoid(self.predscore(F.interpolate(resinfo2, x.size()[2:], mode='bilinear')))
        pred3 = torch.sigmoid(self.predscore(F.interpolate(resinfo3, x.size()[2:], mode='bilinear')))
        pred4 = torch.sigmoid(self.predscore(F.interpolate(resinfo4, x.size()[2:], mode='bilinear')))
        return pred1, pred2, pred3, pred4
    def initialize(self):
        weight_init(self)

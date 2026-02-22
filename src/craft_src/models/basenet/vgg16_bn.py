from collections import namedtuple
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision import models
from torchvision.models import VGG16_BN_Weights


def init_weights(modules):
    """Efficient initialization using Kaiming initialization"""
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class vgg16_bn(torch.nn.Module):
    def __init__(self, pretrained: bool = True, freeze: bool = True):
        super().__init__()
        
        # Use modern weights enum instead of URLs
        weights = VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None
        vgg_model = models.vgg16_bn(weights=weights)
        vgg_features = vgg_model.features
        
        # Create slices more efficiently
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        
        # Define slice ranges
        slice_ranges = [(0, 12), (12, 19), (19, 29), (29, 39)]
        slices = [self.slice1, self.slice2, self.slice3, self.slice4]
        
        for slice_module, (start, end) in zip(slices, slice_ranges):
            for x in range(start, end):
                slice_module.add_module(str(x), vgg_features[x])

        # fc6, fc7 without atrous conv
        self.slice5 = torch.nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.Conv2d(1024, 1024, kernel_size=1),
        )

        if not pretrained:
            init_weights(self.slice1.modules())
            init_weights(self.slice2.modules())
            init_weights(self.slice3.modules())
            init_weights(self.slice4.modules())

        init_weights(self.slice5.modules())  # no pretrained model for fc6 and fc7

        if freeze:
            for param in self.slice1.parameters():  # only first conv
                param.requires_grad = False

    def forward(self, x: torch.Tensor):
        """
        Forward pass with optimized intermediate storage
        """
        h = self.slice1(x)
        h_relu2_2 = h
        h = self.slice2(h)
        h_relu3_2 = h
        h = self.slice3(h)
        h_relu4_3 = h
        h = self.slice4(h)
        h_relu5_3 = h
        h = self.slice5(h)
        h_fc7 = h
        
        # Create named tuple more efficiently
        VggOutputs = namedtuple("VggOutputs", ["fc7", "relu5_3", "relu4_3", "relu3_2", "relu2_2"])
        return VggOutputs(h_fc7, h_relu5_3, h_relu4_3, h_relu3_2, h_relu2_2)

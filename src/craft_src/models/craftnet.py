"""
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from craft_text_detector.models.basenet.vgg16_bn import vgg16_bn, init_weights


class DoubleConv(nn.Module):
    """Optimized double convolution block with better initialization"""
    
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class CraftNet(nn.Module):
    """
    CRAFT Text Detection Network with modern optimizations
    
    Args:
        pretrained: Whether to use pretrained VGG backbone
        freeze: Whether to freeze VGG backbone parameters
        num_classes: Number of output classes (default: 2 for text/link)
    """
    
    def __init__(self, pretrained: bool = False, freeze: bool = False, num_classes: int = 2):
        super().__init__()

        # Base network
        self.basenet = vgg16_bn(pretrained, freeze)

        # U-Net decoder with optimized blocks
        self.upconv1 = DoubleConv(1024, 512, 256)
        self.upconv2 = DoubleConv(512, 256, 128)
        self.upconv3 = DoubleConv(256, 128, 64)
        self.upconv4 = DoubleConv(128, 64, 32)

        # Classification head with proper initialization
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=1),
        )

        # Initialize classification head
        self._init_classifier()
    
    def _init_classifier(self):
        """Initialize classification head with proper weights"""
        for m in self.conv_cls.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optimized interpolation and concatenation
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            Tuple of (predictions, features):
            - predictions: (B, H, W, num_classes) 
            - features: (B, 32, H, W)
        """
        # Extract multi-scale features from backbone
        sources = self.basenet(x)
        
        # U-Net decoder with skip connections
        # Stage 1: Combine fc7 and relu5_3
        y = torch.cat([sources.fc7, sources.relu5_3], dim=1)
        y = self.upconv1(y)

        # Stage 2: Upsample and combine with relu4_3
        y = F.interpolate(
            y, size=sources.relu4_3.size()[2:], mode="bilinear", align_corners=False
        )
        y = torch.cat([y, sources.relu4_3], dim=1)
        y = self.upconv2(y)

        # Stage 3: Upsample and combine with relu3_2
        y = F.interpolate(
            y, size=sources.relu3_2.size()[2:], mode="bilinear", align_corners=False
        )
        y = torch.cat([y, sources.relu3_2], dim=1)
        y = self.upconv3(y)

        # Stage 4: Upsample and combine with relu2_2
        y = F.interpolate(
            y, size=sources.relu2_2.size()[2:], mode="bilinear", align_corners=False
        )
        y = torch.cat([y, sources.relu2_2], dim=1)
        features = self.upconv4(y)

        # Generate predictions
        predictions = self.conv_cls(features)

        # Return in BHWC format for compatibility
        return predictions.permute(0, 2, 3, 1), features
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simplified prediction method that returns only the predictions
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            predictions: (B, H, W, num_classes)
        """
        predictions, _ = self.forward(x)
        return predictions


    def get_feature_maps(self, x: torch.Tensor) -> dict:
        """
        Extract intermediate feature maps for analysis/debugging
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            Dictionary containing all intermediate feature maps
        """
        sources = self.basenet(x)
        
        # Store all intermediate features
        feature_maps = {
            'backbone_fc7': sources.fc7,
            'backbone_relu5_3': sources.relu5_3, 
            'backbone_relu4_3': sources.relu4_3,
            'backbone_relu3_2': sources.relu3_2,
            'backbone_relu2_2': sources.relu2_2,
        }
        
        # Forward through decoder and store intermediate results
        y = torch.cat([sources.fc7, sources.relu5_3], dim=1)
        y = self.upconv1(y)
        feature_maps['upconv1'] = y
        
        y = F.interpolate(y, size=sources.relu4_3.size()[2:], mode="bilinear", align_corners=False)
        y = torch.cat([y, sources.relu4_3], dim=1)
        y = self.upconv2(y)
        feature_maps['upconv2'] = y
        
        y = F.interpolate(y, size=sources.relu3_2.size()[2:], mode="bilinear", align_corners=False)
        y = torch.cat([y, sources.relu3_2], dim=1)
        y = self.upconv3(y)
        feature_maps['upconv3'] = y
        
        y = F.interpolate(y, size=sources.relu2_2.size()[2:], mode="bilinear", align_corners=False)
        y = torch.cat([y, sources.relu2_2], dim=1)
        features = self.upconv4(y)
        feature_maps['upconv4'] = features
        
        predictions = self.conv_cls(features)
        feature_maps['predictions'] = predictions
        
        return feature_maps


def create_craft_model(pretrained: bool = True, freeze: bool = False, 
                      num_classes: int = 2) -> CraftNet:
    """
    Factory function to create a CRAFT model with recommended settings
    
    Args:
        pretrained: Whether to use pretrained VGG backbone
        freeze: Whether to freeze VGG backbone parameters  
        num_classes: Number of output classes
        
    Returns:
        Initialized CraftNet model
    """
    model = CraftNet(pretrained=pretrained, freeze=freeze, num_classes=num_classes)
    return model


if __name__ == "__main__":
    # Test the model
    print("Testing CraftNet...")
    
    # Create model
    model = create_craft_model(pretrained=True, freeze=False)
    
    # Test with different input sizes
    test_sizes = [(768, 768), (512, 512), (1024, 1024)]
    
    for h, w in test_sizes:
        print(f"\nTesting with input size: {h}x{w}")
        
        # Create test input
        test_input = torch.randn(1, 3, h, w)
        
        # Test forward pass
        with torch.no_grad():
            predictions, features = model(test_input)
            print(f"  Predictions shape: {predictions.shape}")
            print(f"  Features shape: {features.shape}")
        
        # Test prediction method
        with torch.no_grad():
            pred_only = model.predict(test_input)
            print(f"  Predict-only shape: {pred_only.shape}")
    
    # Test feature map extraction
    print(f"\nTesting feature map extraction...")
    test_input = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        feature_maps = model.get_feature_maps(test_input)
        print(f"  Number of feature maps: {len(feature_maps)}")
        for name, tensor in feature_maps.items():
            print(f"    {name}: {tensor.shape}")
    
    print("\nCraftNet test completed successfully!")

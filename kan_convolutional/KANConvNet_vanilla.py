import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import sys
import os
sys.path.append('./kan_convolutional')
from KANConv import KAN_Convolutional_Layer
from KANLinear import KANLinear


class KANConvNetVanilla(nn.Module):
    """
    KAN Convolutional Network without Lightning wrapper
    """
    
    def __init__(
        self,
        num_classes: int = 200,
        in_channels: int = 3,
        # Layer configurations: (out_channels, kernel_size, stride, padding)
        conv_layers: List[Tuple[int, Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = None,
        # KAN parameters
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation=torch.nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: tuple = [-1, 1],
    ):
        super().__init__()
        
        # Default layer configuration for ImageNet200 (64x64 input)
        if conv_layers is None:
            conv_layers = [
                (16, (7, 7), (2, 2), (3, 3)),   # 64x64 -> 32x32
                (32, (3, 3), (2, 2), (1, 1)),   # 32x32 -> 16x16
                (64, (3, 3), (2, 2), (1, 1)),   # 16x16 -> 8x8
            ]
        
        self.conv_layers = nn.ModuleList()
        
        # Create convolutional layers
        current_channels = in_channels
        for i, (out_channels, kernel_size, stride, padding) in enumerate(conv_layers):
            conv_layer = KAN_Convolutional_Layer(
                in_channels=current_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range
            )
            self.conv_layers.append(conv_layer)
            current_channels = out_channels
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final classification layer
        self.classifier = KANLinear(
            in_features=current_channels,
            out_features=num_classes,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range
        )
        
    def forward(self, x):
        # Convolutional layers
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            x = F.relu(x)
            if i < len(self.conv_layers) - 1:  # Don't pool after the last conv layer
                x = F.max_pool2d(x, 2, 2)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    def get_regularization_loss(self, regularize_activation: float = 1.0, regularize_entropy: float = 1.0):
        """Compute regularization loss from all KAN layers"""
        reg_loss = 0.0
        
        # Add regularization from convolutional layers
        for conv_layer in self.conv_layers:
            for conv in conv_layer.convs:
                reg_loss += conv.conv.regularization_loss(
                    regularize_activation, 
                    regularize_entropy
                )
        
        # Add regularization from classifier
        reg_loss += self.classifier.regularization_loss(
            regularize_activation, 
            regularize_entropy
        )
        
        return reg_loss
    
    def count_parameters(self):
        """Count total number of parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        } 
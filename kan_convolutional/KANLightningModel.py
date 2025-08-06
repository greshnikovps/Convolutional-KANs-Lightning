import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy
from typing import Optional, Dict, Any, List, Tuple
import sys
import os
sys.path.append('./kan_convolutional')
from KANConv import KAN_Convolutional_Layer
from KANLinear import KANLinear


class KANConvNet(pl.LightningModule):
    """
    KAN Convolutional Network wrapped in PyTorch Lightning
    """
    
    def __init__(
        self,
        num_classes: int = 1000,
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
        # Training parameters
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        regularize_activation: float = 1.0,
        regularize_entropy: float = 1.0,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Default layer configuration for a smaller network
        if conv_layers is None:
            conv_layers = [
                (16, (5, 5), (2, 2), (2, 2)),   # 3 -> 16
                (32, (3, 3), (2, 2), (1, 1)),    # 16 -> 32
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
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes, top_k=1)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes, top_k=1)
        self.train_acc_top5 = Accuracy(task='multiclass', num_classes=num_classes, top_k=5)
        self.val_acc_top5 = Accuracy(task='multiclass', num_classes=num_classes, top_k=5)
        
        # Regularization parameters
        self.regularize_activation = regularize_activation
        self.regularize_entropy = regularize_entropy
        
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
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Add regularization loss
        reg_loss = self._get_regularization_loss()
        total_loss = loss + reg_loss
        
        # Metrics
        self.train_acc(logits, y)
        self.train_acc_top5(logits, y)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_reg_loss', reg_loss, prog_bar=True)
        self.log('train_total_loss', total_loss, prog_bar=True)
        self.log('train_acc', self.train_acc, prog_bar=True)
        self.log('train_acc_top5', self.train_acc_top5, prog_bar=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Metrics
        self.val_acc(logits, y)
        self.val_acc_top5(logits, y)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)
        self.log('val_acc_top5', self.val_acc_top5, prog_bar=True)
        
        return loss
    
    def _get_regularization_loss(self):
        """Compute regularization loss from all KAN layers"""
        reg_loss = 0.0
        
        # Add regularization from convolutional layers
        for conv_layer in self.conv_layers:
            for conv in conv_layer.convs:
                reg_loss += conv.conv.regularization_loss(
                    self.regularize_activation, 
                    self.regularize_entropy
                )
        
        # Add regularization from classifier
        reg_loss += self.classifier.regularization_loss(
            self.regularize_activation, 
            self.regularize_entropy
        )
        
        return reg_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=100, 
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
    
    def count_parameters(self):
        """Count total number of parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        } 
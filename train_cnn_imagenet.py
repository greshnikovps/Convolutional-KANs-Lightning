import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import sys
import os
import argparse
from datetime import datetime
from torchmetrics import Accuracy
from typing import List, Tuple


class CNNConvNet(pl.LightningModule):
    """
    Standard CNN for comparison with KAN model
    """
    
    def __init__(
        self,
        num_classes: int = 1000,
        in_channels: int = 3,
        # Layer configurations: (out_channels, kernel_size, stride, padding)
        conv_layers: List[Tuple[int, Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = None,
        # Training parameters
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
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
        self.bn_layers = nn.ModuleList()
        
        # Create convolutional layers
        current_channels = in_channels
        for out_channels, kernel_size, stride, padding in conv_layers:
            conv_layer = nn.Conv2d(current_channels, out_channels, kernel_size, stride, padding)
            bn_layer = nn.BatchNorm2d(out_channels)
            
            self.conv_layers.append(conv_layer)
            self.bn_layers.append(bn_layer)
            current_channels = out_channels
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final classification layer
        self.classifier = nn.Linear(current_channels, num_classes)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics
        self.train_acc = Accuracy(task='multiclass', num_classes=num_classes, top_k=1)
        self.val_acc = Accuracy(task='multiclass', num_classes=num_classes, top_k=1)
        self.train_acc_top5 = Accuracy(task='multiclass', num_classes=num_classes, top_k=5)
        self.val_acc_top5 = Accuracy(task='multiclass', num_classes=num_classes, top_k=5)
        
    def forward(self, x):
        # Convolutional layers
        for i, (conv_layer, bn_layer) in enumerate(zip(self.conv_layers, self.bn_layers)):
            x = conv_layer(x)
            x = bn_layer(x)
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
        
        # Metrics
        self.train_acc(logits, y)
        self.train_acc_top5(logits, y)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc, prog_bar=True)
        self.log('train_acc_top5', self.train_acc_top5, prog_bar=True)
        
        return loss
    
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
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Metrics
        self.val_acc(logits, y)  # Reuse val_acc for test
        self.val_acc_top5(logits, y)  # Reuse val_acc_top5 for test
        
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', self.val_acc, prog_bar=True)
        self.log('test_acc_top5', self.val_acc_top5, prog_bar=True)
        
        return loss
    
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


def create_imagenet_dataloaders(data_dir, batch_size=32, num_workers=4):
    """
    Create ImageNet dataloaders with standard transforms
    """
    # Standard ImageNet transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(data_dir, 'train'),
        transform=train_transform
    )
    
    val_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(data_dir, 'val'),
        transform=val_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description='Train CNN model on ImageNet')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to ImageNet dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for dataloader')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--precision', type=int, default=16, help='Training precision (16 or 32)')
    
    args = parser.parse_args()
    
    # Create model
    model = CNNConvNet(
        num_classes=1000,
        in_channels=3,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Print model information
    param_info = model.count_parameters()
    print("=" * 50)
    print("CNN MODEL INFORMATION")
    print("=" * 50)
    print(f"Total parameters: {param_info['total_parameters']:,}")
    print(f"Trainable parameters: {param_info['trainable_parameters']:,}")
    print(f"Model size: {param_info['total_parameters'] * 4 / 1024 / 1024:.2f} MB")
    print("=" * 50)
    
    # Create dataloaders
    print("Creating ImageNet dataloaders...")
    train_loader, val_loader = create_imagenet_dataloaders(
        args.data_dir, 
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath='checkpoints/cnn_imagenet',
        filename='cnn_imagenet-{epoch:02d}-{val_acc:.3f}',
        save_top_k=3,
        mode='max',
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Setup logger
    logger = TensorBoardLogger(
        'logs', 
        name=f'cnn_imagenet_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus if args.gpus > 0 else None,
        precision=args.precision,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=50,
        val_check_interval=0.25,  # Validate 4 times per epoch
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,  # Effective batch size = batch_size * 2
    )
    
    # Train the model
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)
    
    # Evaluate on validation set
    print("Evaluating final model...")
    results = trainer.test(model, val_loader)
    
    print("=" * 50)
    print("FINAL EVALUATION RESULTS")
    print("=" * 50)
    for key, value in results[0].items():
        print(f"{key}: {value:.4f}")
    print("=" * 50)
    
    # Save final model
    trainer.save_checkpoint(f"cnn_imagenet_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ckpt")
    print("Training completed!")


if __name__ == "__main__":
    main() 
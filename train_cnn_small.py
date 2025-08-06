import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import sys
import os
import argparse
from datetime import datetime
from typing import List, Tuple

# Add the train_cnn_imagenet module to the path
from train_cnn_imagenet import CNNConvNet


def create_small_imagenet_dataloaders(data_dir, batch_size=32, num_workers=4, subset_size=1000):
    """
    Create small ImageNet dataloaders with subset for faster training
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
    
    # Create subsets for faster training
    train_subset = Subset(train_dataset, range(min(subset_size, len(train_dataset))))
    val_subset = Subset(val_dataset, range(min(subset_size // 4, len(val_dataset))))
    
    # Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def parse_layer_config(layer_str: str) -> List[Tuple[int, Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
    """
    Parse layer configuration from string format: "16,5,2,2;32,3,2,1"
    Each layer is: out_channels,kernel_size,stride,padding
    """
    layers = []
    for layer in layer_str.split(';'):
        parts = layer.split(',')
        if len(parts) == 4:
            out_channels = int(parts[0])
            kernel_size = (int(parts[1]), int(parts[1]))  # Square kernel
            stride = (int(parts[2]), int(parts[2]))       # Square stride
            padding = (int(parts[3]), int(parts[3]))      # Square padding
            layers.append((out_channels, kernel_size, stride, padding))
    return layers


def main():
    parser = argparse.ArgumentParser(description='Train CNN model on small ImageNet subset')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to ImageNet dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for dataloader')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--precision', type=int, default=32, help='Training precision (16 or 32)')
    parser.add_argument('--subset_size', type=int, default=1000, help='Number of training samples to use')
    
    # Layer configuration
    parser.add_argument('--layers', type=str, default="16,5,2,2;32,3,2,1", 
                       help='Layer configuration: "out_channels,kernel_size,stride,padding;..."')
    
    args = parser.parse_args()
    
    # Parse layer configuration
    conv_layers = parse_layer_config(args.layers)
    
    # Create model
    model = CNNConvNet(
        num_classes=1000,
        in_channels=3,
        conv_layers=conv_layers,
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
    print(f"Layer configuration: {args.layers}")
    print(f"Training on subset of {args.subset_size} samples")
    print("=" * 50)
    
    # Create dataloaders
    print("Creating small ImageNet dataloaders...")
    train_loader, val_loader = create_small_imagenet_dataloaders(
        args.data_dir, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        subset_size=args.subset_size
    )
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath='checkpoints/cnn_small',
        filename='cnn_small-{epoch:02d}-{val_acc:.3f}',
        save_top_k=3,
        mode='max',
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Setup logger
    logger = TensorBoardLogger(
        'logs', 
        name=f'cnn_small_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus if args.gpus > 0 else None,
        precision=args.precision,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=10,
        val_check_interval=0.5,  # Validate 2 times per epoch
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
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
    trainer.save_checkpoint(f"cnn_small_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ckpt")
    print("Training completed!")


if __name__ == "__main__":
    main() 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import sys
import os
import argparse
from datetime import datetime
from typing import List, Tuple
import time
from tqdm import tqdm

# Add the kan_convolutional directory to the path
sys.path.append('./kan_convolutional')
from KANConvNet_vanilla import KANConvNetVanilla


def create_imagenet200_dataloaders(data_dir, batch_size=32, num_workers=4):
    """
    Create ImageNet200 dataloaders with standard transforms
    """
    # Standard transforms for 64x64 images
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
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


def parse_layer_config(layer_str: str) -> List[Tuple[int, Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
    """
    Parse layer configuration from string format: "32,7,2,3;64,3,2,1;128,3,2,1"
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


def train_epoch(model, train_loader, criterion, optimizer, device, regularize_activation=1.0, regularize_entropy=1.0):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, target) in enumerate(pbar):
        # Move data to device (DataParallel handles the rest)
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        # Compute main loss
        loss = criterion(output, target)
        
        # Add regularization loss
        reg_loss = model.get_regularization_loss(regularize_activation, regularize_entropy)
        total_loss_batch = loss + reg_loss
        
        total_loss_batch.backward()
        optimizer.step()
        
        total_loss += total_loss_batch.item()
        
        # Calculate accuracy
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{total_loss_batch.item():.4f}',
            'Reg Loss': f'{reg_loss.item():.4f}',
            'Acc': f'{100. * correct / total:.2f}%'
        })
    
    return total_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for data, target in pbar:
            # Move data to device (DataParallel handles the rest)
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })
    
    return total_loss / len(val_loader), 100. * correct / total


def main():
    parser = argparse.ArgumentParser(description='Train KAN model on ImageNet200 (vanilla PyTorch)')
    parser.add_argument('--data_dir', type=str, default='./imagenet200/tiny-imagenet-200', 
                       help='Path to ImageNet200 dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for dataloader')
    parser.add_argument('--gpu', type=int, default=None, help='Specific GPU device ID (if None, use all available GPUs)')
    
    # KAN-specific parameters
    parser.add_argument('--grid_size', type=int, default=5, help='Grid size for KAN layers')
    parser.add_argument('--spline_order', type=int, default=3, help='Spline order for KAN layers')
    parser.add_argument('--scale_noise', type=float, default=0.1, help='Scale noise for KAN layers')
    parser.add_argument('--scale_base', type=float, default=1.0, help='Scale base for KAN layers')
    parser.add_argument('--scale_spline', type=float, default=1.0, help='Scale spline for KAN layers')
    parser.add_argument('--regularize_activation', type=float, default=1.0, help='Activation regularization weight')
    parser.add_argument('--regularize_entropy', type=float, default=1.0, help='Entropy regularization weight')
    
    # Layer configuration
    parser.add_argument('--layers', type=str, default="16,7,2,3;32,3,2,1;64,3,2,1", 
                       help='Layer configuration: "out_channels,kernel_size,stride,padding;..."')
    
    args = parser.parse_args()
    
    # Set device
    if torch.cuda.is_available():
        if args.gpu is not None:
            device = torch.device(f'cuda:{args.gpu}')
            print(f"Using specific GPU: {device}")
        else:
            device = torch.device('cuda')
            print(f"Using all available GPUs: {torch.cuda.device_count()} GPUs")
    else:
        device = torch.device('cpu')
        print(f"Using device: {device}")
    
    # Parse layer configuration
    conv_layers = parse_layer_config(args.layers)
    
    # Create model
    model = KANConvNetVanilla(
        num_classes=200,
        in_channels=3,
        conv_layers=conv_layers,
        grid_size=args.grid_size,
        spline_order=args.spline_order,
        scale_noise=args.scale_noise,
        scale_base=args.scale_base,
        scale_spline=args.scale_spline,
    )
    
    model = model.to(device)
    
    # Use DataParallel if multiple GPUs are available and no specific GPU is specified
    if torch.cuda.is_available() and args.gpu is None and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
    
    # Print model information
    param_info = model.count_parameters()
    print("=" * 50)
    print("KAN MODEL INFORMATION (VANILLA)")
    print("=" * 50)
    print(f"Total parameters: {param_info['total_parameters']:,}")
    print(f"Trainable parameters: {param_info['trainable_parameters']:,}")
    print(f"Model size: {param_info['total_parameters'] * 4 / 1024 / 1024:.2f} MB")
    print(f"Layer configuration: {args.layers}")
    print("=" * 50)
    
    # Create dataloaders
    print("Creating ImageNet200 dataloaders...")
    train_loader, val_loader = create_imagenet200_dataloaders(
        args.data_dir, 
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Setup training components
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs, 
        eta_min=1e-6
    )
    
    # Training loop
    print("Starting training...")
    best_val_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            args.regularize_activation, args.regularize_entropy
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Print epoch summary
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save the model state dict (DataParallel handles this automatically)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'args': args
            }, f"checkpoints/kan_vanilla_imagenet200_best.pt")
            print(f"New best model saved! Val Acc: {val_acc:.2f}%")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            # Save the model state dict (DataParallel handles this automatically)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'args': args
            }, f"checkpoints/kan_vanilla_imagenet200_epoch_{epoch+1}.pt")
    
    # Save final model
    # Save the model state dict (DataParallel handles this automatically)
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'args': args
    }, f"checkpoints/kan_vanilla_imagenet200_final.pt")
    
    print("=" * 50)
    print("TRAINING COMPLETED!")
    print("=" * 50)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Final validation accuracy: {val_acc:.2f}%")


if __name__ == "__main__":
    main() 
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

# Add the architectures_28x28 directory to the path
sys.path.append('./architectures_28x28')
from KKAN import KKAN_Small


class KKAN_Small_Lightning(pl.LightningModule):
    """
    KKAN_Small wrapped in PyTorch Lightning for MNIST training
    """
    
    def __init__(
        self,
        grid_size: int = 5,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Create KKAN_Small model
        self.model = KKAN_Small(grid_size=grid_size)
        
        # Loss function
        self.criterion = nn.NLLLoss()  # For log_softmax output
        
        # Metrics
        self.train_acc = Accuracy(task='multiclass', num_classes=10)
        self.val_acc = Accuracy(task='multiclass', num_classes=10)
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Metrics
        self.train_acc(logits, y)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Metrics
        self.val_acc(logits, y)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Metrics
        self.val_acc(logits, y)
        
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', self.val_acc, prog_bar=True)
        
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


def create_mnist_dataloaders(batch_size=32, num_workers=2):
    """
    Create MNIST dataloaders
    """
    # MNIST transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Create datasets
    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description='Train KKAN_Small on MNIST')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for dataloader')
    parser.add_argument('--gpus', type=int, default=0, help='Number of GPUs to use')
    parser.add_argument('--precision', type=int, default=32, help='Training precision (16 or 32)')
    parser.add_argument('--grid_size', type=int, default=5, help='Grid size for KAN layers')
    
    args = parser.parse_args()
    
    # Create model
    model = KKAN_Small_Lightning(
        grid_size=args.grid_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Print model information
    param_info = model.count_parameters()
    print("=" * 50)
    print("KKAN_SMALL MODEL INFORMATION")
    print("=" * 50)
    print(f"Total parameters: {param_info['total_parameters']:,}")
    print(f"Trainable parameters: {param_info['trainable_parameters']:,}")
    print(f"Model size: {param_info['total_parameters'] * 4 / 1024 / 1024:.2f} MB")
    print(f"Grid size: {args.grid_size}")
    print(f"Input shape: 28x28x1 (MNIST)")
    print(f"Output classes: 10")
    print("=" * 50)
    
    # Create dataloaders
    print("Creating MNIST dataloaders...")
    train_loader, test_loader = create_mnist_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath='checkpoints/kkansmall_mnist',
        filename='kkansmall_mnist-{epoch:02d}-{val_acc:.3f}',
        save_top_k=3,
        mode='max',
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Setup logger
    logger = TensorBoardLogger(
        'logs', 
        name=f'kkansmall_mnist_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu' if args.gpus > 0 else 'cpu',
        devices=args.gpus if args.gpus > 0 else 1,
        precision=args.precision,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=50,
        val_check_interval=0.25,  # Validate 4 times per epoch
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
    )
    
    # Train the model
    print("Starting training...")
    trainer.fit(model, train_loader, test_loader)
    
    # Evaluate on test set
    print("Evaluating final model...")
    results = trainer.test(model, test_loader)
    
    print("=" * 50)
    print("FINAL EVALUATION RESULTS")
    print("=" * 50)
    for key, value in results[0].items():
        print(f"{key}: {value:.4f}")
    print("=" * 50)
    
    # Save final model
    trainer.save_checkpoint(f"kkansmall_mnist_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ckpt")
    print("Training completed!")


if __name__ == "__main__":
    main() 
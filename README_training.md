# Training KAN and CNN Models on ImageNet

This project contains scripts for training KAN (Kolmogorov-Arnold Networks) and CNN models on ImageNet.

## Models

### KAN Model
- **Parameters**: ~378K (1.44 MB)
- **Architecture**: 2 convolutional layers (16->32 channels)
- **Features**: Uses KAN-specific layers with B-spline activations

### CNN Model  
- **Parameters**: ~39K (0.15 MB)
- **Architecture**: 2 convolutional layers (16->32 channels)
- **Features**: Standard CNN layers with BatchNorm

## Quick Training (for testing)

### Training KAN Model
```bash
python train_kan_small.py --data_dir /path/to/imagenet --epochs 5 --subset_size 500
```

### Training CNN Model
```bash
python train_cnn_small.py --data_dir /path/to/imagenet --epochs 5 --subset_size 500
```

## Full Training

### Training KAN Model
```bash
python train_kan_imagenet.py --data_dir /path/to/imagenet --epochs 100 --batch_size 32
```

### Training CNN Model
```bash
python train_cnn_imagenet.py --data_dir /path/to/imagenet --epochs 100 --batch_size 32
```

## Parameters

### General Parameters
- `--data_dir`: Path to ImageNet dataset
- `--batch_size`: Batch size (default 16 for small, 32 for full)
- `--epochs`: Number of epochs
- `--learning_rate`: Learning rate (default 1e-3)
- `--gpus`: Number of GPUs (default 1)
- `--precision`: Training precision 16 or 32 (default 32)

### KAN-specific Parameters
- `--grid_size`: Grid size for KAN layers (default 5)
- `--spline_order`: B-spline order (default 3)
- `--regularize_activation`: Activation regularization weight (default 1.0)
- `--regularize_entropy`: Entropy regularization weight (default 1.0)

### Architecture Parameters
- `--layers`: Layer configuration in format "out_channels,kernel_size,stride,padding;..."
- `--subset_size`: Subset size for quick training (only for small scripts)

## Usage Examples

### Quick Testing on Small Subset
```bash
# KAN model on 500 samples, 5 epochs
python train_kan_small.py --data_dir /path/to/imagenet --epochs 5 --subset_size 500 --gpus 0

# CNN model on 500 samples, 5 epochs  
python train_cnn_small.py --data_dir /path/to/imagenet --epochs 5 --subset_size 500 --gpus 0
```

### Training with Custom Architecture
```bash
# KAN model with custom layers
python train_kan_small.py --data_dir /path/to/imagenet --layers "8,3,1,1;16,3,2,1" --epochs 10

# CNN model with custom layers
python train_cnn_small.py --data_dir /path/to/imagenet --layers "8,3,1,1;16,3,2,1" --epochs 10
```

### Full Training on GPU
```bash
# KAN model on full dataset
python train_kan_imagenet.py --data_dir /path/to/imagenet --epochs 100 --batch_size 32 --gpus 1

# CNN model on full dataset
python train_cnn_imagenet.py --data_dir /path/to/imagenet --epochs 100 --batch_size 32 --gpus 1
```

## Monitoring

All experiments are logged to TensorBoard:
```bash
tensorboard --logdir logs
```

## Results

Models are saved to:
- `checkpoints/kan_small/` - KAN model checkpoints
- `checkpoints/cnn_small/` - CNN model checkpoints
- `logs/` - TensorBoard logs

## Model Comparison

After training, use the `compare_models.py` script to compare results:
```bash
python compare_models.py --kan_checkpoint path/to/kan.ckpt --cnn_checkpoint path/to/cnn.ckpt --data_dir /path/to/imagenet
```

## Requirements

- PyTorch >= 2.0.0
- PyTorch Lightning >= 2.0.0
- TorchMetrics >= 1.0.0
- ImageNet dataset

## ImageNet Structure

Make sure your ImageNet dataset has the following structure:
```
/path/to/imagenet/
├── train/
│   ├── n01440764/
│   ├── n01443537/
│   └── ...
└── val/
    ├── n01440764/
    ├── n01443537/
    └── ...
``` 
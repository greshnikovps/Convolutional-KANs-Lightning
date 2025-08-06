# Convolutional-KANs-Lightning

PyTorch-Lightning wrap-up of convolutional KAN (Kolmogorov-Arnold Networks) for image classification tasks.

## Overview

This repository contains implementations of Convolutional KAN models with PyTorch Lightning training scripts for easy experimentation and training on various datasets including ImageNet and MNIST.

## Features

- **KAN Convolutional Layers**: Implementation of KAN-based convolutional layers with B-spline activations
- **PyTorch Lightning Integration**: All models wrapped in PyTorch Lightning for easy training
- **Multiple Architectures**: 
  - KKAN (KAN Convolutions + KAN Linear layers)
  - KANC MLP (KAN Convolutions + MLP)
  - CNN (Standard Convolutions + MLP) for comparison
- **Flexible Training Scripts**: 
  - Full ImageNet training
  - Small subset training for quick experiments
  - MNIST training for KKAN_Small model
- **Model Comparison Tools**: Scripts to compare KAN vs CNN performance
- **Parameter Counting**: Tools to analyze model complexity

## Quick Start

### Installation

```bash
git clone https://github.com/greshnikovps/Convolutional-KANs-Lightning.git
cd Convolutional-KANs-Lightning
pip install -r requirements_imagenet.txt
```

### Training KKAN_Small on MNIST

```bash
python train_kkansmall_mnist.py --epochs 10 --batch_size 64 --gpus 0
```

### Training KAN Models on ImageNet (Small Subset)

```bash
# Create test dataset
python create_test_dataset.py

# Train KAN model
python train_kan_small.py --data_dir test_imagenet --epochs 5 --subset_size 100 --num_classes 10 --gpus 0

# Train CNN model for comparison
python train_cnn_small.py --data_dir test_imagenet --epochs 5 --subset_size 100 --num_classes 10 --gpus 0
```

### Model Comparison

```bash
python compare_models.py --kan_checkpoint path/to/kan_model.ckpt --cnn_checkpoint path/to/cnn_model.ckpt
```

## Model Architectures

### KKAN_Small (MNIST)
- **Parameters**: ~15K
- **Input**: 28x28x1 (MNIST)
- **Output**: 10 classes
- **Architecture**: 2 KAN convolutional layers + KAN linear layer

### KANConvNet (ImageNet)
- **Parameters**: ~378K (configurable)
- **Input**: 224x224x3 (ImageNet)
- **Output**: Configurable (default: 10 classes)
- **Architecture**: Configurable KAN convolutional layers + KAN linear classifier

### CNNConvNet (ImageNet)
- **Parameters**: ~378K (configurable)
- **Input**: 224x224x3 (ImageNet)
- **Output**: Configurable (default: 10 classes)
- **Architecture**: Configurable CNN layers + linear classifier

## Key Files

### Training Scripts
- `train_kkansmall_mnist.py` - Train KKAN_Small on MNIST
- `train_kan_small.py` - Train KAN model on small ImageNet subset
- `train_cnn_small.py` - Train CNN model on small ImageNet subset
- `train_kan_imagenet.py` - Train KAN model on full ImageNet
- `train_cnn_imagenet.py` - Train CNN model on full ImageNet

### Model Definitions
- `kan_convolutional/KANLightningModel.py` - PyTorch Lightning wrapper for KAN models
- `architectures_28x28/KKAN.py` - KKAN_Small model definition
- `kan_convolutional/KANConv.py` - KAN convolutional layer implementation
- `kan_convolutional/KANLinear.py` - KAN linear layer implementation

### Utility Scripts
- `compare_models.py` - Compare trained models
- `test_models.py` - Test model functionality
- `create_test_dataset.py` - Create synthetic test dataset
- `requirements_imagenet.txt` - Dependencies for ImageNet training

## Results

### MNIST Results (KKAN_Small)
- **Parameters**: 15,200
- **Training Time**: ~50 minutes (5 epochs on CPU)
- **Final Accuracy**: ~97% (varies by run)

### ImageNet Results (Small Subset)
- **KAN Model**: ~378K parameters
- **CNN Model**: ~378K parameters
- **Training Time**: ~20-30 minutes (5 epochs on CPU)

## Configuration

### Layer Configuration
You can configure convolutional layers using the `--layers` parameter:

```bash
# Example: 2 layers with custom configuration
python train_kan_small.py --layers "16,5,2,2;32,3,2,1"
```

Format: `"out_channels,kernel_size,stride,padding;..."`

### KAN Parameters
- `--grid_size`: Grid size for B-splines (default: 5)
- `--spline_order`: Spline order (default: 3)
- `--scale_noise`: Scale noise for initialization (default: 0.1)
- `--regularize_activation`: Activation regularization weight (default: 1.0)
- `--regularize_entropy`: Entropy regularization weight (default: 1.0)

## Monitoring Training

### TensorBoard
Training logs are saved to `logs/` directory. View with:
```bash
tensorboard --logdir logs
```

### Checkpoints
Models are automatically saved to `checkpoints/` directory with best validation accuracy.

## Contributing

We welcome contributions! Please feel free to submit pull requests or open issues for:
- Bug fixes
- New model architectures
- Performance improvements
- Documentation updates

## License

MIT License - see LICENSE file for details.

## Acknowledgments

This work builds upon the original Convolutional-KANs implementation by AntonioTepsich, adding PyTorch Lightning integration for easier training and experimentation.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{convolutional-kans-lightning,
  title={Convolutional-KANs-Lightning: PyTorch Lightning Implementation of Convolutional Kolmogorov-Arnold Networks},
  author={greshnikovps},
  year={2025},
  url={https://github.com/greshnikovps/Convolutional-KANs-Lightning}
}
```

# KAN Convolutional Models on ImageNet

This project contains implementation of convolutional KAN (Kolmogorov-Arnold Networks) models for image classification on the ImageNet dataset, as well as comparison with standard CNN models.

## Project Structure

- `kan_convolutional/KANLightningModel.py` - Lightning module for KAN convolutional model
- `train_kan_imagenet.py` - Script for training KAN model on ImageNet
- `train_cnn_imagenet.py` - Script for training standard CNN model on ImageNet
- `compare_models.py` - Script for comparing training results
- `requirements_imagenet.txt` - Project dependencies

## Installation

1. Install dependencies:
```bash
pip install -r requirements_imagenet.txt
```

2. Prepare ImageNet dataset:
   - Download ImageNet dataset
   - Extract to folder with the following structure:
```
imagenet/
├── train/
│   ├── n01440764/
│   ├── n01443537/
│   └── ...
└── val/
    ├── n01440764/
    ├── n01443537/
    └── ...
```

## Training KAN Model

```bash
python train_kan_imagenet.py \
    --data_dir /path/to/imagenet \
    --batch_size 32 \
    --epochs 100 \
    --learning_rate 1e-3 \
    --grid_size 5 \
    --spline_order 3 \
    --gpus 1 \
    --precision 16
```

### KAN Model Parameters:
- `--grid_size`: Grid size for KAN (default: 5)
- `--spline_order`: Spline order (default: 3)
- `--scale_noise`: Noise scale (default: 0.1)
- `--scale_base`: Base function scale (default: 1.0)
- `--scale_spline`: Spline scale (default: 1.0)
- `--regularize_activation`: Activation regularization (default: 1.0)
- `--regularize_entropy`: Entropy regularization (default: 1.0)

## Training CNN Model

```bash
python train_cnn_imagenet.py \
    --data_dir /path/to/imagenet \
    --batch_size 32 \
    --epochs 100 \
    --learning_rate 1e-3 \
    --gpus 1 \
    --precision 16
```

## Model Comparison

After training both models, use the comparison script:

```bash
python compare_models.py \
    --kan_checkpoint path/to/kan_model.ckpt \
    --cnn_checkpoint path/to/cnn_model.ckpt \
    --data_dir /path/to/imagenet \
    --create_plots
```

## Model Architectures

### KAN Model:
- 4 convolutional layers with KAN kernels
- Channel sizes: 3 → 64 → 128 → 256 → 512
- Kernel sizes: 7x7, 3x3, 3x3, 3x3
- Global Average Pooling
- KAN classifier

### CNN Model:
- 4 convolutional layers with standard kernels
- Batch Normalization after each layer
- Channel sizes: 3 → 64 → 128 → 256 → 512
- Kernel sizes: 7x7, 3x3, 3x3, 3x3
- Global Average Pooling
- Linear classifier

## Training Monitoring

Use TensorBoard for monitoring:

```bash
tensorboard --logdir logs
```

## Results

The comparison script outputs:
- Number of parameters for each model
- Model size in MB
- Loss, Top-1 and Top-5 accuracy
- Differences between models
- Saves results to JSON file
- Creates comparison plots (optional)

## KAN Model Features

1. **Non-linear kernels**: KAN kernels can learn complex non-linear functions
2. **Regularization**: Built-in regularization through splines
3. **Interpretability**: Ability to analyze learned functions
4. **Adaptability**: Automatic grid adaptation to data

## System Requirements

- CUDA-compatible GPU (recommended)
- Minimum 16GB RAM
- Sufficient disk space for ImageNet (~150GB)
- Python 3.8+

## Notes

- Full training on ImageNet will require significant time
- Recommended to use mixed precision (precision=16)
- Models are automatically saved to `checkpoints/` folder
- Logs are saved to `logs/` folder for TensorBoard 
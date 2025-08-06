import torch
import torch.nn as nn
import sys
import os

# Add the kan_convolutional directory to the path
sys.path.append('./kan_convolutional')
from KANLightningModel import KANConvNet
from train_cnn_imagenet import CNNConvNet


def test_model_creation():
    """Test that models can be created successfully"""
    print("Testing model creation...")
    
    # Test KAN model with default (smaller) configuration
    kan_model = KANConvNet(
        num_classes=1000,
        in_channels=3,
        grid_size=5,
        spline_order=3
    )
    
    # Test CNN model with default (smaller) configuration
    cnn_model = CNNConvNet(
        num_classes=1000,
        in_channels=3
    )
    
    print("✓ Models created successfully")
    return kan_model, cnn_model


def test_forward_pass(kan_model, cnn_model):
    """Test that models can perform forward pass"""
    print("Testing forward pass...")
    
    # Create dummy input
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    
    # Test KAN model
    kan_output = kan_model(input_tensor)
    assert kan_output.shape == (batch_size, 1000), f"KAN output shape: {kan_output.shape}"
    
    # Test CNN model
    cnn_output = cnn_model(input_tensor)
    assert cnn_output.shape == (batch_size, 1000), f"CNN output shape: {cnn_output.shape}"
    
    print("✓ Forward pass successful")


def test_loss_computation(kan_model, cnn_model):
    """Test that models can compute loss"""
    print("Testing loss computation...")
    
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    labels = torch.randint(0, 1000, (batch_size,))
    
    # Test KAN model
    kan_output = kan_model(input_tensor)
    kan_loss = kan_model.criterion(kan_output, labels)
    assert isinstance(kan_loss, torch.Tensor), "KAN loss should be a tensor"
    
    # Test CNN model
    cnn_output = cnn_model(input_tensor)
    cnn_loss = cnn_model.criterion(cnn_output, labels)
    assert isinstance(cnn_loss, torch.Tensor), "CNN loss should be a tensor"
    
    print("✓ Loss computation successful")


def test_regularization_loss(kan_model):
    """Test that KAN model can compute regularization loss"""
    print("Testing KAN regularization loss...")
    
    reg_loss = kan_model._get_regularization_loss()
    assert isinstance(reg_loss, torch.Tensor), "Regularization loss should be a tensor"
    assert reg_loss >= 0, "Regularization loss should be non-negative"
    
    print("✓ Regularization loss computation successful")


def test_with_small_dataset(kan_model, cnn_model):
    """Test models with a small dataset"""
    print("Testing with small dataset...")
    
    # Create small dummy dataset
    batch_size = 2
    num_samples = 10
    
    # Create dummy data
    inputs = torch.randn(num_samples, 3, 224, 224)
    labels = torch.randint(0, 1000, (num_samples,))
    
    # Test training step for both models
    for i in range(0, num_samples, batch_size):
        batch_inputs = inputs[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        
        # Test KAN training step
        kan_loss = kan_model.training_step((batch_inputs, batch_labels), 0)
        assert isinstance(kan_loss, torch.Tensor), "KAN training step should return loss"
        
        # Test CNN training step
        cnn_loss = cnn_model.training_step((batch_inputs, batch_labels), 0)
        assert isinstance(cnn_loss, torch.Tensor), "CNN training step should return loss"
    
    print("✓ Small dataset test successful")


def compare_parameter_counts(kan_model, cnn_model):
    """Compare parameter counts between models"""
    print("Comparing parameter counts...")
    
    kan_params = kan_model.count_parameters()
    cnn_params = cnn_model.count_parameters()
    
    print("=" * 50)
    print("MODEL PARAMETER COMPARISON")
    print("=" * 50)
    print("KAN Model:")
    print(f"  Total parameters: {kan_params['total_parameters']:,}")
    print(f"  Trainable parameters: {kan_params['trainable_parameters']:,}")
    print(f"  Model size: {kan_params['total_parameters'] * 4 / 1024 / 1024:.2f} MB")
    print()
    print("CNN Model:")
    print(f"  Total parameters: {cnn_params['total_parameters']:,}")
    print(f"  Trainable parameters: {cnn_params['trainable_parameters']:,}")
    print(f"  Model size: {cnn_params['total_parameters'] * 4 / 1024 / 1024:.2f} MB")
    print()
    print("Difference:")
    diff_params = kan_params['total_parameters'] - cnn_params['total_parameters']
    diff_size = (kan_params['total_parameters'] - cnn_params['total_parameters']) * 4 / 1024 / 1024
    print(f"  Parameters: {diff_params:,}")
    print(f"  Size: {diff_size:.2f} MB")
    print("=" * 50)
    
    # Check if KAN model has less than 1M parameters
    if kan_params['total_parameters'] < 1_000_000:
        print("✓ KAN model has less than 1M parameters")
    else:
        print(f"⚠ KAN model has {kan_params['total_parameters']:,} parameters (more than 1M)")


def test_custom_architectures():
    """Test models with custom layer configurations"""
    print("Testing custom architectures...")
    
    # Test KAN model with custom layers
    custom_layers = [
        (16, (5, 5), (2, 2), (2, 2)),   # 3 -> 16
        (32, (3, 3), (2, 2), (1, 1)),    # 16 -> 32
    ]
    
    kan_custom = KANConvNet(
        num_classes=1000,
        in_channels=3,
        conv_layers=custom_layers,
        grid_size=5,
        spline_order=3
    )
    
    # Test CNN model with custom layers
    cnn_custom = CNNConvNet(
        num_classes=1000,
        in_channels=3,
        conv_layers=custom_layers
    )
    
    # Test forward pass
    input_tensor = torch.randn(2, 3, 224, 224)
    kan_output = kan_custom(input_tensor)
    cnn_output = cnn_custom(input_tensor)
    
    assert kan_output.shape == (2, 1000), f"Custom KAN output shape: {kan_output.shape}"
    assert cnn_output.shape == (2, 1000), f"Custom CNN output shape: {cnn_output.shape}"
    
    print("✓ Custom architectures work correctly")


def main():
    print("Starting model tests...")
    
    # Test 1: Model creation
    kan_model, cnn_model = test_model_creation()
    
    # Test 2: Parameter comparison
    compare_parameter_counts(kan_model, cnn_model)
    
    # Test 3: Forward pass
    test_forward_pass(kan_model, cnn_model)
    
    # Test 4: Loss computation
    test_loss_computation(kan_model, cnn_model)
    
    # Test 5: KAN regularization
    test_regularization_loss(kan_model)
    
    # Test 6: Small dataset
    test_with_small_dataset(kan_model, cnn_model)
    
    # Test 7: Custom architectures
    test_custom_architectures()
    
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED! ✓")
    print("=" * 50)
    print("\nModels are ready for training on ImageNet!")


if __name__ == "__main__":
    main() 
import os
import shutil
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

def create_test_imagenet_structure(base_dir="test_imagenet", num_classes=10, samples_per_class=50):
    """
    Create a small test ImageNet-like dataset structure
    """
    # Create directory structure
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Create class directories
    for i in range(num_classes):
        class_name = f"class_{i:03d}"
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
    
    print(f"Created test dataset structure:")
    print(f"  Base directory: {base_dir}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Samples per class: {samples_per_class}")
    print(f"  Total samples: {num_classes * samples_per_class}")
    
    # Generate synthetic images
    print("Generating synthetic images...")
    
    for class_idx in range(num_classes):
        class_name = f"class_{class_idx:03d}"
        
        # Create training samples
        for sample_idx in range(samples_per_class):
            # Create a random image with some class-specific patterns
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
            # Add some class-specific patterns
            if class_idx % 3 == 0:
                # Add horizontal lines
                for i in range(0, 224, 20):
                    img_array[i:i+5, :, :] = [255, 0, 0]
            elif class_idx % 3 == 1:
                # Add vertical lines
                for i in range(0, 224, 20):
                    img_array[:, i:i+5, :] = [0, 255, 0]
            else:
                # Add diagonal lines
                for i in range(0, 224, 15):
                    img_array[i, i:i+10, :] = [0, 0, 255]
            
            # Convert to PIL Image
            img = Image.fromarray(img_array)
            
            # Save training image
            train_path = os.path.join(train_dir, class_name, f"train_{sample_idx:03d}.jpg")
            img.save(train_path)
            
            # Save validation image (every 5th sample)
            if sample_idx % 5 == 0:
                val_path = os.path.join(val_dir, class_name, f"val_{sample_idx:03d}.jpg")
                img.save(val_path)
    
    print("Test dataset created successfully!")
    print(f"Training samples: {num_classes * samples_per_class}")
    print(f"Validation samples: {num_classes * (samples_per_class // 5)}")
    
    return base_dir

def test_dataloader(data_dir):
    """
    Test that the dataloader works with the test dataset
    """
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    
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
        batch_size=8,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    print("Testing dataloaders...")
    
    # Test training loader
    for i, (images, labels) in enumerate(train_loader):
        print(f"Training batch {i+1}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Labels: {labels}")
        break
    
    # Test validation loader
    for i, (images, labels) in enumerate(val_loader):
        print(f"Validation batch {i+1}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Labels: {labels}")
        break
    
    print("Dataloader test completed successfully!")

if __name__ == "__main__":
    # Create test dataset
    test_dir = create_test_imagenet_structure("test_imagenet", num_classes=10, samples_per_class=50)
    
    # Test dataloader
    test_dataloader(test_dir)
    
    print("\nTest dataset is ready for training!")
    print(f"Use --data_dir {test_dir} in training scripts") 
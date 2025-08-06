import torch
import pytorch_lightning as pl
import argparse
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from kan_convolutional.KANLightningModel import KANConvNet
from train_cnn_imagenet import CNNConvNet


def load_model_from_checkpoint(checkpoint_path, model_class):
    """
    Load model from checkpoint
    """
    model = model_class.load_from_checkpoint(checkpoint_path)
    return model


def evaluate_model(model, dataloader, device='cuda'):
    """
    Evaluate model on dataloader
    """
    model.to(device)
    model.eval()
    
    total_loss = 0
    correct = 0
    correct_top5 = 0
    total = 0
    
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            loss = criterion(output, target)
            total_loss += loss.item()
            
            # Top-1 accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            # Top-5 accuracy
            _, top5_pred = output.topk(5, 1, True, True)
            top5_pred = top5_pred.t()
            correct_top5 += top5_pred.eq(target.view(1, -1).expand_as(top5_pred)).sum().item()
            
            total += target.size(0)
    
    accuracy = 100. * correct / total
    accuracy_top5 = 100. * correct_top5 / total
    avg_loss = total_loss / len(dataloader)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'accuracy_top5': accuracy_top5
    }


def compare_models(kan_checkpoint, cnn_checkpoint, data_dir, batch_size=32, num_workers=4):
    """
    Compare KAN and CNN models
    """
    from train_kan_imagenet import create_imagenet_dataloaders
    
    # Create dataloaders
    _, val_loader = create_imagenet_dataloaders(
        data_dir, 
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Load models
    print("Loading KAN model...")
    kan_model = load_model_from_checkpoint(kan_checkpoint, KANConvNet)
    
    print("Loading CNN model...")
    cnn_model = load_model_from_checkpoint(cnn_checkpoint, CNNConvNet)
    
    # Get parameter counts
    kan_params = kan_model.count_parameters()
    cnn_params = cnn_model.count_parameters()
    
    # Evaluate models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Evaluating KAN model...")
    kan_results = evaluate_model(kan_model, val_loader, device)
    
    print("Evaluating CNN model...")
    cnn_results = evaluate_model(cnn_model, val_loader, device)
    
    # Print comparison
    print("=" * 80)
    print("MODEL COMPARISON RESULTS")
    print("=" * 80)
    
    print(f"{'Metric':<20} {'KAN Model':<15} {'CNN Model':<15} {'Difference':<15}")
    print("-" * 80)
    print(f"{'Parameters':<20} {kan_params['total_parameters']:<15,} {cnn_params['total_parameters']:<15,} {kan_params['total_parameters'] - cnn_params['total_parameters']:<15,}")
    print(f"{'Model Size (MB)':<20} {kan_params['total_parameters'] * 4 / 1024 / 1024:<15.2f} {cnn_params['total_parameters'] * 4 / 1024 / 1024:<15.2f} {(kan_params['total_parameters'] - cnn_params['total_parameters']) * 4 / 1024 / 1024:<15.2f}")
    print(f"{'Loss':<20} {kan_results['loss']:<15.4f} {cnn_results['loss']:<15.4f} {kan_results['loss'] - cnn_results['loss']:<15.4f}")
    print(f"{'Top-1 Accuracy':<20} {kan_results['accuracy']:<15.2f}% {cnn_results['accuracy']:<15.2f}% {kan_results['accuracy'] - cnn_results['accuracy']:<15.2f}%")
    print(f"{'Top-5 Accuracy':<20} {kan_results['accuracy_top5']:<15.2f}% {cnn_results['accuracy_top5']:<15.2f}% {kan_results['accuracy_top5'] - cnn_results['accuracy_top5']:<15.2f}%")
    print("=" * 80)
    
    # Save results to file
    results = {
        'kan_model': {
            'parameters': kan_params['total_parameters'],
            'model_size_mb': kan_params['total_parameters'] * 4 / 1024 / 1024,
            'loss': kan_results['loss'],
            'accuracy': kan_results['accuracy'],
            'accuracy_top5': kan_results['accuracy_top5']
        },
        'cnn_model': {
            'parameters': cnn_params['total_parameters'],
            'model_size_mb': cnn_params['total_parameters'] * 4 / 1024 / 1024,
            'loss': cnn_results['loss'],
            'accuracy': cnn_results['accuracy'],
            'accuracy_top5': cnn_results['accuracy_top5']
        },
        'comparison': {
            'parameter_difference': kan_params['total_parameters'] - cnn_params['total_parameters'],
            'size_difference_mb': (kan_params['total_parameters'] - cnn_params['total_parameters']) * 4 / 1024 / 1024,
            'loss_difference': kan_results['loss'] - cnn_results['loss'],
            'accuracy_difference': kan_results['accuracy'] - cnn_results['accuracy'],
            'accuracy_top5_difference': kan_results['accuracy_top5'] - cnn_results['accuracy_top5']
        }
    }
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'model_comparison_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to model_comparison_{timestamp}.json")
    
    return results


def create_comparison_plots(results):
    """
    Create comparison plots
    """
    metrics = ['accuracy', 'accuracy_top5', 'loss']
    kan_values = [results['kan_model'][m] for m in metrics]
    cnn_values = [results['cnn_model'][m] for m in metrics]
    
    # Create bar plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy comparison
    x = np.arange(len(metrics))
    width = 0.35
    
    ax1.bar(x - width/2, kan_values, width, label='KAN Model', alpha=0.8)
    ax1.bar(x + width/2, cnn_values, width, label='CNN Model', alpha=0.8)
    
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Value')
    ax1.set_title('Model Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Parameter comparison
    models = ['KAN', 'CNN']
    parameters = [results['kan_model']['parameters'], results['cnn_model']['parameters']]
    
    ax2.bar(models, parameters, alpha=0.8)
    ax2.set_ylabel('Number of Parameters')
    ax2.set_title('Model Size Comparison')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(parameters):
        ax2.text(i, v + max(parameters) * 0.01, f'{v:,}', ha='center', va='bottom')
    
    plt.tight_layout()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'model_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Comparison plot saved as model_comparison_{timestamp}.png")


def main():
    parser = argparse.ArgumentParser(description='Compare KAN and CNN models')
    parser.add_argument('--kan_checkpoint', type=str, required=True, help='Path to KAN model checkpoint')
    parser.add_argument('--cnn_checkpoint', type=str, required=True, help='Path to CNN model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to ImageNet dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for dataloader')
    parser.add_argument('--create_plots', action='store_true', help='Create comparison plots')
    
    args = parser.parse_args()
    
    # Compare models
    results = compare_models(
        args.kan_checkpoint,
        args.cnn_checkpoint,
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create plots if requested
    if args.create_plots:
        create_comparison_plots(results)


if __name__ == "__main__":
    main() 
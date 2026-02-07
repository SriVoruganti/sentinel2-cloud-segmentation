"""
Evaluation Script for Cloud Segmentation Model
Evaluates trained U-Net model on test set and generates metrics
"""

import os
import sys
import argparse
import json
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dataset import get_dataloaders
from model import get_model


class ModelEvaluator:
    """Evaluate cloud segmentation model."""
    
    def __init__(self, model_path, data_dir, output_dir, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Using device: {self.device}")
        print(f"Output directory: {self.output_dir}")
        
        # Load model
        print("\nLoading model...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        self.model = get_model(
            encoder_name='resnet34',
            encoder_weights=None,
            in_channels=6,
            classes=1
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from: {model_path}")
        print(f"Best validation Dice: {checkpoint.get('best_val_dice', 'N/A'):.4f}")
        
        # Load data
        print("\nLoading test data...")
        _, _, self.test_loader = get_dataloaders(
            data_dir=data_dir,
            batch_size=16,
            num_workers=4
        )
    
    def calculate_metrics(self, pred, target, threshold=0.5):
        """Calculate segmentation metrics."""
        pred_binary = (torch.sigmoid(pred) > threshold).float()
        target_binary = target
        
        # Flatten
        pred_flat = pred_binary.view(-1).cpu().numpy()
        target_flat = target_binary.view(-1).cpu().numpy()
        
        # Intersection and Union
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection
        
        # Metrics
        dice = (2. * intersection + 1e-7) / (pred_flat.sum() + target_flat.sum() + 1e-7)
        iou = (intersection + 1e-7) / (union + 1e-7)
        
        # Pixel accuracy
        correct = (pred_flat == target_flat).sum()
        accuracy = correct / len(target_flat)
        
        # Precision and Recall
        true_positives = intersection
        false_positives = pred_flat.sum() - intersection
        false_negatives = target_flat.sum() - intersection
        
        precision = true_positives / (true_positives + false_positives + 1e-7)
        recall = true_positives / (true_positives + false_negatives + 1e-7)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        
        return {
            'dice': dice,
            'iou': iou,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'pred_flat': pred_flat,
            'target_flat': target_flat
        }
    
    def evaluate(self):
        """Run evaluation on test set."""
        print("\nEvaluating on test set...")
        
        all_metrics = []
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for images, masks in tqdm(self.test_loader, desc="Testing"):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate metrics
                metrics = self.calculate_metrics(outputs, masks)
                all_metrics.append(metrics)
                
                all_preds.extend(metrics['pred_flat'])
                all_targets.extend(metrics['target_flat'])
        
        # Aggregate metrics
        results = {
            'dice': float(np.mean([m['dice'] for m in all_metrics])),
            'dice_std': float(np.std([m['dice'] for m in all_metrics])),
            'iou': float(np.mean([m['iou'] for m in all_metrics])),
            'iou_std': float(np.std([m['iou'] for m in all_metrics])),
            'accuracy': float(np.mean([m['accuracy'] for m in all_metrics])),
            'precision': float(np.mean([m['precision'] for m in all_metrics])),
            'recall': float(np.mean([m['recall'] for m in all_metrics])),
            'f1': float(np.mean([m['f1'] for m in all_metrics])),
            'num_samples': int(len(self.test_loader.dataset))
        }
        
        # Print results
        print(f"\n{'='*60}")
        print(f"TEST SET RESULTS")
        print(f"{'='*60}")
        print(f"Number of test samples: {results['num_samples']}")
        print(f"\nDice Coefficient: {results['dice']:.4f} ± {results['dice_std']:.4f}")
        print(f"IoU (Jaccard):    {results['iou']:.4f} ± {results['iou_std']:.4f}")
        print(f"Accuracy:         {results['accuracy']:.4f}")
        print(f"Precision:        {results['precision']:.4f}")
        print(f"Recall:           {results['recall']:.4f}")
        print(f"F1-Score:         {results['f1']:.4f}")
        print(f"{'='*60}\n")
        
        # Save results
        results_file = self.output_dir / 'test_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {results_file}")
        
        # Generate confusion matrix
        self.plot_confusion_matrix(
            np.array(all_targets),
            np.array(all_preds)
        )
        
        # Generate metrics distribution
        self.plot_metrics_distribution(all_metrics)
        
        return results
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Clear', 'Cloud'],
            yticklabels=['Clear', 'Cloud']
        )
        plt.title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        output_file = self.output_dir / 'confusion_matrix.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to: {output_file}")
    
    def plot_metrics_distribution(self, all_metrics):
        """Plot distribution of metrics across batches."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        metrics_to_plot = ['dice', 'iou', 'precision', 'recall']
        titles = ['Dice Coefficient', 'IoU (Jaccard)', 'Precision', 'Recall']
        
        for ax, metric, title in zip(axes.flat, metrics_to_plot, titles):
            values = [m[metric] for m in all_metrics]
            
            ax.hist(values, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
            ax.axvline(np.mean(values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(values):.3f}')
            ax.set_xlabel(title, fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title(f'{title} Distribution', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_file = self.output_dir / 'metrics_distribution.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Metrics distribution saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate cloud segmentation model')
    parser.add_argument('--model_path', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', required=True, help='Path to dataset directory')
    parser.add_argument('--output_dir', required=True, help='Path to save evaluation results')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if available')
    
    args = parser.parse_args()
    
    device = 'cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu'
    
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=device
    )
    
    evaluator.evaluate()
    
    print("\n✓ Evaluation complete!")


if __name__ == '__main__':
    main()

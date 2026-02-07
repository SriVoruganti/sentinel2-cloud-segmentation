"""
Visualization Script for Cloud Segmentation
Generates visual examples of model predictions
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dataset import get_dataloaders
from model import get_model


class PredictionVisualizer:
    """Visualize model predictions."""
    
    def __init__(self, model_path, data_dir, output_dir, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Using device: {self.device}")
        
        # Load model
        print("Loading model...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        self.model = get_model(
            encoder_name='resnet34',
            encoder_weights=None,
            in_channels=6,
            classes=1
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load data
        print("Loading data...")
        _, _, self.test_loader = get_dataloaders(
            data_dir=data_dir,
            batch_size=1,  # One at a time for visualization
            num_workers=0
        )
    
    def denormalize_image(self, image):
        """Denormalize image for visualization."""
        # Image is (C, H, W), values in [0, 1] range
        # Use RGB bands (B04, B03, B02) for true color
        rgb = image[[2, 1, 0], :, :]  # Reverse order: R, G, B
        
        # Normalize to [0, 1] for display
        rgb = rgb.cpu().numpy()
        rgb = np.transpose(rgb, (1, 2, 0))
        
        # Enhance contrast
        rgb = np.clip(rgb * 2.5, 0, 1)
        
        return rgb
    
    def visualize_prediction(self, image, mask, prediction, save_path, title=""):
        """Create visualization of input, ground truth, and prediction."""
        # Prepare data
        rgb = self.denormalize_image(image)
        mask_np = mask.squeeze().cpu().numpy()
        pred_np = (torch.sigmoid(prediction) > 0.5).squeeze().cpu().numpy()
        
        # Calculate metrics
        intersection = (pred_np * mask_np).sum()
        union = pred_np.sum() + mask_np.sum() - intersection
        dice = (2. * intersection + 1e-7) / (pred_np.sum() + mask_np.sum() + 1e-7)
        iou = (intersection + 1e-7) / (union + 1e-7)
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Row 1: RGB, Ground Truth, Prediction
        axes[0, 0].imshow(rgb)
        axes[0, 0].set_title('RGB Image (Bands 4,3,2)', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(mask_np, cmap='gray')
        axes[0, 1].set_title('Ground Truth Cloud Mask', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(pred_np, cmap='gray')
        axes[0, 2].set_title(f'Predicted Cloud Mask\nDice: {dice:.3f}, IoU: {iou:.3f}', 
                            fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Row 2: Overlays
        # RGB + Ground Truth overlay
        axes[1, 0].imshow(rgb)
        mask_overlay = np.ma.masked_where(mask_np == 0, mask_np)
        axes[1, 0].imshow(mask_overlay, cmap='Reds', alpha=0.5)
        axes[1, 0].set_title('RGB + Ground Truth Overlay', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        # RGB + Prediction overlay
        axes[1, 1].imshow(rgb)
        pred_overlay = np.ma.masked_where(pred_np == 0, pred_np)
        axes[1, 1].imshow(pred_overlay, cmap='Blues', alpha=0.5)
        axes[1, 1].set_title('RGB + Prediction Overlay', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        # Comparison: TP, FP, FN
        comparison = np.zeros((*mask_np.shape, 3))
        # True Positives (Green)
        comparison[np.logical_and(pred_np == 1, mask_np == 1)] = [0, 1, 0]
        # False Positives (Red)
        comparison[np.logical_and(pred_np == 1, mask_np == 0)] = [1, 0, 0]
        # False Negatives (Blue)
        comparison[np.logical_and(pred_np == 0, mask_np == 1)] = [0, 0, 1]
        
        axes[1, 2].imshow(comparison)
        axes[1, 2].set_title('Comparison\nGreen: TP, Red: FP, Blue: FN', 
                            fontsize=12, fontweight='bold')
        axes[1, 2].axis('off')
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_examples(self, num_examples=10, random_seed=42):
        """Generate visualization examples."""
        print(f"\nGenerating {num_examples} prediction examples...")
        
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Get random indices
        dataset_size = len(self.test_loader.dataset)
        indices = random.sample(range(dataset_size), min(num_examples, dataset_size))
        
        with torch.no_grad():
            for idx, (images, masks) in enumerate(self.test_loader):
                if idx not in indices:
                    continue
                
                if len([f for f in self.output_dir.glob('*.png')]) >= num_examples:
                    break
                
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Save visualization
                save_path = self.output_dir / f'prediction_{idx:03d}.png'
                self.visualize_prediction(
                    images[0],
                    masks[0],
                    outputs[0],
                    save_path,
                    title=f'Test Sample {idx}'
                )
                
                print(f"  Saved: {save_path.name}")
        
        print(f"\n✓ Generated {num_examples} visualizations")
        print(f"  Output directory: {self.output_dir}")
    
    def create_summary_grid(self, num_examples=9):
        """Create a summary grid of predictions."""
        print(f"\nCreating summary grid with {num_examples} examples...")
        
        examples = []
        
        with torch.no_grad():
            for idx, (images, masks) in enumerate(self.test_loader):
                if len(examples) >= num_examples:
                    break
                
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                
                examples.append({
                    'image': images[0],
                    'mask': masks[0],
                    'prediction': outputs[0]
                })
        
        # Create grid
        rows = int(np.sqrt(num_examples))
        cols = int(np.ceil(num_examples / rows))
        
        fig, axes = plt.subplots(rows, cols * 3, figsize=(cols * 5, rows * 5))
        
        for idx, example in enumerate(examples):
            row = idx // cols
            col = idx % cols
            
            rgb = self.denormalize_image(example['image'])
            mask_np = example['mask'].squeeze().cpu().numpy()
            pred_np = (torch.sigmoid(example['prediction']) > 0.5).squeeze().cpu().numpy()
            
            # RGB
            ax = axes[row, col * 3] if rows > 1 else axes[col * 3]
            ax.imshow(rgb)
            ax.set_title(f'Sample {idx}', fontsize=10)
            ax.axis('off')
            
            # Ground Truth
            ax = axes[row, col * 3 + 1] if rows > 1 else axes[col * 3 + 1]
            ax.imshow(mask_np, cmap='gray')
            ax.set_title('GT', fontsize=10)
            ax.axis('off')
            
            # Prediction
            ax = axes[row, col * 3 + 2] if rows > 1 else axes[col * 3 + 2]
            ax.imshow(pred_np, cmap='gray')
            ax.set_title('Pred', fontsize=10)
            ax.axis('off')
        
        plt.suptitle('Cloud Segmentation Results Summary', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / 'summary_grid.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Summary grid saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize cloud segmentation predictions')
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--data_dir', required=True, help='Path to dataset')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--num_examples', type=int, default=10, help='Number of examples')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    device = 'cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu'
    
    visualizer = PredictionVisualizer(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=device
    )
    
    # Generate individual examples
    visualizer.generate_examples(
        num_examples=args.num_examples,
        random_seed=args.seed
    )
    
    # Create summary grid
    visualizer.create_summary_grid(num_examples=9)
    
    print("\n✓ Visualization complete!")


if __name__ == '__main__':
    main()

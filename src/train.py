"""
Training Script for Cloud Segmentation U-Net Model

Usage:
    python train.py --data_dir /path/to/dataset --output_dir /path/to/models
"""

import os
import sys
import argparse
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from dataset import get_dataloaders
from model import get_model, get_loss_function


class Trainer:
    """Trainer class for cloud segmentation model."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config['use_gpu'] else 'cpu')
        
        print(f"Using device: {self.device}")
        
        # Create output directory
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup model
        print("\nInitializing model...")
        self.model = get_model(
            encoder_name=config['encoder'],
            encoder_weights=config['encoder_weights'],
            in_channels=config['in_channels'],
            classes=1
        ).to(self.device)
        
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {n_params:,}")
        
        # Setup loss, optimizer, scheduler
        self.criterion = get_loss_function(config['loss_type'])
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        if config['scheduler'] == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config['epochs'],
                eta_min=1e-6
            )
        elif config['scheduler'] == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            self.scheduler = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_dice = 0.0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_dice': [],
            'val_iou': [],
            'lr': []
        }
    
    def calculate_metrics(self, pred, target, threshold=0.5):
        """Calculate segmentation metrics."""
        pred = torch.sigmoid(pred)
        pred_binary = (pred > threshold).float()
        target_binary = target
        
        # Flatten
        pred_flat = pred_binary.view(-1)
        target_flat = target_binary.view(-1)
        
        # Intersection and Union
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection
        
        # Dice coefficient
        dice = (2. * intersection + 1e-7) / (pred_flat.sum() + target_flat.sum() + 1e-7)
        
        # IoU
        iou = (intersection + 1e-7) / (union + 1e-7)
        
        # Accuracy
        correct = (pred_flat == target_flat).sum()
        accuracy = correct / target_flat.numel()
        
        return {
            'dice': dice.item(),
            'iou': iou.item(),
            'accuracy': accuracy.item()
        }
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader):
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        all_metrics = []
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validating"):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                total_loss += loss.item()
                
                # Calculate metrics
                metrics = self.calculate_metrics(outputs, masks)
                all_metrics.append(metrics)
        
        avg_loss = total_loss / len(val_loader)
        
        # Average metrics
        avg_metrics = {
            'dice': np.mean([m['dice'] for m in all_metrics]),
            'iou': np.mean([m['iou'] for m in all_metrics]),
            'accuracy': np.mean([m['accuracy'] for m in all_metrics])
        }
        
        return avg_loss, avg_metrics
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_dice': self.best_val_dice,
            'history': self.history,
            'config': self.config
        }
        
        # Save latest
        latest_path = self.output_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            best_path = self.output_dir / 'checkpoint_best.pth'
            torch.save(checkpoint, best_path)
            print(f"  Saved best model (Dice: {self.best_val_dice:.4f})")
    
    def train(self, train_loader, val_loader):
        """Main training loop."""
        print(f"\nStarting training...")
        print(f"Epochs: {self.config['epochs']}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Learning rate: {self.config['learning_rate']}")
        print(f"{'='*60}\n")
        
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_metrics = self.validate(val_loader)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['dice'])
                else:
                    self.scheduler.step()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_dice'].append(val_metrics['dice'])
            self.history['val_iou'].append(val_metrics['iou'])
            self.history['lr'].append(current_lr)
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{self.config['epochs']}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Dice: {val_metrics['dice']:.4f}")
            print(f"  Val IoU: {val_metrics['iou']:.4f}")
            print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"  LR: {current_lr:.6f}")
            
            # Check if best model
            is_best = val_metrics['dice'] > self.best_val_dice
            if is_best:
                self.best_val_dice = val_metrics['dice']
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(is_best=is_best)
            
            # Early stopping
            if self.config['early_stopping'] and patience_counter >= self.config['early_stopping_patience']:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
        
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Best validation Dice: {self.best_val_dice:.4f}")
        print(f"{'='*60}\n")
        
        # Save training history
        history_file = self.output_dir / 'training_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"Training history saved to: {history_file}")


def main():
    parser = argparse.ArgumentParser(description='Train cloud segmentation model')
    
    # Data
    parser.add_argument('--data_dir', required=True, help='Path to dataset directory')
    parser.add_argument('--output_dir', required=True, help='Path to save models and logs')
    
    # Model
    parser.add_argument('--encoder', default='resnet34', help='Encoder architecture')
    parser.add_argument('--encoder_weights', default='imagenet', help='Pretrained weights')
    parser.add_argument('--in_channels', type=int, default=6, help='Number of input channels')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    # Loss and optimization
    parser.add_argument('--loss_type', default='combined', choices=['dice', 'bce', 'combined'])
    parser.add_argument('--scheduler', default='cosine', choices=['cosine', 'plateau', 'none'])
    
    # Early stopping
    parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping')
    parser.add_argument('--early_stopping_patience', type=int, default=15, help='Early stopping patience')
    
    # Hardware
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Convert args to config dict
    config = vars(args)
    
    # Print configuration
    print("="*60)
    print("CLOUD SEGMENTATION MODEL TRAINING")
    print("="*60)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*60)
    
    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create trainer and train
    trainer = Trainer(config)
    trainer.train(train_loader, val_loader)
    
    print("\n✓ Training complete!")


if __name__ == '__main__':
    main()

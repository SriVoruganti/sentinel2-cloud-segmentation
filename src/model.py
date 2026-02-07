"""
U-Net Model for Cloud Segmentation
Uses segmentation_models_pytorch for pretrained encoders
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


def get_model(encoder_name='resnet34', encoder_weights='imagenet', in_channels=6, classes=1):
    """
    Create U-Net model with pretrained encoder.
    
    Args:
        encoder_name: Backbone encoder (resnet34, resnet50, efficientnet-b0, etc.)
        encoder_weights: Pretrained weights ('imagenet' or None)
        in_channels: Number of input channels (6 for Sentinel-2)
        classes: Number of output classes (1 for binary segmentation)
    
    Returns:
        model: U-Net model
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None  # We'll use sigmoid in training
    )
    
    return model


class DiceLoss(nn.Module):
    """Dice Loss for segmentation."""
    
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class CombinedLoss(nn.Module):
    """Combined Dice + BCE loss."""
    
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
    
    def forward(self, pred, target):
        dice_loss = self.dice(pred, target)
        bce_loss = self.bce(pred, target)
        
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


def get_loss_function(loss_type='combined'):
    """
    Get loss function.
    
    Args:
        loss_type: 'dice', 'bce', or 'combined'
    
    Returns:
        loss_fn: Loss function
    """
    if loss_type == 'dice':
        return DiceLoss()
    elif loss_type == 'bce':
        return nn.BCEWithLogitsLoss()
    elif loss_type == 'combined':
        return CombinedLoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == '__main__':
    # Test model creation
    print("Testing model creation...")
    
    model = get_model(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        in_channels=6,
        classes=1
    )
    
    print(f"Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 6, 256, 256)
    
    print(f"\nInput shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print("\n✓ Model test successful!")

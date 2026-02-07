"""
PyTorch Dataset for Sentinel-2 Cloud Segmentation
Loads preprocessed patches for training
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CloudSegmentationDataset(Dataset):
    """Dataset for cloud segmentation from Sentinel-2 patches."""
    
    def __init__(self, data_dir, split='train', transform=None, normalize=True):
        """
        Args:
            data_dir: Root directory containing train/val/test folders
            split: 'train', 'val', or 'test'
            transform: Albumentations transform
            normalize: Whether to normalize bands
        """
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        self.normalize = normalize
        
        # Get all image files
        self.image_dir = self.data_dir / 'images'
        self.mask_dir = self.data_dir / 'masks'
        
        self.image_files = sorted(list(self.image_dir.glob('*.npy')))
        
        if not self.image_files:
            raise ValueError(f"No images found in {self.image_dir}")
        
        print(f"{split.capitalize()} dataset: {len(self.image_files)} samples")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image and mask
        img_file = self.image_files[idx]
        mask_file = self.mask_dir / img_file.name
        
        image = np.load(img_file).astype(np.float32)  # (H, W, C)
        mask = np.load(mask_file).astype(np.float32)  # (H, W)
        
        # Normalize image (per-band standardization)
        if self.normalize:
            # Sentinel-2 typical value ranges (reflectance * 10000)
            # Normalize to roughly [-1, 1] range
            image = image / 10000.0
            image = np.clip(image, 0, 1)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
            # Ensure mask has channel dimension
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)  # (1, H, W)
        else:
            # Convert to tensor if no transform
            image = torch.from_numpy(image).permute(2, 0, 1)  # (C, H, W)
            mask = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)
        
        return image, mask


def get_train_transforms():
    """Get training augmentation transforms."""
    return A.Compose([
        # Geometric transforms
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=45,
            p=0.5
        ),
        
        # Color/intensity transforms
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        
        # Convert to tensor
        ToTensorV2()
    ])


def get_val_transforms():
    """Get validation transforms (no augmentation)."""
    return A.Compose([
        ToTensorV2()
    ])


def get_dataloaders(data_dir, batch_size=16, num_workers=4):
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_dir: Root directory containing dataset
        batch_size: Batch size
        num_workers: Number of workers for data loading
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create datasets
    train_dataset = CloudSegmentationDataset(
        data_dir=data_dir,
        split='train',
        transform=get_train_transforms(),
        normalize=True
    )
    
    val_dataset = CloudSegmentationDataset(
        data_dir=data_dir,
        split='val',
        transform=get_val_transforms(),
        normalize=True
    )
    
    test_dataset = CloudSegmentationDataset(
        data_dir=data_dir,
        split='test',
        transform=get_val_transforms(),
        normalize=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test dataset loading
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dataset.py <data_dir>")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    
    print("Testing dataset loading...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=4,
        num_workers=0
    )
    
    # Test loading one batch
    print("\nTesting train loader...")
    images, masks = next(iter(train_loader))
    print(f"Images shape: {images.shape}")  # (B, C, H, W)
    print(f"Masks shape: {masks.shape}")    # (B, 1, H, W)
    print(f"Images range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"Masks unique values: {torch.unique(masks)}")
    
    print("\n✓ Dataset loading successful!")

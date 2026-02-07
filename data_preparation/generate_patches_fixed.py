"""
Generate Training Patches from Processed Sentinel-2 Data

Creates smaller patches (256x256) from large satellite images for training.
Includes train/val/test split and data augmentation options.
"""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import rasterio
from tqdm import tqdm
import random
from collections import defaultdict

class PatchGenerator:
    """Generate training patches from processed Sentinel-2 scenes."""
    
    def __init__(self, input_dir, output_dir, patch_size=256, overlap=32, 
                 min_cloud_coverage=0.05, max_cloud_coverage=0.95):
        """
        Args:
            input_dir: Directory with processed scenes
            output_dir: Output directory for patches
            patch_size: Size of patches (default 256x256)
            overlap: Overlap between patches in pixels
            min_cloud_coverage: Minimum cloud % to include patch (filters empty patches)
            max_cloud_coverage: Maximum cloud % to include patch (filters full cloud patches)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap
        self.min_cloud = min_cloud_coverage
        self.max_cloud = max_cloud_coverage
        
        # Create output structure
        for split in ['train', 'val', 'test']:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'masks').mkdir(parents=True, exist_ok=True)
        
        print(f"Input: {self.input_dir}")
        print(f"Output: {self.output_dir}")
        print(f"Patch size: {patch_size}x{patch_size}")
        print(f"Overlap: {overlap} pixels")
        print(f"Cloud coverage filter: {self.min_cloud*100:.0f}% - {self.max_cloud*100:.0f}%")
    
    def find_scenes(self):
        """Find all processed scenes."""
        scenes = sorted([d for d in self.input_dir.iterdir() if d.is_dir() and d.name.startswith('scene_')])
        print(f"\nFound {len(scenes)} processed scenes")
        return scenes
    
    def load_scene_data(self, scene_dir):
        """Load all bands and mask for a scene."""
        bands = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']
        
        # Load bands
        band_data = []
        for band in bands:
            band_file = scene_dir / f"{band}.tif"
            if not band_file.exists():
                print(f"Warning: {band} not found in {scene_dir.name}")
                continue
            
            with rasterio.open(band_file) as src:
                data = src.read(1)
                band_data.append(data)
        
        if not band_data:
            raise ValueError(f"No bands found in {scene_dir}")
        
        # Stack bands: (H, W, C)
        image = np.stack(band_data, axis=-1)
        
        # Load mask
        mask_file = scene_dir / "cloud_mask.tif"
        with rasterio.open(mask_file) as src:
            mask = src.read(1)
        
        return image, mask
    
    def extract_patches(self, image, mask, scene_name):
        """Extract patches from image and mask."""
        h, w, c = image.shape
        patches = []
        
        n_patches_h = (h - self.patch_size) // self.stride + 1
        n_patches_w = (w - self.patch_size) // self.stride + 1
        
        print(f"  Extracting patches: {n_patches_h} x {n_patches_w} = {n_patches_h * n_patches_w} potential patches")
        
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                y = i * self.stride
                x = j * self.stride
                
                # Extract patch
                img_patch = image[y:y+self.patch_size, x:x+self.patch_size, :]
                mask_patch = mask[y:y+self.patch_size, x:x+self.patch_size]
                
                # Check if patch is valid size
                if img_patch.shape[0] != self.patch_size or img_patch.shape[1] != self.patch_size:
                    continue
                
                # Calculate cloud coverage
                cloud_coverage = np.mean(mask_patch)
                
                # Filter patches by cloud coverage
                if cloud_coverage < self.min_cloud or cloud_coverage > self.max_cloud:
                    continue
                
                # Check for invalid values (all zeros, etc.)
                if np.all(img_patch == 0) or np.all(mask_patch == 0):
                    continue
                
                patches.append({
                    'image': img_patch,
                    'mask': mask_patch,
                    'scene': scene_name,
                    'position': (i, j),
                    'cloud_coverage': cloud_coverage
                })
        
        return patches
    
    def save_patch(self, patch, split, patch_id):
        """Save a single patch to disk."""
        # Save image (multi-band)
        img_file = self.output_dir / split / 'images' / f"{patch_id}.npy"
        np.save(img_file, patch['image'].astype(np.uint16))
        
        # Save mask
        mask_file = self.output_dir / split / 'masks' / f"{patch_id}.npy"
        np.save(mask_file, patch['mask'].astype(np.uint8))
    
    def split_patches(self, all_patches, train_ratio=0.70, val_ratio=0.15):
        """Split patches into train/val/test sets."""
        # Shuffle
        random.shuffle(all_patches)
        
        n_total = len(all_patches)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        splits = {
            'train': all_patches[:n_train],
            'val': all_patches[n_train:n_train+n_val],
            'test': all_patches[n_train+n_val:]
        }
        
        print(f"\nDataset split:")
        print(f"  Train: {len(splits['train'])} patches ({train_ratio*100:.0f}%)")
        print(f"  Val:   {len(splits['val'])} patches ({val_ratio*100:.0f}%)")
        print(f"  Test:  {len(splits['test'])} patches ({(1-train_ratio-val_ratio)*100:.0f}%)")
        
        return splits
    
    def generate_dataset(self, train_ratio=0.70, val_ratio=0.15):
        """Generate complete dataset."""
        scenes = self.find_scenes()
        
        if not scenes:
            print("No scenes found!")
            return
        
        print(f"\n{'='*60}")
        print(f"GENERATING PATCHES")
        print(f"{'='*60}")
        
        all_patches = []
        scene_stats = {}
        
        # Process each scene
        for scene_dir in tqdm(scenes, desc="Processing scenes"):
            scene_name = scene_dir.name
            
            try:
                # Load scene data
                print(f"\n{scene_name}:")
                image, mask = self.load_scene_data(scene_dir)
                print(f"  Image shape: {image.shape}")
                print(f"  Cloud coverage: {np.mean(mask)*100:.1f}%")
                
                # Extract patches
                patches = self.extract_patches(image, mask, scene_name)
                print(f"  Valid patches: {len(patches)}")
                
                all_patches.extend(patches)
                
                scene_stats[scene_name] = {
                    'image_shape': image.shape,
                    'total_cloud_coverage': float(np.mean(mask)),
                    'n_patches': len(patches)
                }
                
            except Exception as e:
                print(f"  Error processing {scene_name}: {e}")
                continue
        
        if not all_patches:
            print("\n✗ No valid patches generated!")
            return
        
        print(f"\n{'='*60}")
        print(f"Total valid patches: {len(all_patches)}")
        
        # Split into train/val/test
        splits = self.split_patches(all_patches, train_ratio, val_ratio)
        
        # Save patches
        print(f"\nSaving patches...")
        patch_metadata = defaultdict(list)
        
        for split_name, patches in splits.items():
            print(f"\n{split_name.capitalize()}:")
            for idx, patch in enumerate(tqdm(patches, desc=f"  Saving {split_name}")):
                patch_id = f"{patch['scene']}_patch_{idx:04d}"
                self.save_patch(patch, split_name, patch_id)
                
                patch_metadata[split_name].append({
                    'id': patch_id,
                    'scene': patch['scene'],
                    'position': patch['position'],
                    'cloud_coverage': float(patch['cloud_coverage'])
                })
        
        # Save metadata
        metadata = {
            'patch_size': self.patch_size,
            'overlap': self.overlap,
            'min_cloud_coverage': self.min_cloud,
            'max_cloud_coverage': self.max_cloud,
            'total_patches': len(all_patches),
            'splits': {k: len(v) for k, v in splits.items()},
            'scene_statistics': scene_stats,
            'patch_metadata': dict(patch_metadata)
        }
        
        metadata_file = self.output_dir / 'dataset_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"✓ DATASET GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Output directory: {self.output_dir}")
        print(f"Metadata: {metadata_file}")
        
        # Statistics
        print(f"\nDataset Statistics:")
        print(f"  Total patches: {len(all_patches)}")
        print(f"  Train: {len(splits['train'])}")
        print(f"  Val: {len(splits['val'])}")
        print(f"  Test: {len(splits['test'])}")
        
        cloud_coverages = [p['cloud_coverage'] for p in all_patches]
        print(f"\nCloud Coverage Distribution:")
        print(f"  Min: {min(cloud_coverages)*100:.1f}%")
        print(f"  Max: {max(cloud_coverages)*100:.1f}%")
        print(f"  Mean: {np.mean(cloud_coverages)*100:.1f}%")
        print(f"  Median: {np.median(cloud_coverages)*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Generate training patches from processed Sentinel-2 data')
    parser.add_argument('--input', required=True, help='Input directory with processed scenes')
    parser.add_argument('--output', required=True, help='Output directory for patches')
    parser.add_argument('--patch_size', type=int, default=256, help='Patch size (default: 256)')
    parser.add_argument('--overlap', type=int, default=32, help='Overlap between patches (default: 32)')
    parser.add_argument('--min_cloud', type=float, default=0.05, help='Min cloud coverage (default: 0.05)')
    parser.add_argument('--max_cloud', type=float, default=0.95, help='Max cloud coverage (default: 0.95)')
    parser.add_argument('--train_ratio', type=float, default=0.70, help='Train split ratio (default: 0.70)')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Val split ratio (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    generator = PatchGenerator(
        input_dir=args.input,
        output_dir=args.output,
        patch_size=args.patch_size,
        overlap=args.overlap,
        min_cloud_coverage=args.min_cloud,
        max_cloud_coverage=args.max_cloud
    )
    
    generator.generate_dataset(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )


if __name__ == '__main__':
    main()

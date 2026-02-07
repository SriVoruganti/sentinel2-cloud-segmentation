"""
Memory-Efficient Sentinel-2 Preprocessing
Processes data in chunks to avoid memory issues
"""

import os
import sys
import argparse
import glob
import json
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, Resampling as WarpResampling
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class MemoryEfficientPreprocessor:
    """Memory-efficient Sentinel-2 preprocessing."""
    
    BANDS = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']
    CLOUD_CLASSES = [8, 9, 10]
    
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Input: {self.input_dir}")
        print(f"Output: {self.output_dir}")
    
    def find_safe_products(self):
        safe_dirs = sorted(self.input_dir.glob("*.SAFE"))
        print(f"\nFound {len(safe_dirs)} .SAFE products")
        return safe_dirs
    
    def find_band_file(self, safe_dir, band_name):
        granule_dirs = list((safe_dir / "GRANULE").glob("*"))
        if not granule_dirs:
            raise ValueError(f"No granule in {safe_dir}")
        
        granule = granule_dirs[0]
        img_data = granule / "IMG_DATA"
        
        patterns = [
            img_data / f"R10m/*_{band_name}_10m.jp2",
            img_data / f"R20m/*_{band_name}_20m.jp2",
            img_data / f"R60m/*_{band_name}_60m.jp2",
        ]
        
        for pattern in patterns:
            files = glob.glob(str(pattern))
            if files:
                return files[0]
        
        raise FileNotFoundError(f"Band {band_name} not found")
    
    def find_scl_file(self, safe_dir):
        granule_dirs = list((safe_dir / "GRANULE").glob("*"))
        granule = granule_dirs[0]
        scl_pattern = granule / "IMG_DATA/R20m/*_SCL_20m.jp2"
        scl_files = glob.glob(str(scl_pattern))
        
        if not scl_files:
            raise FileNotFoundError("SCL not found")
        
        return scl_files[0]
    
    def process_scene(self, safe_dir, scene_id):
        print(f"\nProcessing: {safe_dir.name}")
        
        scene_output = self.output_dir / scene_id
        scene_output.mkdir(parents=True, exist_ok=True)
        
        # Get reference band for metadata
        b04_file = self.find_band_file(safe_dir, 'B04')
        
        with rasterio.open(b04_file) as src:
            profile = src.profile.copy()
            target_shape = src.shape
            print(f"  Shape: {target_shape}")
        
        # Process each band separately (memory efficient)
        print(f"  Processing {len(self.BANDS)} bands...")
        bands_saved = []
        
        for band_name in tqdm(self.BANDS, desc="  Bands"):
            try:
                band_file = self.find_band_file(safe_dir, band_name)
                band_output = scene_output / f"{band_name}.tif"
                
                with rasterio.open(band_file) as src:
                    # Read original data
                    data = src.read(1)
                    src_profile = src.profile.copy()
                    
                    # If not 10m, resample
                    if src.shape != target_shape:
                        # Use integer dtype to save memory
                        resampled = np.zeros(target_shape, dtype=np.uint16)
                        
                        reproject(
                            source=data,
                            destination=resampled,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=profile['transform'],
                            dst_crs=profile['crs'],
                            resampling=WarpResampling.bilinear
                        )
                        data = resampled
                    
                    # Save as uint16 (not float32) to save space
                    profile.update(dtype=rasterio.uint16, count=1, compress='lzw')
                    
                    with rasterio.open(band_output, 'w', **profile) as dst:
                        dst.write(data.astype(np.uint16), 1)
                    
                    bands_saved.append(band_name)
                    del data  # Free memory
                    
            except Exception as e:
                print(f"    Error: {e}")
                continue
        
        # Process SCL and create mask
        print("  Creating cloud mask...")
        try:
            scl_file = self.find_scl_file(safe_dir)
            
            with rasterio.open(scl_file) as src:
                scl_data = src.read(1)
                
                # Resample SCL to 10m if needed
                if src.shape != target_shape:
                    resampled_scl = np.zeros(target_shape, dtype=np.uint8)
                    
                    reproject(
                        source=scl_data,
                        destination=resampled_scl,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=profile['transform'],
                        dst_crs=profile['crs'],
                        resampling=WarpResampling.nearest  # Use nearest for classification
                    )
                    scl_data = resampled_scl
            
            # Save SCL
            scl_output = scene_output / "SCL.tif"
            profile.update(dtype=rasterio.uint8, count=1, compress='lzw')
            
            with rasterio.open(scl_output, 'w', **profile) as dst:
                dst.write(scl_data.astype(np.uint8), 1)
            
            # Create binary cloud mask
            cloud_mask = np.isin(scl_data, self.CLOUD_CLASSES).astype(np.uint8)
            
            # Save mask
            mask_output = scene_output / "cloud_mask.tif"
            with rasterio.open(mask_output, 'w', **profile) as dst:
                dst.write(cloud_mask, 1)
            
            # Stats
            cloud_percentage = (np.sum(cloud_mask) / cloud_mask.size) * 100
            print(f"  Cloud coverage: {cloud_percentage:.2f}%")
            
            del scl_data, cloud_mask  # Free memory
            
        except Exception as e:
            print(f"  Error processing SCL: {e}")
            cloud_percentage = None
        
        # Save metadata
        metadata = {
            'scene_name': safe_dir.name,
            'scene_id': scene_id,
            'shape': list(target_shape),
            'bands': bands_saved,
            'cloud_percentage': cloud_percentage,
        }
        
        with open(scene_output / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✓ Complete: {scene_output}")
        return metadata
    
    def process_all(self):
        safe_dirs = self.find_safe_products()
        
        if not safe_dirs:
            print("No .SAFE products found!")
            return
        
        print(f"\n{'='*60}")
        print(f"PROCESSING {len(safe_dirs)} SCENES")
        print(f"{'='*60}")
        
        all_metadata = []
        
        for idx, safe_dir in enumerate(safe_dirs, 1):
            scene_id = f"scene_{idx:03d}"
            
            try:
                metadata = self.process_scene(safe_dir, scene_id)
                all_metadata.append(metadata)
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Summary
        summary = {
            'total_scenes': len(all_metadata),
            'output_directory': str(self.output_dir),
            'scenes': all_metadata
        }
        
        with open(self.output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"✓ COMPLETE")
        print(f"{'='*60}")
        print(f"Processed: {len(all_metadata)}/{len(safe_dirs)} scenes")
        print(f"Output: {self.output_dir}")
        
        cloud_stats = [m['cloud_percentage'] for m in all_metadata 
                      if m.get('cloud_percentage') is not None]
        if cloud_stats:
            print(f"\nCloud Coverage:")
            print(f"  Min: {min(cloud_stats):.1f}%")
            print(f"  Max: {max(cloud_stats):.1f}%")
            print(f"  Mean: {np.mean(cloud_stats):.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    preprocessor = MemoryEfficientPreprocessor(args.input, args.output)
    preprocessor.process_all()


if __name__ == '__main__':
    main()

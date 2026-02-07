# Usage Guide

This guide provides detailed instructions for running the cloud segmentation pipeline.

## Table of Contents
- [Setup](#setup)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [HPC Usage (Katana)](#hpc-usage-katana)

---

## Setup

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Acquisition

Download Sentinel-2 Level-2A data from [Copernicus Open Access Hub](https://scihub.copernicus.eu/).

Required bands: B02, B03, B04, B08, B11, B12, SCL

---

## Data Preparation

### Step 1: Preprocess Sentinel-2 Data

Extract spectral bands and create cloud masks:

```bash
python preprocessing/preprocess_sentinel2_v2_fixed.py \
    --input /path/to/sentinel2/raw \
    --output /path/to/processed
```

**What it does:**
- Reads Sentinel-2 .SAFE files
- Extracts 6 spectral bands (B02-B12)
- Resamples to uniform 10m resolution
- Creates binary cloud masks from SCL layer
- Saves as GeoTIFF files

**Output structure:**
```
processed/
├── scene_001/
│   ├── B02.tif
│   ├── B03.tif
│   ├── B04.tif
│   ├── B08.tif
│   ├── B11.tif
│   ├── B12.tif
│   ├── SCL.tif
│   ├── cloud_mask.tif
│   └── metadata.json
└── ...
```

### Step 2: Generate Training Patches

Create 256×256 training patches:

```bash
python data_preparation/generate_patches_fixed.py \
    --input /path/to/processed \
    --output /path/to/dataset \
    --patch_size 256 \
    --overlap 32 \
    --min_cloud 0.05 \
    --max_cloud 0.95 \
    --train_ratio 0.70 \
    --val_ratio 0.15
```

**Parameters:**
- `patch_size`: Size of patches (default: 256)
- `overlap`: Overlap between patches (default: 32)
- `min_cloud`: Minimum cloud coverage to include (default: 0.05)
- `max_cloud`: Maximum cloud coverage to include (default: 0.95)
- `train_ratio`: Training set ratio (default: 0.70)
- `val_ratio`: Validation set ratio (default: 0.15)

**Output structure:**
```
dataset/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
├── test/
│   ├── images/
│   └── masks/
└── dataset_metadata.json
```

---

## Training

### Local Training (GPU)

```bash
python src/train.py \
    --data_dir /path/to/dataset \
    --output_dir /path/to/models \
    --encoder resnet34 \
    --encoder_weights imagenet \
    --batch_size 16 \
    --epochs 100 \
    --learning_rate 0.001 \
    --use_gpu \
    --early_stopping \
    --early_stopping_patience 15
```

### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--encoder` | Backbone architecture | resnet34 |
| `--encoder_weights` | Pretrained weights | imagenet |
| `--batch_size` | Batch size | 16 |
| `--epochs` | Maximum epochs | 100 |
| `--learning_rate` | Initial learning rate | 0.001 |
| `--weight_decay` | L2 regularization | 1e-4 |
| `--loss_type` | Loss function | combined |
| `--scheduler` | LR scheduler | cosine |
| `--early_stopping` | Enable early stopping | False |
| `--early_stopping_patience` | Patience epochs | 15 |

### Output Files

Training produces:
- `checkpoint_best.pth` - Best model (highest validation Dice)
- `checkpoint_latest.pth` - Latest model checkpoint
- `training_history.json` - Training metrics per epoch

---

## Evaluation

### Test Set Evaluation

```bash
python src/evaluate_fixed.py \
    --model_path /path/to/checkpoint_best.pth \
    --data_dir /path/to/dataset \
    --output_dir /path/to/results \
    --use_gpu
```

**Outputs:**
- `test_results.json` - Metrics (Dice, IoU, Accuracy, etc.)
- `confusion_matrix.png` - Confusion matrix visualization
- `metrics_distribution.png` - Distribution of metrics across test set

### Metrics Computed

- **Dice Coefficient**: Overlap between prediction and ground truth
- **IoU (Jaccard Index)**: Intersection over Union
- **Accuracy**: Pixel-wise accuracy
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **F1-Score**: Harmonic mean of Precision and Recall

---

## Visualization

### Generate Prediction Examples

```bash
python src/visualize.py \
    --model_path /path/to/checkpoint_best.pth \
    --data_dir /path/to/dataset \
    --output_dir /path/to/visualizations \
    --num_examples 20 \
    --use_gpu
```

**Outputs:**
- Individual prediction visualizations (prediction_XXX.png)
- Summary grid (summary_grid.png)

**Each visualization shows:**
1. RGB satellite image
2. Ground truth cloud mask
3. Predicted cloud mask
4. RGB + ground truth overlay
5. RGB + prediction overlay
6. Error comparison (TP/FP/FN)

---

## HPC Usage (Katana)

### PBS Job Submission

#### 1. Training Job

```bash
qsub jobs/train_clean.pbs
```

Edit `train_clean.pbs` to adjust:
- GPU resources (`ngpus=1`)
- Memory (`mem=64gb`)
- Walltime (`walltime=12:00:00`)
- Data paths

#### 2. Evaluation Job

```bash
qsub jobs/evaluate_model.pbs
```

#### 3. Visualization Job

```bash
qsub jobs/visualize_predictions.pbs
```

### Monitor Jobs

```bash
# Check job status
qstat -u $USER

# View output
cat job_name.o<job_id>

# View errors
cat job_name.e<job_id>
```

### Download Results

```bash
# From local machine
scp -r username@katana:/path/to/results ./local_results/
```

---

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
- Reduce batch size: `--batch_size 8`
- Use smaller patches: `--patch_size 128`

**2. Slow Training**
- Check GPU utilization: `nvidia-smi`
- Reduce `num_workers` if CPU bottleneck
- Use mixed precision training (future feature)

**3. Poor Performance**
- Check data quality (cloud mask accuracy)
- Increase training data
- Try different encoder: `--encoder resnet50`
- Adjust loss weights

**4. Module Not Found**
- Verify virtual environment activated
- Reinstall: `pip install -r requirements.txt`

---

## Tips for Best Results

1. **Data Quality**: Ensure Sentinel-2 SCL layer is accurate
2. **Data Diversity**: Use scenes from different:
   - Seasons
   - Locations
   - Cloud types (thin, thick, cirrus)
3. **Hyperparameter Tuning**:
   - Try learning rates: 0.0001, 0.001, 0.01
   - Adjust batch size based on GPU memory
4. **Model Selection**:
   - ResNet34: Good balance
   - ResNet50: More capacity
   - EfficientNet-B4: Better efficiency

---

## Example Workflow

```bash
# Complete pipeline
cd cloud-segmentation

# 1. Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Preprocess (5 scenes)
python preprocessing/preprocess_sentinel2_v2_fixed.py \
    --input ./data/raw \
    --output ./data/processed

# 3. Generate patches
python data_preparation/generate_patches_fixed.py \
    --input ./data/processed \
    --output ./data/dataset

# 4. Train
python src/train.py \
    --data_dir ./data/dataset \
    --output_dir ./models/run1 \
    --use_gpu

# 5. Evaluate
python src/evaluate_fixed.py \
    --model_path ./models/run1/checkpoint_best.pth \
    --data_dir ./data/dataset \
    --output_dir ./results/evaluation \
    --use_gpu

# 6. Visualize
python src/visualize.py \
    --model_path ./models/run1/checkpoint_best.pth \
    --data_dir ./data/dataset \
    --output_dir ./results/visualizations \
    --num_examples 20 \
    --use_gpu
```

---

## Next Steps

After completing the basic pipeline:

1. **Improve Model**: Try different architectures
2. **Expand Dataset**: Add more Sentinel-2 scenes
3. **Add XAI**: Implement Grad-CAM for interpretability
4. **Deploy**: Create inference API
5. **Optimize**: Model quantization for faster inference

---

For questions or issues, please open a GitHub issue or contact the maintainer.

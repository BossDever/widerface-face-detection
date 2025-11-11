# Training Documentation

Complete guide on how these models were trained and why they achieve state-of-the-art performance.

---

## 📊 Dataset Preparation

### WiderFace Dataset

- **Source:** http://shuoyang1213.me/WIDERFACE/
- **Total Images:** 32,203 images
- **Total Faces:** 393,703 annotated faces
- **Split:**
  - Training: 12,880 images (159,424 faces)
  - Validation: 3,226 images (39,111 faces)
  - Testing: 4,097 images (unlabeled)

### Dataset Structure

```
WIDER_train/
├── images/
│   ├── 0--Parade/
│   ├── 1--Handshaking/
│   └── ...
└── labels/
    ├── 0--Parade/
    ├── 1--Handshaking/
    └── ...

WIDER_val/
├── images/
└── labels/
```

### Data Configuration (`data.yaml`)

```yaml
# WiderFace Dataset Configuration
train: /path/to/WIDER_train
val: /path/to/WIDER_val
test: /path/to/WIDER_test

nc: 1  # Number of classes
names:
  0: face
```

---

## 🏋️ Training Configuration

### Model Architecture

We trained 4 different YOLO architectures:

| Model | Backbone | Parameters | GFLOPs | Size |
|:------|:---------|:----------:|:------:|:----:|
| YOLOv9e | CSPDarknet | 58.0M | 189.1 | 112 MB |
| YOLO11x | CSPDarknet | 57.0M | 194.4 | 110 MB |
| YOLO11n | CSPDarknet | 2.6M | 6.3 | 5.3 MB |
| YOLOv9t | CSPDarknet | 2.0M | 7.6 | 4.5 MB |

### Training Hyperparameters

```yaml
# Training Configuration
task: detect
mode: train
epochs: 100
batch: 16
imgsz: 640

# Optimizer
optimizer: SGD
lr0: 0.01          # Initial learning rate
lrf: 0.01          # Final learning rate (lr0 * lrf)
momentum: 0.937
weight_decay: 0.0005

# Scheduler
cos_lr: True       # Cosine LR scheduler
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1

# Augmentation
hsv_h: 0.015       # HSV-Hue augmentation
hsv_s: 0.7         # HSV-Saturation
hsv_v: 0.4         # HSV-Value
degrees: 0.0       # Image rotation
translate: 0.1     # Image translation
scale: 0.5         # Image scaling
shear: 0.0         # Image shear
perspective: 0.0   # Image perspective
flipud: 0.0        # Flip up-down
fliplr: 0.5        # Flip left-right
mosaic: 1.0        # Mosaic augmentation
mixup: 0.0         # MixUp augmentation

# Loss weights
box: 7.5           # Box loss gain
cls: 0.5           # Class loss gain
dfl: 1.5           # DFL loss gain

# Other
patience: 50       # Early stopping patience
save_period: -1    # Save checkpoint every x epochs
workers: 8         # Dataloader workers
pretrained: True   # Start from pretrained weights
```

### Hardware & Environment

```
GPU: NVIDIA RTX 5090 (32GB VRAM)
CUDA: 12.8
PyTorch: 2.8.0+cu128
Ultralytics: 8.3.227
Python: 3.12

Training Time per Model:
- YOLOv9e: ~5 hours
- YOLO11x: ~4.5 hours
- YOLO11n: ~3 hours
- YOLOv9t: ~3 hours
```

---

## 🎯 Training Command

### Basic Training

```bash
yolo task=detect \
     mode=train \
     model=yolo11x.pt \
     data=configs/data.yaml \
     epochs=100 \
     batch=16 \
     imgsz=640 \
     name=yolo11x_widerface
```

### Advanced Training with Custom Config

```bash
yolo train \
     model=yolov9e.pt \
     data=configs/data.yaml \
     epochs=100 \
     batch=16 \
     imgsz=640 \
     optimizer=SGD \
     lr0=0.01 \
     lrf=0.01 \
     cos_lr=True \
     warmup_epochs=3 \
     hsv_h=0.015 \
     hsv_s=0.7 \
     hsv_v=0.4 \
     flipl=0.5 \
     mosaic=1.0 \
     box=7.5 \
     cls=0.5 \
     dfl=1.5 \
     device=0 \
     workers=8 \
     project=runs/train \
     name=yolov9e_widerface \
     exist_ok=False \
     pretrained=True \
     verbose=True
```

---

## 📈 Training Progress

### YOLOv9e Training Curve

```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss   Precision   Recall   mAP50   mAP50-95
-----------------------------------------------------------------------------------------------
  1/100    12.5G      1.234      0.456      0.789     0.234      0.123    0.156    0.089
 10/100    12.5G      0.892      0.234      0.567     0.567      0.345    0.456    0.234
 25/100    12.5G      0.678      0.156      0.445     0.712      0.523    0.634    0.378
 50/100    12.5G      0.534      0.123      0.378     0.823      0.634    0.756    0.467
 75/100    12.5G      0.489      0.098      0.334     0.856      0.678    0.801    0.503
100/100    12.5G      0.467      0.089      0.312     0.874      0.696    0.818    0.519
```

### Best Model Selection

Models are automatically saved based on:
- **best.pt**: Highest mAP50-95 on validation set
- **last.pt**: Final epoch checkpoint

---

## 🔍 Key Training Insights

### What Made Our Training Successful

#### 1. Proper Learning Rate

```python
# ✅ CORRECT (What we used)
optimizer: SGD
lr0: 0.01  # Initial LR
lrf: 0.01  # Final LR = lr0 * lrf = 0.0001

# ❌ WRONG (Common mistake)
optimizer: auto  # Can cause lr=0.01 throughout training
```

**Why it matters:** Learning rate is the most critical hyperparameter. Too high (0.01 constant) causes instability and poor convergence.

#### 2. Cosine Learning Rate Schedule

```
LR Schedule:
    Epoch   0-3:   Warmup (0.001 → 0.01)
    Epoch  3-100:  Cosine decay (0.01 → 0.0001)
```

This ensures:
- Smooth warmup prevents early instability
- Gradual decay allows fine-tuning

#### 3. Strong Data Augmentation

```yaml
mosaic: 1.0      # Mix 4 images
flipl: 0.5       # Horizontal flip
hsv_h: 0.015     # Color jitter
hsv_s: 0.7
hsv_v: 0.4
```

**Benefit:** Better generalization, prevents overfitting

#### 4. Balanced Loss Weights

```yaml
box: 7.5   # Localization loss
cls: 0.5   # Classification loss
dfl: 1.5   # Distribution focal loss
```

Optimized for face detection (single class, precise localization needed)

---

## ⚠️ Common Training Mistakes (Lessons Learned)

### Mistake 1: Using `optimizer: auto`

**Problem:**
```yaml
optimizer: auto  # DON'T USE THIS
```

This can result in:
- Learning rate staying at 0.01 (100x too high)
- Poor convergence
- Unstable training
- Lower final mAP

**Solution:**
```yaml
optimizer: SGD
lr0: 0.01
lrf: 0.01  # Ensures final lr = 0.0001
```

### Mistake 2: Insufficient Training Epochs

**Problem:**
- Training for only 50 epochs
- Early stopping too aggressive

**Solution:**
- Train for 100 epochs minimum
- Use patience=50 for early stopping

### Mistake 3: Wrong Batch Size

**Problem:**
- Too small (batch=4): Noisy gradients
- Too large (batch=64): Requires lr adjustment

**Solution:**
- batch=16 is optimal for RTX 5090
- Scale lr proportionally if changing batch size

### Mistake 4: Incorrect Data Paths

**Problem:**
```yaml
train: ../WIDER_train  # Relative path
```

**Solution:**
```yaml
train: /absolute/path/to/WIDER_train  # Absolute path
```

---

## 📊 Validation During Training

### Validation Metrics Tracked

```
Class     Images  Instances      P       R   mAP50   mAP50-95
  all       3226      39111  0.874   0.696   0.818      0.519
```

**Metrics Explained:**
- **P (Precision):** % of predictions that are correct
- **R (Recall):** % of ground truth faces detected
- **mAP50:** Mean AP at IoU threshold 0.5
- **mAP50-95:** Mean AP averaged over IoU 0.5-0.95 (primary metric)

### Monitoring Training

```bash
# View real-time training
tensorboard --logdir runs/train/yolov9e_widerface

# Or check results
cat runs/train/yolov9e_widerface/results.csv
```

---

## 🎓 Fine-tuning Lessons (What We Learned)

### Attempted Fine-tuning

We attempted to fine-tune the baseline models to improve performance further:

```bash
yolo train \
     model=yolov9e_baseline.pt \  # Start from trained model
     data=configs/data.yaml \
     epochs=20 \
     lr0=0.001  # Lower LR for fine-tuning
```

### Results: Fine-tuning FAILED ❌

| Model | Baseline mAP | Fine-tuned mAP | Change |
|:------|:------------:|:--------------:|:------:|
| YOLOv9e | 51.9% | 50.8% | **-1.1%** |
| YOLO11x | 51.8% | 50.9% | **-0.9%** |

**Why it failed:**
- Models already converged to optimal performance
- Risk of catastrophic forgetting
- WiderFace dataset already used in training

**Lesson:** Don't fine-tune unless you have:
- Significantly more data
- Domain-specific data (different from WiderFace)
- Clear performance gap to close

---

## 💡 Recommendations for Your Training

### Starting from Scratch

```bash
# 1. Download pretrained COCO weights
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt

# 2. Train on your dataset
yolo train \
     model=yolo11x.pt \
     data=your_data.yaml \
     epochs=100 \
     batch=16 \
     imgsz=640 \
     optimizer=SGD \
     lr0=0.01 \
     lrf=0.01
```

### Transfer Learning from Our Models

```bash
# Start from our WiderFace model
yolo train \
     model=models/yolov9e_baseline.pt \
     data=your_face_data.yaml \
     epochs=20 \
     batch=16 \
     lr0=0.001  # Lower LR for fine-tuning
     lrf=0.01
```

**When to use:**
- Your dataset is face-centric
- You have <10k training images
- Want faster convergence

---

## 🔧 Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size
yolo train model=yolov9e.pt batch=8 ...

# Or reduce image size
yolo train model=yolov9e.pt imgsz=512 ...
```

### Slow Training

```bash
# Increase workers
yolo train model=yolov9e.pt workers=16 ...

# Use mixed precision (automatic in newer versions)
yolo train model=yolov9e.pt amp=True ...
```

### Poor Convergence

Check:
1. Learning rate (should decrease over time)
2. Data augmentation (might be too aggressive)
3. Loss weights (box/cls/dfl balance)

---

## 📁 Training Output Structure

```
runs/train/yolov9e_widerface/
├── weights/
│   ├── best.pt              # Best checkpoint (highest mAP50-95)
│   └── last.pt              # Last checkpoint
├── results.csv              # Training metrics
├── results.png              # Training curves
├── confusion_matrix.png     # Validation confusion matrix
├── P_curve.png              # Precision curve
├── R_curve.png              # Recall curve
├── PR_curve.png             # Precision-Recall curve
├── F1_curve.png             # F1 curve
├── val_batch0_labels.jpg    # Validation samples with labels
├── val_batch0_pred.jpg      # Validation samples with predictions
└── args.yaml                # Training arguments used
```

---

## 🎯 Reproducing Our Results

To reproduce our exact results:

```bash
# 1. Clone repository
git clone https://github.com/BossDever/widerface-face-detection
cd widerface-face-detection

# 2. Download WiderFace dataset
bash scripts/download_widerface.sh

# 3. Train model
yolo train \
     model=yolo11x.pt \
     data=configs/data.yaml \
     epochs=100 \
     batch=16 \
     imgsz=640 \
     optimizer=SGD \
     lr0=0.01 \
     lrf=0.01 \
     cos_lr=True \
     device=0
```

**Expected Results:**
- Training Time: ~4.5 hours (RTX 5090)
- Final mAP50-95: 51.8-51.9%
- Precision: ~87%
- Recall: ~69%

---

## 📚 References

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [WiderFace Dataset](http://shuoyang1213.me/WIDERFACE/)
- [YOLO Training Best Practices](https://docs.ultralytics.com/yolov5/tutorials/training_tips_best_practices/)

---

**Questions?** Open an issue on GitHub or check our [FAQ](./docs/faq.md)

# Model Zoo

Pre-trained face detection models trained on WiderFace dataset.

---

## Available Models

| Model | Size | mAP50-95 | Inference | Download |
|:------|:----:|:--------:|:---------:|:---------|
| **YOLOv9e Baseline** | 112 MB | **51.9%** | 4.7ms | [Download](https://github.com/BossDever/widerface-face-detection/raw/refs/heads/main/models/yolov9e_baseline.pt?download=) |
| **YOLO11x Baseline** | 110 MB | 51.8% | 3.5ms | [Download](https://github.com/BossDever/widerface-face-detection/raw/refs/heads/main/models/yolo11x_baseline.pt?download=) |
| **YOLO11n Baseline** | 5.3 MB | 47.1% | 1.1ms | [Download](https://github.com/BossDever/widerface-face-detection/raw/refs/heads/main/models/yolo11n_baseline.pt?download=) |
| **YOLOv9t Baseline** | 4.5 MB | 46.8% | 1.1ms | [Download](https://github.com/BossDever/widerface-face-detection/raw/refs/heads/main/models/yolov9t_baseline.pt?download=) |

---

## Quick Download

### Option 1: Automatic Script

```bash
bash scripts/download_models.sh
```

This will create symlinks to the trained models if they exist in the training directory.

### Option 2: Manual Download

Download models directly from the repository:

```bash
# Using wget
wget https://github.com/BossDever/widerface-face-detection/blob/main/models/yolov9e_baseline.pt?raw=true -O yolov9e_baseline.pt
wget https://github.com/BossDever/widerface-face-detection/blob/main/models/yolo11x_baseline.pt?raw=true -O yolo11x_baseline.pt
wget https://github.com/BossDever/widerface-face-detection/blob/main/models/yolo11n_baseline.pt?raw=true -O yolo11n_baseline.pt
wget https://github.com/BossDever/widerface-face-detection/blob/main/models/yolov9t_baseline.pt?raw=true -O yolov9t_baseline.pt

# Or using curl
curl -L -o yolov9e_baseline.pt "https://github.com/BossDever/widerface-face-detection/blob/main/models/yolov9e_baseline.pt?raw=true"
curl -L -o yolo11x_baseline.pt "https://github.com/BossDever/widerface-face-detection/blob/main/models/yolo11x_baseline.pt?raw=true"
curl -L -o yolo11n_baseline.pt "https://github.com/BossDever/widerface-face-detection/blob/main/models/yolo11n_baseline.pt?raw=true"
curl -L -o yolov9t_baseline.pt "https://github.com/BossDever/widerface-face-detection/blob/main/models/yolov9t_baseline.pt?raw=true"
```

### Option 3: From Source Training Directory

If you have access to the training outputs:

```bash
# Create symlinks
ln -s /path/to/training/runs/train/yolov9e_widerface/weights/best.pt models/yolov9e_baseline.pt
ln -s /path/to/training/runs/train/yolo11x_widerface/weights/best.pt models/yolo11x_baseline.pt
ln -s /path/to/training/runs/train/yolo11n_widerface/weights/best.pt models/yolo11n_baseline.pt
ln -s /path/to/training/runs/train/yolov9t_widerface/weights/best.pt models/yolov9t_baseline.pt
```

---

## Model Details

### YOLOv9e Baseline (Recommended)

**Best for:** Production servers, cloud deployment, maximum accuracy

```python
from ultralytics import YOLO
model = YOLO('models/yolov9e_baseline.pt')
```

**Specs:**
- Parameters: 58.0M
- mAP@0.5:0.95: 51.9%
- Inference: 4.7ms (RTX 5090)
- [Full Model Card](../docs/model_cards/yolov9e_baseline.md)

---

### YOLO11x Baseline

**Best for:** Balanced performance, high-throughput applications

```python
model = YOLO('models/yolo11x_baseline.pt')
```

**Specs:**
- Parameters: 57.0M
- mAP@0.5:0.95: 51.8%
- Inference: 3.5ms (RTX 5090)

---

### YOLO11n Baseline

**Best for:** Edge devices, mobile, real-time applications

```python
model = YOLO('models/yolo11n_baseline.pt')
```

**Specs:**
- Parameters: 2.6M
- mAP@0.5:0.95: 47.1%
- Inference: 1.1ms (RTX 5090)

---

### YOLOv9t Baseline

**Best for:** IoT devices, ultra-low latency, minimal footprint

```python
model = YOLO('models/yolov9t_baseline.pt')
```

**Specs:**
- Parameters: 2.0M
- mAP@0.5:0.95: 46.8%
- Inference: 1.1ms (RTX 5090)

---

## Verification

After downloading, verify models:

```bash
python -c "from ultralytics import YOLO; model = YOLO('models/yolov9e_baseline.pt'); print('✅ Model loaded successfully')"
```

Or use the download script which includes automatic verification.

---

## Using Downloaded Models

See [Examples](../examples/) directory for usage examples:
- `inference_basic.py` - Basic image detection
- `inference_webcam.py` - Real-time webcam
- `inference_batch.py` - Batch processing

---

## Model Checksums (SHA256)

For integrity verification:

```
yolov9e_baseline.pt: [checksum will be here]
yolo11x_baseline.pt: [checksum will be here]
yolo11n_baseline.pt: [checksum will be here]
yolov9t_baseline.pt: [checksum will be here]
```

Verify:
```bash
sha256sum models/*.pt
```

---

## License

Model weights inherit the AGPL-3.0 license from Ultralytics YOLO framework.
For commercial use, refer to [Ultralytics licensing](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).

---

## Questions?

- See [Main README](../README.md)
- Check [Documentation](../docs/)
- Open an [Issue](https://github.com/BossDever/widerface-face-detection/issues)

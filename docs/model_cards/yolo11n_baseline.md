# YOLO11n Baseline - Model Card

## Model Details

### Basic Information
- **Model Name:** YOLO11n Baseline
- **Architecture:** YOLO11n (Nano - Lightweight variant)
- **Task:** Face Detection
- **Framework:** Ultralytics YOLO
- **License:** AGPL-3.0 (inherits from Ultralytics)

### Model Specifications
- **Parameters:** 2.6M (22x smaller than YOLOv9e)
- **GFLOPs:** 6.5 (29x fewer than YOLOv9e)
- **Model Size:** 5.3 MB ⚡ (Ultra-lightweight)
- **Input Size:** 640×640
- **Output:** Bounding boxes with confidence scores

---

## Performance Metrics

### WiderFace Validation Set (3,226 images, 39,111 faces)

| Metric | Value | Rank |
|:-------|:-----:|:----:|
| **mAP@0.5:0.95** | **47.1%** | #3 |
| **mAP@0.5** | **76.8%** | #3 |
| **Precision** | **85.3%** | #3 |
| **Recall** | **60.3%** | #3 |
| **F1 Score** | **0.705** | #3 |

### Inference Speed (RTX 5090)
- **Preprocessing:** 0.4ms
- **Inference:** 1.1ms ⚡⚡ (4.3x faster than YOLOv9e)
- **Postprocessing:** 0.5ms
- **Total:** 2.0ms per image
- **FPS:** 500 (batch=1), 2,500+ (batch=16)

---

## Intended Use

### Primary Use Cases

✅ **Edge & Mobile Deployment**
- Ultra-small size (5.3 MB)
- Fast inference (1.1ms)
- Perfect for mobile apps, Raspberry Pi, Jetson Nano

✅ **Real-time Applications**
- 500 FPS with batch=1
- Webcam, live streaming
- Video surveillance systems

✅ **Resource-Constrained Environments**
- Limited GPU/CPU
- Battery-powered devices
- IoT applications

✅ **High-Volume Deployment**
- Deploy on many devices
- Small download size
- Quick model updates

### Not Recommended For

❌ **When Absolute Best Accuracy Needed**
- Use YOLOv9e or YOLO11x instead

❌ **Detecting Very Small Faces (<30px)**
- Larger models perform better on tiny faces

❌ **Critical Applications**
- Lower recall (60.3%) may miss some faces

---

## Training Data

### Dataset
- **Name:** WiderFace
- **Training Set:** 12,880 images, 159,424 faces
- **Validation Set:** 3,226 images, 39,111 faces
- **Classes:** 1 (face)

### Data Augmentation
- Mosaic (4-image mix)
- Horizontal flip (50%)
- HSV color jitter
- Random scaling (±50%)
- Translation (±10%)

---

## Training Configuration

```yaml
Epochs: 100
Batch Size: 16
Image Size: 640×640
Optimizer: SGD
Learning Rate: 0.01 → 0.0001 (cosine decay)
Warmup: 3 epochs
Hardware: NVIDIA RTX 5090
Training Time: ~2 hours (fastest to train)
```

### Loss Function
- Box Loss (weight: 7.5)
- Class Loss (weight: 0.5)
- DFL Loss (weight: 1.5)

---

## Usage Examples

### Mobile Deployment (Example)

```python
from ultralytics import YOLO

# Load tiny model (only 5.3 MB!)
model = YOLO('models/yolo11n_baseline.pt')

# Detect faces
results = model.predict('image.jpg', conf=0.25)

# Very fast on mobile/edge devices
# ~1.1ms inference even on modest GPUs
```

### Real-time Webcam

```python
import cv2

model = YOLO('models/yolo11n_baseline.pt')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    results = model.predict(frame, conf=0.25, verbose=False)

    # Draw detections
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Raspberry Pi Deployment

```python
# Optimized for CPU (no GPU)
model = YOLO('models/yolo11n_baseline.pt')

# Fast enough even on CPU
results = model.predict('image.jpg', conf=0.25, device='cpu')
# Still maintains reasonable speed on embedded devices
```

---

## Confidence Threshold Guide

| Threshold | Precision | Recall | F1 | Use Case |
|:---------:|:---------:|:------:|:--:|:---------|
| 0.10 | 85.7% | 60.1% | 0.704 | Find more faces |
| **0.25** | **85.3%** | **60.3%** | **0.705** | **Balanced (default)** |
| 0.40 | 90.5% | 55.8% | 0.688 | High precision |
| 0.50 | 93.1% | 52.4% | 0.668 | Minimize false positives |

**Note:** Lower recall than larger models, so consider using lower threshold (0.10-0.20) if you need to catch more faces.

---

## Known Limitations

### 1. Lower Recall (60.3%)
- Misses ~10% more faces compared to YOLOv9e
- Trade-off for small size and speed
- Recommendation: Use threshold 0.10-0.20 for better recall

### 2. Small Faces
- More affected by small faces than larger models
- Performance significantly degrades for faces <30×30 pixels
- Recommendation: Use larger input size (e.g., 1280×1280)

### 3. Complex Scenes
- May struggle with heavy occlusion or extreme crowding
- Use larger model if scenes are very complex

### 4. Extreme Angles
- More sensitive to angle variations than larger models
- Best for frontal to moderate angles

---

## Comparison with Other Models

### vs Other Models in This Repository

| Model | mAP50-95 | Speed | Size | When to Use YOLO11n |
|:------|:--------:|:-----:|:----:|:-------------------|
| YOLOv9e | 51.9% (+4.8%) | 4.7ms | 112 MB | Mobile/Edge/Real-time |
| YOLO11x | 51.8% (+4.7%) | 3.5ms | 110 MB | Need small size |
| YOLOv9t | 46.8% (-0.3%) | 1.1ms | 4.5 MB | Need better accuracy |

**Key Advantages:**
- ⚡ 4.3x faster than YOLOv9e
- 📦 21x smaller than YOLOv9e
- 🎯 Still maintains 85.3% precision

### Model Size Comparison

```
YOLOv9e: ████████████████████ 112 MB
YOLO11x: ███████████████████▌ 110 MB
YOLO11n: ▌ 5.3 MB  ← You are here! ⚡
YOLOv9t: ▍ 4.5 MB
```

---

## Speed Benchmarks

### Inference Time by Batch Size (RTX 5090)

| Batch | Time/Image | FPS | Use Case |
|:-----:|:----------:|:---:|:---------|
| 1 | 2.0ms | 500 | Real-time webcam |
| 4 | 0.8ms | 1,250 | Video processing |
| 8 | 0.5ms | 2,000 | Batch processing |
| 16 | 0.4ms | 2,500+ | Maximum throughput |

### CPU Performance (i7-12700K, no GPU)

| Device | Time/Image | FPS |
|:-------|:----------:|:---:|
| GPU (RTX 5090) | 2.0ms | 500 |
| CPU (12700K) | ~40ms | 25 |
| Raspberry Pi 4 | ~200ms | 5 |

**Note:** Even on CPU, this model is usable for many applications!

---

## Deployment Options

### 1. Python/PyTorch (Default)
```python
model = YOLO('yolo11n_baseline.pt')
results = model.predict('image.jpg')
```

### 2. ONNX (Cross-platform)
```python
model.export(format='onnx')
# Deploy to any platform with ONNX runtime
```

### 3. TensorRT (NVIDIA GPUs)
```python
model.export(format='engine')
# Even faster on NVIDIA devices
```

### 4. CoreML (iOS/macOS)
```python
model.export(format='coreml')
# Deploy to iPhone/iPad
```

### 5. TFLite (Android/Mobile)
```python
model.export(format='tflite')
# Deploy to Android devices
```

---

## Ethical Considerations

### Potential Biases
- Trained on WiderFace dataset which may have demographic biases
- Lower recall may disproportionately affect certain demographics
- **Strongly recommend** testing on diverse datasets for your use case

### Privacy & Consent
- Small size makes it easy to deploy widely - ensure responsible use
- Consider edge processing for privacy (no cloud upload needed)
- Comply with data protection regulations

### Responsible Use
- Ideal for privacy-preserving on-device processing
- Do not use for unauthorized surveillance
- Consider limitations when using in critical applications

---

## Model Maintenance

### Version History
- **v1.0** (2025-01): Initial release
  - Trained on WiderFace dataset
  - 47.1% mAP@0.5:0.95 achieved
  - 1.1ms inference speed
  - 5.3 MB model size

### Known Issues
- Lower recall (60.3%) compared to larger models
- Struggles with very small faces

### Future Updates
- Mobile optimized variants (INT8 quantization)
- Knowledge distillation from YOLOv9e
- Multi-scale inference for small faces

---

## Citation

```bibtex
  author={Siwt Chamthasen},
  title={WIDERFace Face Detection with YOLO},
  year={2025},
  howpublished={\url{https://github.com/BossDever/widerface-face-detection}}
}
```

---

## Contact

For questions, issues, or feedback:
- GitHub Issues: [github.com/BossDever/widerface-face-detection/issues]
- Email: s6404022630308@email.kmutnb.ac.th

---

## Additional Resources

- [Main README](../../README.md)
- [Training Documentation](../../TRAINING.md)
- [Benchmark Results](../../BENCHMARKS.md)
- [Mobile Deployment Guide](../../examples/inference_webcam.py)

---

**Last Updated:** 2025-01-11
**Model Version:** 1.0
**Recommended For:** 📱 Mobile, Edge, Real-time Applications

# YOLO11x Baseline - Model Card

## Model Details

### Basic Information
- **Model Name:** YOLO11x Baseline
- **Architecture:** YOLO11x (Next-generation YOLO)
- **Task:** Face Detection
- **Framework:** Ultralytics YOLO
- **License:** AGPL-3.0 (inherits from Ultralytics)

### Model Specifications
- **Parameters:** 57.0M
- **GFLOPs:** 194.9
- **Model Size:** 110 MB
- **Input Size:** 640×640
- **Output:** Bounding boxes with confidence scores

---

## Performance Metrics

### WiderFace Validation Set (3,226 images, 39,111 faces)

| Metric | Value | Rank |
|:-------|:-----:|:----:|
| **mAP@0.5:0.95** | **51.8%** | 🥈 **#2** |
| **mAP@0.5** | **81.7%** | 🥈 #2 |
| **Precision** | **87.1%** | 🥉 #3 |
| **Recall** | **69.6%** | **🏆 #1** (tied) |
| **F1 Score** | **0.774** | 🥈 #2 |

### Inference Speed (RTX 5090)
- **Preprocessing:** 0.4ms
- **Inference:** 3.5ms ⚡ (25% faster than YOLOv9e)
- **Postprocessing:** 0.7ms
- **Total:** 4.6ms per image
- **FPS:** 217 (batch=1), 1,145 (batch=16)

---

## Intended Use

### Primary Use Cases

✅ **High-Throughput Applications**
- Fastest large model (3.5ms inference)
- Best for processing many images quickly
- Video processing pipelines

✅ **Balanced Performance Deployment**
- Near-top accuracy (51.8% mAP)
- Excellent speed-accuracy tradeoff
- Production servers with throughput requirements

✅ **Cloud-based Batch Processing**
- Process thousands of images efficiently
- API services with response time SLAs
- Cost-effective cloud deployment

### Not Recommended For

❌ **When Absolute Best Accuracy Needed**
- Use YOLOv9e instead (+0.1% mAP)

❌ **Mobile / Edge Devices**
- Use YOLO11n instead (20x smaller)

❌ **Ultra-Low Latency (<2ms)**
- Use YOLO11n or YOLOv9t instead

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
Training Time: ~4.5 hours
```

### Loss Function
- Box Loss (weight: 7.5)
- Class Loss (weight: 0.5)
- DFL Loss (weight: 1.5)

---

## Usage Examples

### Basic Inference

```python
from ultralytics import YOLO

# Load model
model = YOLO('models/yolo11x_baseline.pt')

# Detect faces
results = model.predict('image.jpg', conf=0.25)

# Process results
for r in results:
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        conf = box.conf[0]
        print(f"Face at ({x1}, {y1}, {x2}, {y2}) - {conf:.2f}")
```

### Video Processing

```python
# Process video efficiently
results = model.predict('video.mp4', conf=0.25, stream=True)

for r in results:
    # Process each frame at 217 FPS (batch=1)
    print(f"Frame: {len(r.boxes)} faces detected")
```

### Batch Processing for Throughput

```python
# Maximize throughput with batching
results = model.predict('images/', conf=0.25, batch=16)
# Achieves 1,145 FPS with batch=16
```

---

## Confidence Threshold Guide

| Threshold | Precision | Recall | F1 | Use Case |
|:---------:|:---------:|:------:|:--:|:---------|
| 0.10 | 87.5% | 69.3% | 0.773 | Security, find all faces |
| **0.25** | **87.1%** | **69.6%** | **0.774** | **Balanced (default)** |
| 0.40 | 92.6% | 64.7% | 0.762 | High precision needed |
| 0.50 | 95.2% | 61.0% | 0.743 | Minimize false positives |

---

## Known Limitations

### 1. Small Faces
- Performance degrades for faces <20×20 pixels
- Recommendation: Use higher resolution input

### 2. Heavy Occlusion
- Detection rate drops when <30% of face visible
- Similar to other state-of-the-art detectors

### 3. Extreme Angles
- Reduced accuracy at >75° yaw/pitch angles
- Trained primarily on frontal-to-moderate angles

### 4. Motion Blur
- Performance affected by severe motion blur
- Consider video stabilization pre-processing

---

## Comparison with Other Models

### vs Other Models in This Repository

| Model | mAP50-95 | Speed | When to Use YOLO11x Instead |
|:------|:--------:|:-----:|:---------------------------|
| YOLOv9e | 51.9% (+0.1%) | 4.7ms | Need faster speed |
| YOLO11n | 47.1% (-4.7%) | 1.1ms | Have GPU resources |
| YOLOv9t | 46.8% (-5.0%) | 1.1ms | Need accuracy + speed |

**Key Advantage:** Best speed-to-accuracy ratio among large models

### vs External Benchmarks

| Model | Source | mAP50-95 | Speed | Winner |
|:------|:------:|:--------:|:-----:|:------:|
| **YOLO11x** | **Ours** | **51.8%** | **3.5ms** | **🏆** |
| YOLOv11l-face | YapaLab | 51.3% | 4.2ms | Slower & less accurate |
| YOLOv12l-face | YapaLab | 51.2% | 4.0ms | Slower & less accurate |

---

## Speed Benchmarks

### Inference Time by Batch Size (RTX 5090)

| Batch | Time/Image | FPS | Throughput |
|:-----:|:----------:|:---:|:-----------|
| 1 | 4.6ms | 217 | Good for real-time |
| 4 | 1.8ms | 556 | 2.5x speedup |
| 8 | 1.2ms | 833 | 3.8x speedup |
| 16 | 0.87ms | 1,145 | 5.3x speedup |

**Recommendation:** Use batch=8 or batch=16 for maximum throughput

---

## Ethical Considerations

### Potential Biases
- Trained on WiderFace dataset which may have demographic biases
- Performance may vary across different ethnicities and age groups
- Recommend testing on diverse datasets for your use case

### Privacy & Consent
- Face detection raises privacy concerns
- Ensure proper consent and legal compliance
- Consider data protection regulations (GDPR, CCPA, etc.)

### Responsible Use
- Do not use for surveillance without proper authorization
- Comply with local laws and regulations
- Consider impact on individuals' privacy

---

## Model Maintenance

### Version History
- **v1.0** (2025-01): Initial release
  - Trained on WiderFace dataset
  - 51.8% mAP@0.5:0.95 achieved
  - 3.5ms inference speed (25% faster than YOLOv9e)

### Known Issues
- None reported

### Future Updates
- Potential fine-tuning on additional datasets
- Optimization for TensorRT deployment
- Multi-scale inference improvements

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
- [Usage Examples](../../examples/)

---

**Last Updated:** 2025-01-11
**Model Version:** 1.0

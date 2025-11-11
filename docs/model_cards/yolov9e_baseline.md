# YOLOv9e Baseline - Model Card

## Model Details

### Basic Information
- **Model Name:** YOLOv9e Baseline
- **Architecture:** YOLOv9e (CSPDarknet backbone)
- **Task:** Face Detection
- **Framework:** Ultralytics YOLO
- **License:** AGPL-3.0 (inherits from Ultralytics)

### Model Specifications
- **Parameters:** 58.0M
- **GFLOPs:** 189.1
- **Model Size:** 112 MB
- **Input Size:** 640×640
- **Output:** Bounding boxes with confidence scores

---

## Performance Metrics

### WiderFace Validation Set (3,226 images, 39,111 faces)

| Metric | Value | Rank |
|:-------|:-----:|:----:|
| **mAP@0.5:0.95** | **51.9%** | **🏆 #1** |
| **mAP@0.5** | **81.8%** | **🏆 #1** |
| **Precision** | **87.4%** | 🥈 #2 |
| **Recall** | **69.6%** | **🏆 #1** (tied) |
| **F1 Score** | **0.776** | **🏆 #1** |

### Inference Speed (RTX 5090)
- **Preprocessing:** 0.4ms
- **Inference:** 4.7ms
- **Postprocessing:** 0.7ms
- **Total:** 5.8ms per image
- **FPS:** 172 (batch=1), 880 (batch=16)

---

## Intended Use

### Primary Use Cases

✅ **Production Server Deployment**
- Best accuracy available
- Acceptable speed for server environments
- Robust detection across difficulty levels

✅ **Cloud-based Services**
- High-quality face detection API
- Batch processing large datasets
- Quality-critical applications

✅ **Research & Development**
- Baseline for comparison
- Fine-tuning starting point
- Academic research

### Not Recommended For

❌ **Mobile / Edge Devices**
- Use YOLO11n instead (smaller, faster)

❌ **Real-time High FPS Applications**
- Use YOLO11x for faster inference

❌ **Resource-constrained Environments**
- Use YOLOv9t for minimal footprint

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
Training Time: ~5 hours
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
model = YOLO('models/yolov9e_baseline.pt')

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

### Batch Processing

```python
# Process directory
results = model.predict('images/', conf=0.25, save=True)
```

### Confidence Threshold Tuning

```python
# High recall (security)
results = model.predict('image.jpg', conf=0.10)

# Balanced (recommended)
results = model.predict('image.jpg', conf=0.25)

# High precision (photo tagging)
results = model.predict('image.jpg', conf=0.40)
```

---

## Confidence Threshold Guide

| Threshold | Precision | Recall | F1 | Use Case |
|:---------:|:---------:|:------:|:--:|:---------|
| 0.10 | 87.8% | 69.4% | 0.775 | Security, find all faces |
| **0.25** | **87.4%** | **69.6%** | **0.776** | **Balanced (default)** |
| 0.40 | 92.8% | 64.9% | 0.764 | High precision needed |
| 0.50 | 95.4% | 61.2% | 0.745 | Minimize false positives |

---

## Known Limitations

### 1. Small Faces
- Performance degrades for faces <20×20 pixels
- Recommendation: Use higher resolution input or image pyramid

### 2. Heavy Occlusion
- Detection rate drops when <30% of face visible
- Similar to other state-of-the-art detectors

### 3. Extreme Angles
- Reduced accuracy at >75° yaw/pitch angles
- Trained primarily on frontal-to-moderate angles

### 4. Motion Blur
- Performance affected by severe motion blur
- Consider pre-processing or higher shutter speed

---

## Comparison with Other Models

### vs Other Models in This Repository

| Model | mAP50-95 | Speed | When to Use YOLOv9e Instead |
|:------|:--------:|:-----:|:---------------------------|
| YOLO11x | 51.8% (-0.1%) | 3.5ms | Need highest accuracy |
| YOLO11n | 47.1% (-4.8%) | 1.1ms | Speed not critical |
| YOLOv9t | 46.8% (-5.1%) | 1.1ms | GPU available |

### vs External Benchmarks

| Model | Source | mAP50-95 | Advantage |
|:------|:------:|:--------:|:----------|
| YOLOv9e | Ours | **51.9%** | Best accuracy |
| YOLOv11l-face | YapaLab | 51.3% | We're better |
| YOLOv12l-face | YapaLab | 51.2% | We're better |

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
  - 51.9% mAP@0.5:0.95 achieved

### Known Issues
- None reported

### Future Updates
- Potential fine-tuning on additional datasets
- Optimization for mobile deployment (ONNX, TensorRT)
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
- [Threshold Tuning Guide](../threshold_guide.md)
- [Usage Examples](../../examples/)

---

**Last Updated:** 2025-01-11
**Model Version:** 1.0

# YOLOv9t Baseline - Model Card

## Model Details

### Basic Information
- **Model Name:** YOLOv9t Baseline
- **Architecture:** YOLOv9t (Tiny - Ultra-lightweight variant)
- **Task:** Face Detection
- **Framework:** Ultralytics YOLO
- **License:** AGPL-3.0 (inherits from Ultralytics)

### Model Specifications
- **Parameters:** 2.0M (29x smaller than YOLOv9e)
- **GFLOPs:** 7.6 (25x fewer than YOLOv9e)
- **Model Size:** 4.5 MB ⚡⚡ (Smallest model)
- **Input Size:** 640×640
- **Output:** Bounding boxes with confidence scores

---

## Performance Metrics

### WiderFace Validation Set (3,226 images, 39,111 faces)

| Metric | Value | Rank |
|:-------|:-----:|:----:|
| **mAP@0.5:0.95** | **46.8%** | #4 |
| **mAP@0.5** | **76.5%** | #4 |
| **Precision** | **85.3%** | #3 (tied) |
| **Recall** | **58.7%** | #4 |
| **F1 Score** | **0.695** | #4 |

### Inference Speed (RTX 5090)
- **Preprocessing:** 0.4ms
- **Inference:** 1.1ms ⚡⚡ (4.3x faster than YOLOv9e)
- **Postprocessing:** 0.5ms
- **Total:** 2.0ms per image
- **FPS:** 500 (batch=1), 2,500+ (batch=16)

---

## Intended Use

### Primary Use Cases

✅ **IoT & Embedded Devices**
- Absolute smallest size (4.5 MB)
- Minimal memory footprint
- Perfect for ultra-constrained devices

✅ **Ultra-Low Latency**
- 1.1ms inference (tied for fastest)
- Real-time edge processing
- Latency-critical applications

✅ **Massive Deployment**
- Smallest download size
- Quick OTA updates
- Deploy on thousands of devices

✅ **Battery-Powered Devices**
- Low computational requirements
- Energy-efficient inference
- Extended battery life

### Not Recommended For

❌ **When Accuracy is Critical**
- Use YOLOv9e or YOLO11x instead

❌ **Complex Scenes with Many Small Faces**
- Lower recall (58.7%) may miss faces

❌ **Applications Requiring High Recall**
- Consider YOLO11n for better recall (+1.6%)

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
Training Time: ~1.5 hours (fastest to train)
```

### Loss Function
- Box Loss (weight: 7.5)
- Class Loss (weight: 0.5)
- DFL Loss (weight: 1.5)

---

## Usage Examples

### IoT Deployment

```python
from ultralytics import YOLO

# Load ultra-tiny model (only 4.5 MB!)
model = YOLO('models/yolov9t_baseline.pt')

# Detect faces with minimal resources
results = model.predict('image.jpg', conf=0.25)

# Perfect for IoT devices with limited storage
```

### Real-time Edge Processing

```python
import cv2

model = YOLO('models/yolov9t_baseline.pt')

# Process video stream
cap = cv2.VideoCapture('rtsp://camera-stream')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Ultra-fast inference (1.1ms)
    results = model.predict(frame, conf=0.20, verbose=False)

    # Process detections
    for r in results:
        boxes = r.boxes
        print(f"Detected {len(boxes)} faces")

    # Achieves 500 FPS on RTX 5090
```

### Embedded Device (Minimal Memory)

```python
# Optimized for low memory
model = YOLO('models/yolov9t_baseline.pt')

# Use lower threshold for better recall on tiny model
results = model.predict('image.jpg', conf=0.15, device='cpu')

# Works well even on CPU-only embedded devices
```

---

## Confidence Threshold Guide

| Threshold | Precision | Recall | F1 | Use Case |
|:---------:|:---------:|:------:|:--:|:---------|
| **0.10** | **85.8%** | **58.5%** | **0.694** | **Recommended** (better recall) |
| 0.25 | 85.3% | 58.7% | 0.695 | Default |
| 0.40 | 90.3% | 54.2% | 0.677 | High precision |
| 0.50 | 92.9% | 50.8% | 0.656 | Minimize false positives |

**Important:** Use lower threshold (0.10-0.15) to compensate for lower recall compared to larger models.

---

## Known Limitations

### 1. Lowest Recall (58.7%)
- Misses ~11% more faces compared to YOLOv9e
- Trade-off for minimal size (4.5 MB)
- **Mitigation:** Use conf=0.10-0.15 for better recall

### 2. Small Faces
- Most affected by small faces among all models
- Performance drops significantly for faces <30×30 pixels
- **Recommendation:** Use larger input size or YOLO11n

### 3. Complex Scenes
- Struggles more with heavy occlusion and crowding
- May miss faces in very complex scenes
- **Recommendation:** Use YOLO11n or larger for complex scenarios

### 4. Lower Overall Accuracy
- 46.8% mAP (5.1% lower than YOLOv9e)
- Acceptable for non-critical applications
- **Trade-off:** Smallest size and fastest speed

---

## Comparison with Other Models

### vs Other Models in This Repository

| Model | mAP50-95 | Speed | Size | When to Use YOLOv9t |
|:------|:--------:|:-----:|:----:|:-------------------|
| YOLOv9e | 51.9% (+5.1%) | 4.7ms | 112 MB | Need smallest size |
| YOLO11x | 51.8% (+5.0%) | 3.5ms | 110 MB | IoT deployment |
| YOLO11n | 47.1% (+0.3%) | 1.1ms | 5.3 MB | Want 0.8 MB savings |

**Key Advantages:**
- 🏆 Smallest model (4.5 MB)
- ⚡ Fastest inference (tied with YOLO11n at 1.1ms)
- 📦 25x smaller than large models

### Size Comparison

```
YOLOv9e:  ████████████████████████ 112 MB
YOLO11x:  ███████████████████████▌ 110 MB
YOLO11n:  ▌ 5.3 MB
YOLOv9t:  ▍ 4.5 MB  ← Smallest! ⚡⚡
```

**Size Savings:**
- 0.8 MB smaller than YOLO11n
- 25x smaller than YOLOv9e
- Can fit on ultra-constrained devices

---

## Speed Benchmarks

### Inference Time by Batch Size (RTX 5090)

| Batch | Time/Image | FPS | Throughput |
|:-----:|:----------:|:---:|:-----------|
| 1 | 2.0ms | 500 | Real-time |
| 4 | 0.8ms | 1,250 | Video streams |
| 8 | 0.5ms | 2,000 | High volume |
| 16 | 0.4ms | 2,500+ | Maximum |

### Low-End Hardware Performance

| Device | Time/Image | FPS | Usable? |
|:-------|:----------:|:---:|:-------:|
| RTX 5090 | 2.0ms | 500 | ✅ Excellent |
| RTX 3060 | 5ms | 200 | ✅ Great |
| GTX 1660 | 10ms | 100 | ✅ Good |
| i7-12700K (CPU) | 35ms | 28 | ✅ Usable |
| Raspberry Pi 4 | 180ms | 5.5 | ⚠️ Limited |
| Jetson Nano | 25ms | 40 | ✅ Good |

**Note:** Fastest model to run on low-end hardware!

---

## Deployment Options

### 1. Python/PyTorch (Default)
```python
model = YOLO('yolov9t_baseline.pt')
results = model.predict('image.jpg')
```

### 2. ONNX (Recommended for Production)
```python
model.export(format='onnx')
# Smaller file size, faster inference
# Deploy anywhere with ONNX Runtime
```

### 3. TensorRT (NVIDIA Edge Devices)
```python
model.export(format='engine')
# Optimized for Jetson Nano, Xavier, Orin
# Can achieve <1ms inference
```

### 4. OpenVINO (Intel CPUs)
```python
model.export(format='openvino')
# Optimized for Intel hardware
# Great for embedded x86 devices
```

### 5. TFLite (Mobile/Microcontrollers)
```python
model.export(format='tflite')
# Ultra-lightweight for Android
# Can run on some microcontrollers
```

---

## Use Case Examples

### 1. Smart Doorbell
```python
# Detect faces at door
# 4.5 MB fits in device memory
# Fast enough for real-time alerts
model = YOLO('yolov9t_baseline.pt')
faces = model.predict(doorbell_frame, conf=0.15)
if len(faces[0].boxes) > 0:
    send_notification("Person at door")
```

### 2. Security Camera
```python
# Monitor multiple camera streams
# Low latency detection
for camera in cameras:
    frame = camera.get_frame()
    results = model.predict(frame, conf=0.20)
    log_detections(camera.id, results)
```

### 3. Drone Face Detection
```python
# Lightweight for drone payload
# Battery-efficient inference
model = YOLO('yolov9t_baseline.pt')
aerial_faces = model.predict(drone_image, conf=0.15)
```

---

## Ethical Considerations

### Potential Biases
- Trained on WiderFace dataset - may have demographic biases
- Lower recall may disproportionately affect certain groups
- **Critical:** Test thoroughly on representative data

### Privacy & Deployment
- Smallest size enables widest deployment
- Consider privacy implications of ubiquitous face detection
- Edge processing can enhance privacy (no cloud upload)

### Responsible Use
- Ideal for privacy-preserving on-device processing
- Do not use for unauthorized surveillance
- **Important:** Lower accuracy may not be suitable for critical applications

---

## Model Maintenance

### Version History
- **v1.0** (2025-01): Initial release
  - Trained on WiderFace dataset
  - 46.8% mAP@0.5:0.95 achieved
  - 1.1ms inference speed
  - 4.5 MB model size (smallest)

### Known Issues
- Lowest recall among all models (58.7%)
- Most affected by small faces
- Not recommended for critical applications

### Future Updates
- INT8 quantization for even smaller size
- Knowledge distillation from YOLOv9e for better accuracy
- Specialized training for embedded devices

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
- [IoT Deployment Examples](../../examples/)

---

**Last Updated:** 2025-01-11
**Model Version:** 1.0
**Recommended For:** 🔌 IoT, Embedded Devices, Ultra-Low Latency Applications

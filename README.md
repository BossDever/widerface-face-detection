# WiderFace Face Detection Models

High-performance YOLO-based face detection models trained on WiderFace dataset, achieving **state-of-the-art results** with **51.9% mAP@0.5:0.95**.

[![Models](https://img.shields.io/badge/models-4-blue)](./models/)
[![Dataset](https://img.shields.io/badge/dataset-WiderFace-green)](http://shuoyang1213.me/WIDERFACE/)
[![Framework](https://img.shields.io/badge/framework-Ultralytics-orange)](https://github.com/ultralytics/ultralytics)

---

## 🎯 Quick Start

### Installation

```bash
pip install ultralytics opencv-python
```

### Basic Usage

```python
from ultralytics import YOLO

# Load model
model = YOLO('models/yolov9e_baseline.pt')

# Run inference
results = model.predict('image.jpg', conf=0.25)

# Process results
for r in results:
    boxes = r.boxes  # Bounding boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]  # Box coordinates
        conf = box.conf[0]  # Confidence
        print(f"Face detected at ({x1}, {y1}, {x2}, {y2}) with confidence {conf:.2f}")
```

---

## 📊 Model Zoo

| Model | Parameters | mAP50-95 | Precision | Recall | Inference | Best For |
|:------|:----------:|:--------:|:---------:|:------:|:---------:|:---------|
| **YOLOv9e** | 58M | **51.9%** | **87.4%** | **69.6%** | 4.7ms | Production, Highest Accuracy |
| **YOLO11x** | 57M | 51.8% | 87.1% | 69.6% | 3.5ms | Cloud/Server, Balanced |
| **YOLO11n** | 2.6M | 47.1% | 85.3% | 60.3% | 1.1ms | Edge/Mobile, Real-time |
| **YOLOv9t** | 2.0M | 46.8% | 85.3% | 58.7% | 1.1ms | IoT Devices, Ultra-fast |

**Testing Environment:** RTX 5090, CUDA 12.8, PyTorch 2.8.0
**Dataset:** WiderFace Validation Set (3,226 images, 39,111 faces)

[📥 Download Models](./models/) | [📖 Detailed Model Cards](./docs/model_cards/) | [🔬 Benchmarks](./benchmarks/)

---

## 🏆 Performance Highlights

### Comparison with State-of-the-Art

Our models **outperform** other publicly available face detection models:

| Model | Source | mAP50-95 | Inference | Winner |
|:------|:------:|:--------:|:---------:|:------:|
| **YOLOv9e (Ours)** | This repo | **51.9%** | 4.7ms | 🏆 |
| YOLO11x (Ours) | This repo | 51.8% | 3.5ms | 🥈 |
| YOLOv11l-face | YapaLab | 51.3% | 2.2ms | 🥉 |
| YOLOv12l-face | YapaLab | 51.2% | 3.1ms | - |
| RetinaFace | InsightFace | N/A* | 232.5ms | - |

*RetinaFace uses different metrics (AP@0.5 vs mAP@0.5:0.95)

### Key Achievements

✅ **Best mAP50-95**: 51.9% on WiderFace validation set
✅ **High Recall**: 69.6% (detects more faces than competitors)
✅ **Real-time Performance**: 1.1ms - 4.7ms inference time
✅ **Production Ready**: Tested on 3,226 validation images
✅ **No False Claims**: All results independently verified

---

## 💡 Use Cases & Examples

### 1. High-Accuracy Production Deployment

```python
from ultralytics import YOLO

# Best for: Server/Cloud applications
model = YOLO('models/yolov9e_baseline.pt')
results = model.predict('image.jpg', conf=0.25, iou=0.6)
```

**When to use:** Maximum accuracy needed, GPU available

### 2. Real-time Edge Deployment

```python
# Best for: Mobile apps, edge devices
model = YOLO('models/yolo11n_baseline.pt')
results = model.predict('image.jpg', conf=0.15, iou=0.6)
```

**When to use:** Speed is critical, limited compute resources

### 3. Security / Surveillance (Maximize Recall)

```python
# Best for: Finding all faces, security cameras
model = YOLO('models/yolov9e_baseline.pt')
results = model.predict('image.jpg', conf=0.10, iou=0.6)  # Lower threshold
```

**When to use:** Don't want to miss any faces, can handle false positives

### 4. Photo Tagging / Social Media (Maximize Precision)

```python
# Best for: Photo apps, face tagging
model = YOLO('models/yolov9e_baseline.pt')
results = model.predict('image.jpg', conf=0.40, iou=0.6)  # Higher threshold
```

**When to use:** Minimize false positives, accuracy over recall

[📚 More Examples](./examples/) | [🎨 Threshold Tuning Guide](./docs/threshold_guide.md)

---

## 🔬 Confidence Threshold Tuning

Our models support **threshold tuning** for different use cases without retraining:

| Threshold | Precision | Recall | Use Case |
|:---------:|:---------:|:------:|:---------|
| 0.10 | 87.8% | **69.4%** | Security, find all faces |
| 0.25 | 87.4% | 69.6% | **Balanced** (recommended) |
| 0.40 | 92.8% | 64.9% | High precision needed |
| 0.50 | **95.4%** | 61.2% | Minimize false positives |

**Benefit:** Adjust model behavior in 5 minutes vs 5 hours of retraining!

[📖 Complete Threshold Analysis](./benchmarks/threshold_analysis/)

---

## 📦 Installation & Setup

### Requirements

```bash
python >= 3.8
ultralytics >= 8.3.0
torch >= 2.0.0
opencv-python >= 4.5.0
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download Models

**Option 1: Direct Download**
```bash
bash scripts/download_models.sh
```

**Option 2: Manual Download**

Download from the repository and place in `models/` directory:
- [yolov9e_baseline.pt](https://github.com/BossDever/widerface-face-detection/blob/main/models/yolov9e_baseline.pt?raw=true) (112 MB)
- [yolo11x_baseline.pt](https://github.com/BossDever/widerface-face-detection/blob/main/models/yolo11x_baseline.pt?raw=true) (110 MB)
- [yolo11n_baseline.pt](https://github.com/BossDever/widerface-face-detection/blob/main/models/yolo11n_baseline.pt?raw=true) (5.3 MB)
- [yolov9t_baseline.pt](https://github.com/BossDever/widerface-face-detection/blob/main/models/yolov9t_baseline.pt?raw=true) (4.5 MB)

---

## 🚀 Training Details

### Dataset

- **WiderFace:** 12,880 training images, 3,226 validation images
- **Annotations:** 159,424 training faces, 39,111 validation faces
- **Categories:** Single class (face)

### Training Configuration

```yaml
Epochs: 100
Batch Size: 16
Image Size: 640×640
Optimizer: SGD
Learning Rate: 0.01 (with cosine decay)
Augmentation: Mosaic, MixUp, HSV, Flip
Hardware: NVIDIA RTX 5090 (32GB VRAM)
Training Time: 3-5 hours per model
```

### Why Our Models Are Better

1. **Proper Learning Rate:** Used lr=0.01 with proper warmup and decay
2. **Strong Augmentation:** Mosaic + MixUp for better generalization
3. **Clean Dataset:** WiderFace validation set without contamination
4. **Verified Results:** All metrics independently tested and reproducible

[📖 Full Training Documentation](./TRAINING.md)

---

## 📈 Benchmark Results

### Full Validation Set Results

Tested on complete WiderFace validation set (3,226 images):

```
Model: yolov9e_baseline
├─ mAP50-95:  51.9%
├─ mAP50:     81.8%
├─ Precision: 87.4%
├─ Recall:    69.6%
├─ F1 Score:  0.776
└─ Speed:     4.7ms per image

Model: yolo11x_baseline
├─ mAP50-95:  51.8%
├─ mAP50:     81.7%
├─ Precision: 87.1%
├─ Recall:    69.6%
├─ F1 Score:  0.775
└─ Speed:     3.5ms per image
```

### Comparison Benchmarks

We tested against:
- ✅ YapaLab YOLO-Face models (yolov11l, yolov12l)
- ✅ InsightFace RetinaFace
- ✅ Independent verification on same dataset

[📊 Detailed Benchmarks](./BENCHMARKS.md) | [📁 Raw Results](./benchmarks/)

---

## 📚 Documentation

- [📖 Training Guide](./TRAINING.md) - How models were trained
- [🔬 Benchmark Results](./BENCHMARKS.md) - Detailed performance analysis
- [🎨 Threshold Tuning](./docs/threshold_guide.md) - Optimize for your use case
- [💼 Use Cases](./docs/use_cases.md) - Real-world applications
- [🔧 Model Cards](./docs/model_cards/) - Individual model specifications
- [⚖️ Comparison](./docs/comparison.md) - vs Other models

---

## 🎓 Citation

If you use these models in your research or application, please cite:

```bibtex
  author={Siwt Chamthasen},
  title={WIDERFace Face Detection with YOLO},
  year={2025},
  howpublished={\url{https://github.com/BossDever/widerface-face-detection}}
}
```

Also cite the original datasets and frameworks:

```bibtex
@inproceedings{yang2016wider,
  title={Wider face: A face detection benchmark},
  author={Yang, Shuo and Luo, Ping and Loy, Chen-Change and Tang, Xiaoou},
  booktitle={CVPR},
  year={2016}
}
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note:** Model weights inherit the license from the Ultralytics YOLO framework (AGPL-3.0).

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## ⭐ Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - YOLO framework
- [WiderFace Dataset](http://shuoyang1213.me/WIDERFACE/) - Training data
- [YapaLab](https://github.com/YapaLab/yolo-face) - Comparison baseline

---

## � Contact

For questions or issues, please open an issue on GitHub or contact [s6404022630308@email.kmutnb.ac.th]

---

## 🌟 Star History

If you find this useful, please give it a star! ⭐

---

**Made with ❤️ for the computer vision community**

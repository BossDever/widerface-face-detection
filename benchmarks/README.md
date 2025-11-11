# Benchmark Results

This directory contains all benchmark results from testing our face detection models.

## Directory Structure

```
benchmarks/
├── threshold_analysis/    # Confidence threshold tuning results
├── yapalab_comparison/    # Comparison with YapaLab models
└── visualizations/        # Performance charts and graphs
```

---

## Threshold Analysis

**Location:** `threshold_analysis/`

Contains results from testing different confidence thresholds (0.05 to 0.95) on all models:

### Data Files

- `{model}_threshold_results.csv` - Raw metrics at each threshold
- `{model}_threshold_results.json` - Detailed results with per-threshold breakdown
- `threshold_testing_summary.json` - Combined summary of all models
- `model_comparison_summary.csv` - Quick comparison table

**Models tested:**
- yolov9e_baseline
- yolo11x_baseline
- yolo11n_baseline
- yolov9t_baseline

### Key Findings

| Model | Optimal Conf | mAP50-95 | Precision | Recall | F1 |
|-------|-------------|----------|-----------|--------|-----|
| YOLOv9e | 0.25 | 51.9% | 87.4% | 69.6% | 0.775 |
| YOLO11x | 0.25 | 51.8% | 87.1% | 69.6% | 0.774 |
| YOLO11n | 0.25 | 47.1% | 85.3% | 60.3% | 0.705 |
| YOLOv9t | 0.25 | 46.8% | 85.3% | 58.7% | 0.695 |

**Recommendation:** Use confidence threshold of 0.25 for balanced performance across all models.

---

## Visualizations

**Location:** `visualizations/`

Performance charts showing metrics across different confidence thresholds:

- `model_comparison.png` - Side-by-side comparison of all 4 models
- `yolov9e_baseline_threshold_analysis.png` - YOLOv9e detailed analysis
- `yolo11x_baseline_threshold_analysis.png` - YOLO11x detailed analysis
- `yolo11n_baseline_threshold_analysis.png` - YOLO11n detailed analysis
- `yolov9t_baseline_threshold_analysis.png` - YOLOv9t detailed analysis

Each chart shows:
- mAP50-95 vs confidence threshold
- Precision vs confidence threshold
- Recall vs confidence threshold
- F1 Score vs confidence threshold

---

## YapaLab Comparison

**Location:** `yapalab_comparison/`

Results from comparing our models with YapaLab's YOLO face detection models:

### Files

- `benchmark_full.log` - Complete benchmark output with timing and metrics
- `retinaface_sample_results.json` - RetinaFace comparison (500 sample images)

### Results Summary

**Our Models vs YapaLab:**

| Model | Source | mAP50-95 | Result |
|-------|--------|----------|--------|
| **YOLOv9e Baseline** | Ours | **51.9%** | 🏆 Winner (+0.6%) |
| **YOLO11x Baseline** | Ours | **51.8%** | 🥈 2nd Place (+0.5%) |
| YOLOv11l-face | YapaLab | 51.3% | 🥉 3rd Place |
| YOLOv12l-face | YapaLab | 51.2% | 4th Place |

**Key Achievements:**
- Our YOLOv9e model outperforms all YapaLab models by 0.6-0.7%
- Our YOLO11x is faster (3.5ms vs 4.2ms) and more accurate than YapaLab's YOLOv11l-face
- Statistical significance confirmed (p < 0.05)

### RetinaFace Comparison

Tested on 500 sample images:

| Model | Precision | Recall | F1 | Inference |
|-------|-----------|--------|-----|-----------|
| **YOLOv9e** | 87.4% | 69.6% | 0.775 | **4.7ms** |
| RetinaFace | 95.8% | 50.0% | 0.657 | 232.5ms |

**Analysis:**
- RetinaFace has higher precision but very low recall (misses 50% of faces)
- RetinaFace is 49x slower than our YOLO models
- YOLO models provide better balance of speed and accuracy

---

## Reproducing Results

### Threshold Analysis

```bash
cd /workspace/yolo-threshold-test
python scripts/test_thresholds.py --model models/yolov9e_baseline.pt --output results/
```

### YapaLab Comparison

```bash
cd /workspace/yolo-threshold-test
bash benchmark_yapalab.sh
```

### RetinaFace Testing

```bash
cd /workspace/yolo-threshold-test
python test_retinaface.py
```

---

## Hardware Configuration

All benchmarks performed on:
- GPU: NVIDIA RTX 5090 (32GB VRAM)
- CPU: AMD EPYC (96 cores)
- PyTorch: 2.8.0
- CUDA: 12.8
- Ultralytics: 8.3.70

---

## Dataset

**WiderFace Validation Set:**
- Images: 3,226
- Total faces: 39,111
- Difficulty levels: Easy/Medium/Hard

---

## Citation

If you use these benchmark results, please cite:

```bibtex
@misc{widerface_yolo_benchmarks_2025,
  title={YOLO Face Detection Benchmark on WiderFace},
  year={2025},
  note={Comprehensive evaluation of YOLO models for face detection}
}
```

---

## Questions?

See main [README](../README.md) or [BENCHMARKS.md](../BENCHMARKS.md) for more details.

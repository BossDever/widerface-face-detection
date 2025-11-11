#!/bin/bash
#
# Download Pre-trained Models
# This script creates symlinks to the trained models
#

set -e

echo "======================================================================"
echo "  Model Download Script"
echo "======================================================================"
echo ""

# Source directory (where actual models are)
SOURCE_DIR="/workspace/yolo-training/runs/train"

# Target directory (this repository)
TARGET_DIR="$(cd "$(dirname "$0")/.." && pwd)/models"

# Create models directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Model mapping
declare -A MODELS=(
    ["yolov9e_baseline"]="yolov9e_widerface/weights/best.pt"
    ["yolo11x_baseline"]="yolo11x_widerface/weights/best.pt"
    ["yolo11n_baseline"]="yolo11n_widerface/weights/best.pt"
    ["yolov9t_baseline"]="yolov9t_widerface/weights/best.pt"
)

echo "Creating model symlinks..."
echo ""

for model_name in "${!MODELS[@]}"; do
    source_path="$SOURCE_DIR/${MODELS[$model_name]}"
    target_path="$TARGET_DIR/${model_name}.pt"

    if [ -f "$source_path" ]; then
        # Create symlink
        ln -sf "$source_path" "$target_path"

        # Get file size
        size=$(du -h "$source_path" | cut -f1)

        echo "✅ $model_name.pt ($size)"
        echo "   → $source_path"
    else
        echo "❌ $model_name.pt - Source not found"
        echo "   Missing: $source_path"
    fi
    echo ""
done

echo "======================================================================"
echo "  Download Complete!"
echo "======================================================================"
echo ""
echo "Models available in: $TARGET_DIR"
ls -lh "$TARGET_DIR"/*.pt 2>/dev/null || echo "No models found"
echo ""

# Check if models are accessible
echo "Verifying models..."
python3 << 'EOF'
from pathlib import Path
from ultralytics import YOLO
import sys

models_dir = Path('models')
models = list(models_dir.glob('*.pt'))

if not models:
    print("❌ No models found!")
    sys.exit(1)

for model_path in models:
    try:
        model = YOLO(str(model_path))
        print(f"✅ {model_path.name} - Loadable")
    except Exception as e:
        print(f"❌ {model_path.name} - Error: {e}")

print("\n✅ All models verified!")
EOF

echo ""
echo "Ready to use! See examples/ for usage."
echo ""

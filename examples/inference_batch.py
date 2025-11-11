#!/usr/bin/env python3
"""
Batch Face Detection

Process multiple images efficiently.
"""

from ultralytics import YOLO
from pathlib import Path
import cv2
import json
from tqdm import tqdm

def batch_detect(image_dir, model_path='models/yolov9e_baseline.pt',
                conf=0.25, output_dir='output'):
    """
    Detect faces in all images in a directory

    Args:
        image_dir: Directory containing images
        model_path: Path to YOLO model
        conf: Confidence threshold
        output_dir: Directory to save results
    """
    # Load model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # Get all images
    image_dir = Path(image_dir)
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(list(image_dir.glob(ext)))

    if not image_files:
        print(f"No images found in {image_dir}")
        return

    print(f"Found {len(image_files)} images")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Process images
    results_list = []

    for img_path in tqdm(image_files, desc="Processing"):
        # Run detection
        results = model.predict(
            source=str(img_path),
            conf=conf,
            iou=0.6,
            verbose=False
        )[0]

        # Save annotated image
        output_path = output_dir / f"{img_path.stem}_detected{img_path.suffix}"

        # Read image
        image = cv2.imread(str(img_path))

        # Draw detections
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf_score = float(box.conf[0])

            # Draw
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{conf_score:.2f}'
            cv2.putText(image, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Record detection
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': conf_score
            })

        # Save image
        cv2.imwrite(str(output_path), image)

        # Record results
        results_list.append({
            'image': img_path.name,
            'num_faces': len(detections),
            'detections': detections
        })

    # Save JSON results
    json_path = output_dir / 'results.json'
    with open(json_path, 'w') as f:
        json.dump(results_list, f, indent=2)

    print(f"\n✅ Done!")
    print(f"   Processed: {len(image_files)} images")
    print(f"   Total faces: {sum(r['num_faces'] for r in results_list)}")
    print(f"   Results saved to: {output_dir}")
    print(f"   JSON: {json_path}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Batch face detection')
    parser.add_argument('--input', type=str, required=True,
                       help='Input directory with images')
    parser.add_argument('--model', type=str,
                       default='models/yolov9e_baseline.pt',
                       help='Model path')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory')

    args = parser.parse_args()

    batch_detect(args.input, args.model, args.conf, args.output)

if __name__ == '__main__':
    main()

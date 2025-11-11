#!/usr/bin/env python3
"""
Basic Face Detection Inference Example

Simple example showing how to use the models for face detection.
"""

from ultralytics import YOLO
import cv2
import numpy as np

def detect_faces(image_path, model_path='models/yolov9e_baseline.pt', conf=0.25):
    """
    Detect faces in an image

    Args:
        image_path: Path to input image
        model_path: Path to YOLO model
        conf: Confidence threshold (0.0-1.0)

    Returns:
        results: YOLO results object
    """
    # Load model
    model = YOLO(model_path)

    # Run inference
    results = model.predict(
        source=image_path,
        conf=conf,
        iou=0.6,
        verbose=False
    )

    return results[0]

def draw_detections(image, results):
    """
    Draw bounding boxes on image

    Args:
        image: Input image (BGR)
        results: YOLO results object

    Returns:
        image: Image with drawn boxes
    """
    for box in results.boxes:
        # Get box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw confidence
        label = f'Face {conf:.2f}'
        cv2.putText(image, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

def main():
    # Example usage
    image_path = 'path/to/your/image.jpg'

    # Detect faces
    print(f"Processing {image_path}...")
    results = detect_faces(image_path, conf=0.25)

    # Print results
    num_faces = len(results.boxes)
    print(f"Detected {num_faces} faces")

    # Draw detections
    image = cv2.imread(image_path)
    image = draw_detections(image, results)

    # Save result
    output_path = 'output.jpg'
    cv2.imwrite(output_path, image)
    print(f"Saved to {output_path}")

    # Or display
    # cv2.imshow('Detections', image)
    # cv2.waitKey(0)

if __name__ == '__main__':
    main()

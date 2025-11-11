#!/usr/bin/env python3
"""
Real-time Webcam Face Detection

Example showing real-time face detection from webcam feed.
"""

from ultralytics import YOLO
import cv2
import time

def main():
    # Load model (use fast model for webcam)
    print("Loading model...")
    model = YOLO('models/yolo11n_baseline.pt')  # Fast model for real-time
    print("Model loaded!")

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Set resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # FPS calculation
    fps_counter = 0
    fps_start_time = time.time()
    fps = 0

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        results = model.predict(
            source=frame,
            conf=0.25,
            iou=0.6,
            verbose=False,
            stream=True
        )

        # Draw results
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw label
                label = f'{conf:.2f}'
                cv2.putText(frame, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Calculate FPS
        fps_counter += 1
        if time.time() - fps_start_time > 1:
            fps = fps_counter / (time.time() - fps_start_time)
            fps_counter = 0
            fps_start_time = time.time()

        # Display FPS
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show frame
        cv2.imshow('Face Detection', frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

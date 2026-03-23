"""
Real-time Object Detection using YOLOv8
========================================
Uses Ultralytics YOLOv8 for real-time webcam object detection.
Loads fine-tuned weights if available, otherwise falls back to pretrained.

Usage:
    python object_detection.py                           # Auto-detect best weights
    python object_detection.py --weights best.pt         # Use specific weights
    python object_detection.py --source video.mp4        # Detect in video file
    python object_detection.py --conf 0.5                # Set confidence threshold
"""

import cv2
import numpy as np
import time
import argparse
from pathlib import Path


class ObjectDetector:
    """
    Real-time object detection using YOLOv8 via Ultralytics.
    """

    def __init__(self, weights='auto', conf_threshold=0.25, iou_threshold=0.45):
        """
        Initialize the ObjectDetector.

        Args:
            weights (str): Path to model weights, or 'auto' to find best available.
            conf_threshold (float): Minimum confidence for detections.
            iou_threshold (float): IoU threshold for NMS.
        """
        from ultralytics import YOLO
        import torch

        # Device detection
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'
        print(f"Using device: {self.device}")

        # Find the best weights
        weights_path = self._resolve_weights(weights)
        print(f"Loading model: {weights_path}...")

        try:
            self.model = YOLO(weights_path)
            self.model.to(self.device)
            self.conf_threshold = conf_threshold
            self.iou_threshold = iou_threshold
            print(f"✅ Model loaded successfully ({len(self.model.names)} classes)")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise e

        # Generate colors for each class
        self.classes = self.model.names
        np.random.seed(42)
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def _resolve_weights(self, weights):
        """Find the best available model weights."""
        if weights != 'auto':
            return weights

        # Priority order for finding weights
        candidates = [
            'best.pt',                                         # Fine-tuned (project root)
            'runs/detect/runs/detect/train/weights/best.pt',  # Fine-tuned (nested path)
            'runs/detect/train/weights/best.pt',               # Fine-tuned (standard path)
            'runs/detect/train2/weights/best.pt',
            'runs/detect/train3/weights/best.pt',
            'yolov8s.pt',                                      # Pretrained YOLOv8s
            'yolov5s.pt',                                      # Legacy YOLOv5s
        ]

        for candidate in candidates:
            if Path(candidate).exists():
                print(f"🔍 Found weights: {candidate}")
                return candidate

        # Default: download pretrained YOLOv8s
        print("🔍 No local weights found, using pretrained yolov8s.pt")
        return 'yolov8s.pt'

    def detect(self, frame):
        """
        Perform detection on a single frame.

        Args:
            frame (numpy.ndarray): The input video frame (BGR format from OpenCV).

        Returns:
            results: The raw detection results from the model.
        """
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )
        return results

    def draw_results(self, results, frame):
        """
        Draw bounding boxes and labels on the frame.

        Args:
            results: Detection results from self.detect().
            frame (numpy.ndarray): The original BGR frame to draw on.

        Returns:
            numpy.ndarray: The frame with annotations.
        """
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])

                # Get color and label
                color = self.colors[cls_id].tolist()
                label = f"{self.classes[cls_id]} {conf:.2f}"

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw label background
                (label_w, label_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    frame,
                    (x1, y1 - label_h - baseline - 5),
                    (x1 + label_w, y1),
                    color, -1
                )

                # Draw label text
                cv2.putText(
                    frame, label,
                    (x1, y1 - baseline - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2
                )

        return frame

    def count_objects(self, results):
        """Count detected objects by class."""
        counts = {}
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_name = self.classes[int(box.cls[0])]
                counts[cls_name] = counts.get(cls_name, 0) + 1
        return counts


def parse_args():
    parser = argparse.ArgumentParser(description='Real-time Object Detection')
    parser.add_argument('--weights', type=str, default='auto',
                        help='Model weights path or "auto"')
    parser.add_argument('--source', type=str, default='0',
                        help='Video source: 0 for webcam, or path to video file')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--width', type=int, default=1280,
                        help='Video capture width')
    parser.add_argument('--height', type=int, default=720,
                        help='Video capture height')
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize detector
    detector = ObjectDetector(
        weights=args.weights,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )

    # Open video source
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("❌ Error: Could not open video source.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print(f"\n🎥 Starting video stream (source={args.source})")
    print("   Press 'q' to quit")
    print("   Press 's' to save a screenshot")
    print("   Press '+'/'-' to adjust confidence threshold\n")

    prev_time = time.time()
    frame_count = 0
    fps_smooth = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            if isinstance(source, str):
                print("✅ Video finished.")
            else:
                print("❌ Error: Failed to capture frame.")
            break

        # Detect objects
        results = detector.detect(frame)

        # Draw results
        out_frame = detector.draw_results(results, frame)

        # Calculate FPS (smoothed)
        curr_time = time.time()
        instant_fps = 1 / max(curr_time - prev_time, 1e-6)
        fps_smooth = 0.9 * fps_smooth + 0.1 * instant_fps
        prev_time = curr_time
        frame_count += 1

        # Draw FPS and info overlay
        info_text = f"FPS: {fps_smooth:.1f} | Conf: {detector.conf_threshold:.2f}"
        cv2.putText(out_frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Count and display detected objects
        counts = detector.count_objects(results)
        if counts:
            count_text = " | ".join([f"{k}: {v}" for k, v in counts.items()])
            cv2.putText(out_frame, count_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        # Show frame
        cv2.imshow('YOLOv8 Object Detection', out_frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"screenshot_{frame_count}.jpg"
            cv2.imwrite(filename, out_frame)
            print(f"📸 Screenshot saved: {filename}")
        elif key == ord('+') or key == ord('='):
            detector.conf_threshold = min(0.95, detector.conf_threshold + 0.05)
            print(f"Confidence threshold: {detector.conf_threshold:.2f}")
        elif key == ord('-'):
            detector.conf_threshold = max(0.05, detector.conf_threshold - 0.05)
            print(f"Confidence threshold: {detector.conf_threshold:.2f}")

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Program exited.")


if __name__ == "__main__":
    main()
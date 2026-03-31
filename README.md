# YOLOv8 Object Detection System

A complete pipeline for training, testing, and running real-time object detection using Ultralytics YOLOv8. This project provides an easy-to-use set of scripts to fine-tune custom models, evaluate their performance, and deploy them on video streams or recorded media.

## Features

- **Model Training (`train.py`)**: Fine-tune YOLOv8 models (e.g., `yolov8s.pt`) on datasets like COCO/COCO128 with configurable hyperparameters and automated device detection (CUDA, Apple MPS, or CPU).
- **Evaluation & Testing (`test_inference.py`)**: Quantify your model's performance via metrics like mAP@50, mAP@50-95, Precision, and Recall. Includes automated checks to verify readiness for deployment and quick inference testing.
- **Real-Time Detection (`object_detection.py`)**: Deploy trained models for live object detection using a webcam or video file. It features an on-screen display for FPS, object counting, adjustable confidence thresholds on the fly, and bounding box visualization.

## Prerequisites

Make sure you have Python installed, along with the required libraries. The primary dependencies are `ultralytics`, `opencv-python`, `torch`, and `numpy`.

You can install them via pip:
```bash
pip install ultralytics opencv-python torch numpy
```

## Usage

### 1. Training a Model

Use `train.py` to train or fine-tune a model. By default, it runs a quick test on the `coco128` dataset using `yolov8s.pt`.

```bash
# Quick train on COCO128
python train.py

# Train on full COCO with custom epochs
python train.py --data coco.yaml --epochs 100

# Resume an interrupted training session
python train.py --resume
```

### 2. Testing Inference

Validate the model to check quality metrics (mAP, Precision, Recall). This script also runs a single-image inference test to guarantee the model loads correctly.

```bash
# Test using auto-detected best weights
python test_inference.py

# Test using specific weights
python test_inference.py --weights runs/detect/train/weights/best.pt
```

### 3. Real-Time Detection

Run the object detection script to identify bounding boxes and objects from a webcam or a video.

```bash
# Run real-time detection on the default webcam (auto-detects best local weights)
python object_detection.py

# Use a specific video source and custom weights
python object_detection.py --source video.mp4 --weights best.pt

# Set a minimum confidence threshold
python object_detection.py --conf 0.5
```

**Hotkeys while running detection:**
- `+` / `=` : Increase confidence threshold by 0.05
- `-` : Decrease confidence threshold by 0.05
- `s` : Save a screenshot of the current frame
- `q` : Quit the video stream

## Project Structure

- `train.py`: Handles model instantiation, dataset fetching, and YOLOv8 training/hyperparameter configuration.
- `test_inference.py`: Runs validation to calculate metrics and provides quality assurance checks.
- `object_detection.py`: The deployment interface utilizing OpenCV and YOLO for real-time bounding box annotations and statistics.
- `best.pt` / `runs/`: Output directories and cached best weights resulting from training.
- `yolov8s.pt` / `yolov5s.pt`: Baseline pretrained weights downloaded by Ultralytics.

## License & Acknowledgements

Powered by [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) and [OpenCV](https://opencv.org/).
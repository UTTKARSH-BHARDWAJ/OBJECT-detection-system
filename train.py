"""
YOLOv8 Object Detection Training Pipeline
==========================================
Fine-tunes a YOLOv8s model on the COCO dataset for robust object detection.
Uses Ultralytics' built-in dataset management to auto-download COCO.

Usage:
    python train.py                    # Train on COCO128 (quick test)
    python train.py --data coco.yaml   # Train on full COCO (slow, comprehensive)
    python train.py --epochs 100       # Custom epochs
    python train.py --resume           # Resume interrupted training
"""

import argparse
import sys
import os
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLOv8 Object Detection Model')
    parser.add_argument('--model', type=str, default='yolov8s.pt',
                        help='Base model to fine-tune (yolov8n.pt, yolov8s.pt, yolov8m.pt)')
    parser.add_argument('--data', type=str, default='coco128.yaml',
                        help='Dataset config (coco128.yaml for quick test, coco.yaml for full)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size for training')
    parser.add_argument('--device', type=str, default=None,
                        help='Device: "mps", "cuda", "cpu", or auto-detect')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from last checkpoint')
    parser.add_argument('--project', type=str, default='runs/detect',
                        help='Project directory for saving results')
    parser.add_argument('--name', type=str, default='train',
                        help='Run name')
    return parser.parse_args()


def detect_device():
    """Auto-detect the best available device."""
    import torch
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"✅ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        print("✅ Using Apple Silicon MPS GPU")
    else:
        device = 'cpu'
        print("⚠️  Using CPU (training will be slow)")
    return device


def train(args):
    """Run the training pipeline."""
    from ultralytics import YOLO

    # Auto-detect device if not specified
    device = args.device if args.device else detect_device()

    print("\n" + "=" * 60)
    print("🚀 YOLOv8 OBJECT DETECTION TRAINING")
    print("=" * 60)
    print(f"  Model:      {args.model}")
    print(f"  Dataset:    {args.data}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Device:     {device}")
    print(f"  Patience:   {args.patience}")
    print("=" * 60 + "\n")

    # Load model
    if args.resume:
        # Resume from last checkpoint
        last_ckpt = Path(args.project) / args.name / 'weights' / 'last.pt'
        if last_ckpt.exists():
            print(f"📂 Resuming from: {last_ckpt}")
            model = YOLO(str(last_ckpt))
        else:
            print(f"⚠️  No checkpoint found at {last_ckpt}, starting fresh")
            model = YOLO(args.model)
    else:
        print(f"📂 Loading base model: {args.model}")
        model = YOLO(args.model)

    # Train the model
    # Ultralytics handles dataset download automatically for coco128.yaml and coco.yaml
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        patience=args.patience,
        project=args.project,
        name=args.name,
        exist_ok=True,
        # Data augmentation settings
        hsv_h=0.015,       # HSV-Hue augmentation
        hsv_s=0.7,         # HSV-Saturation augmentation
        hsv_v=0.4,         # HSV-Value augmentation
        degrees=0.0,       # Rotation (+/- deg)
        translate=0.1,     # Translation (+/- fraction)
        scale=0.5,         # Scale (+/- gain)
        shear=0.0,         # Shear (+/- deg)
        perspective=0.0,   # Perspective (+/- fraction)
        flipud=0.0,        # Flip up-down probability
        fliplr=0.5,        # Flip left-right probability
        mosaic=1.0,        # Mosaic augmentation probability
        mixup=0.1,         # MixUp augmentation probability
        # Training settings
        optimizer='auto',  # SGD, Adam, AdamW, auto
        lr0=0.01,          # Initial learning rate
        lrf=0.01,          # Final learning rate factor
        warmup_epochs=3.0, # Warmup epochs
        warmup_bias_lr=0.1,
        cos_lr=True,       # Cosine LR schedule
        # Saving
        save=True,
        save_period=-1,    # Save checkpoint every N epochs (-1 = disabled, save best only)
        plots=True,        # Generate training plots
        val=True,          # Validate during training
    )

    # Print final results
    best_weights = Path(args.project) / args.name / 'weights' / 'best.pt'
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"  Best weights saved to: {best_weights}")

    # Run validation on best model
    print("\n📊 Running final validation...")
    best_model = YOLO(str(best_weights))
    val_results = best_model.val(data=args.data, device=device)

    print(f"\n📈 Final Metrics:")
    print(f"  mAP@50:      {val_results.box.map50:.4f}")
    print(f"  mAP@50-95:   {val_results.box.map:.4f}")
    print(f"  Precision:   {val_results.box.mp:.4f}")
    print(f"  Recall:      {val_results.box.mr:.4f}")
    print("=" * 60)

    return best_weights


if __name__ == '__main__':
    args = parse_args()
    best_weights = train(args)
    print(f"\n💡 To use this model for detection, run:")
    print(f"   python object_detection.py --weights {best_weights}")

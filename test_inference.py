"""
Test Inference Script
=====================
Validates the trained model by running inference on sample images.

Usage:
    python test_inference.py
    python test_inference.py --weights runs/detect/train/weights/best.pt
"""

import argparse
import sys
from pathlib import Path


def test_inference(weights='auto', data='coco128.yaml'):
    """Run inference tests on validation images."""
    from ultralytics import YOLO
    import torch

    print("=" * 60)
    print("🧪 OBJECT DETECTION INFERENCE TEST")
    print("=" * 60)

    # Resolve weights
    if weights == 'auto':
        candidates = [
            'runs/detect/runs/detect/train/weights/best.pt',
            'runs/detect/train/weights/best.pt',
            'runs/detect/train2/weights/best.pt',
            'yolov8s.pt',
        ]
        for c in candidates:
            if Path(c).exists():
                weights = c
                break
        else:
            weights = 'yolov8s.pt'

    print(f"  Weights: {weights}")

    # Load model
    model = YOLO(weights)
    print(f"  Classes: {len(model.names)}")
    print(f"  Device:  {'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'}")

    # Run validation
    print("\n📊 Running validation on dataset...")
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        val_results = model.val(data=data, device=device, verbose=False)

        map50 = val_results.box.map50
        map50_95 = val_results.box.map
        precision = val_results.box.mp
        recall = val_results.box.mr

        print(f"\n📈 Validation Results:")
        print(f"  mAP@50:      {map50:.4f}")
        print(f"  mAP@50-95:   {map50_95:.4f}")
        print(f"  Precision:   {precision:.4f}")
        print(f"  Recall:      {recall:.4f}")

        # Quality checks
        tests_passed = 0
        tests_total = 4

        def check(name, value, threshold):
            nonlocal tests_passed
            status = "✅ PASS" if value >= threshold else "❌ FAIL"
            if value >= threshold:
                tests_passed += 1
            print(f"  {status}: {name} = {value:.4f} (threshold: {threshold})")

        print(f"\n🔍 Quality Checks:")
        check("mAP@50", map50, 0.40)
        check("mAP@50-95", map50_95, 0.20)
        check("Precision", precision, 0.40)
        check("Recall", recall, 0.30)

        print(f"\n{'=' * 60}")
        print(f"  Results: {tests_passed}/{tests_total} checks passed")

        if tests_passed == tests_total:
            print("  🎉 All quality checks passed! Model is ready for deployment.")
        elif tests_passed >= 2:
            print("  ⚠️  Some checks failed. Model may need more training.")
        else:
            print("  ❌ Most checks failed. Consider training for more epochs.")
        print(f"{'=' * 60}")

        return tests_passed == tests_total

    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        print("   This may happen if the dataset isn't downloaded yet.")
        print("   Try running: python train.py --data coco128.yaml --epochs 1")
        return False


# Quick inference test on a single image
def test_single_image(weights='auto'):
    """Test inference on a single image to verify the model works."""
    from ultralytics import YOLO
    import numpy as np

    print("\n🖼️  Testing single-image inference...")

    if weights == 'auto':
        candidates = [
            'runs/detect/train/weights/best.pt',
            'yolov8s.pt',
        ]
        for c in candidates:
            if Path(c).exists():
                weights = c
                break
        else:
            weights = 'yolov8s.pt'

    model = YOLO(weights)

    # Create a dummy test image (640x640 RGB)
    dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    results = model.predict(dummy_img, verbose=False)

    assert results is not None, "Model returned None"
    assert len(results) > 0, "Model returned empty results"
    print(f"  ✅ Model inference works (detected {len(results[0].boxes)} objects in random noise)")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='auto')
    parser.add_argument('--data', type=str, default='coco128.yaml')
    args = parser.parse_args()

    # Test 1: Basic inference
    test_single_image(args.weights)

    # Test 2: Validation metrics
    test_inference(args.weights, args.data)

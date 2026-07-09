#!/usr/bin/env python3
"""
Visualize YOLO OBB labels on images for verification.

Draws OBB boxes (4-corner contour) with class name, saves to output directory.

Usage:
    python visualize_labels.py
    python visualize_labels.py --image frame_0000  # single image
    python visualize_labels.py --all --show         # all images with window display
"""

import os
import sys
import glob
import argparse
import numpy as np
import cv2
from pathlib import Path


# ── Config ───────────────────────────────────────────────────────────────────

IMAGE_DIR = "/mnt/c/Users/finn.zhang/project/Datasets/yolo_zhonghua_counter/images/val"
LABEL_DIR = "/mnt/c/Users/finn.zhang/project/Datasets/yolo_zhonghua_counter/labels/val"
OUTPUT_DIR = "/home/finn/ultralytics-main-count/ultralytics/labels/label_viz"

# Class names (update as needed)
CLASS_NAMES = {0: "bag"}

COLORS = [
    (0, 255, 0),    # green
    (255, 0, 0),    # blue
    (0, 0, 255),    # red
    (255, 255, 0),  # cyan
    (255, 0, 255),  # magenta
    (0, 255, 255),  # yellow
    (128, 0, 255),  # orange
    (255, 128, 0),  # sky blue
]


# ── Core ─────────────────────────────────────────────────────────────────────

def load_obb_labels(label_path):
    """Load YOLO OBB labels: class x1 y1 x2 y2 x3 y3 x4 y4 (normalized)."""
    labels = []
    if not os.path.exists(label_path):
        return labels

    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            cls_id = int(float(parts[0]))
            coords = [float(x) for x in parts[1:9]]
            # Convert to 4 corner points (x,y)
            pts = [(coords[i], coords[i+1]) for i in range(0, 8, 2)]
            labels.append((cls_id, pts))
    return labels


def draw_obb(img, pts, cls_id, color, thickness=2):
    """Draw oriented bounding box as a contour on the image."""
    h, w = img.shape[:2]
    pts_px = [(int(p[0] * w), int(p[1] * h)) for p in pts]
    pts_arr = np.array(pts_px, dtype=np.int32).reshape((-1, 1, 2))

    # Draw filled box (semi-transparent)
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts_arr], color)
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

    # Draw outline
    cv2.polylines(img, [pts_arr], isClosed=True, color=color, thickness=thickness)

    # Draw corner points and label
    label = f"{CLASS_NAMES.get(cls_id, str(cls_id))}"
    x_min = min(p[0] for p in pts_px)
    y_min = min(p[1] for p in pts_px)
    cv2.putText(img, label, (x_min, y_min - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def visualize_one(image_path, label_path, output_path=None, show=False):
    """Visualize one image with its labels."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"  Cannot read: {image_path}")
        return None

    labels = load_obb_labels(label_path)
    if not labels:
        print(f"  No labels: {os.path.basename(image_path)}")
        # Still save/display the original image
    else:
        for i, (cls_id, pts) in enumerate(labels):
            color = COLORS[cls_id % len(COLORS)]
            draw_obb(img, pts, cls_id, color)

    # Overlay count
    cv2.putText(img, f"Objects: {len(labels)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, img)

    if show:
        cv2.imshow("Label Verification", img)
        cv2.waitKey(0)

    return img


def main():
    parser = argparse.ArgumentParser(description="Visualize YOLO OBB labels")
    parser.add_argument("--image", type=str, default=None,
                        help="Single image basename (without extension)")
    parser.add_argument("--all", action="store_true", help="Process all images")
    parser.add_argument("--show", action="store_true", help="Show each image in a window")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR,
                        help="Output directory for visualization images")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.image:
        basename = args.image
        img_path = os.path.join(IMAGE_DIR, basename + ".jpg")
        label_path = os.path.join(LABEL_DIR, basename + ".txt")
        out_path = os.path.join(args.output, f"viz_{basename}.jpg")

        print(f"Image:  {img_path}")
        print(f"Labels: {label_path}")
        labels = load_obb_labels(label_path)
        print(f"Boxes:  {len(labels)}")

        visualize_one(img_path, label_path, out_path, show=args.show)
        print(f"Saved:  {out_path}")

    else:
        # Process all or a sample
        image_files = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))
        total = len(image_files)
        if total == 0:
            print(f"No images found in {IMAGE_DIR}")
            return

        print(f"Found {total} images")
        ok = 0
        no_label = 0

        for img_path in image_files:
            basename = Path(img_path).stem
            label_path = os.path.join(LABEL_DIR, basename + ".txt")
            out_path = os.path.join(args.output, f"viz_{basename}.jpg")

            labels = load_obb_labels(label_path)
            if not labels:
                no_label += 1

            visualize_one(img_path, label_path, out_path, show=False)
            ok += 1

            if ok % 20 == 0:
                print(f"  [{ok}/{total}] processed...")

        print(f"\nDone: {ok} images visualized")
        print(f"  With labels: {ok - no_label}")
        print(f"  Without labels: {no_label}")
        print(f"  Output: {args.output}")

    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

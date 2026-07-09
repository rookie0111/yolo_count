import argparse

from ultralytics import YOLO


def main():
    """Export a YOLO model to ONNX format."""
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX")
    parser.add_argument("--model", "-m", default="weights/best.pt", help="Path to trained .pt weights")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--output", "-o", default="weights/best.onnx", help="Output ONNX path")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.export(format="onnx", imgsz=args.imgsz)

    print(f"ONNX model exported to {args.output}")


if __name__ == "__main__":
    main()

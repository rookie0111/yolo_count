"""
RKNN conversion tool for RK3588 deployment.
Requires rknn-toolkit2 to be installed on the target platform.

Usage:
    python tools/convert_rknn.py --onnx weights/best.onnx --output weights/bag_counter.rknn
"""

import argparse


def main():
    """Convert ONNX model to RKNN format for RK3588."""
    parser = argparse.ArgumentParser(description="Convert ONNX to RKNN for RK3588")
    parser.add_argument("--onnx", default="weights/best.onnx", help="Input ONNX model path")
    parser.add_argument("--output", default="weights/bag_counter.rknn", help="Output RKNN model path")
    parser.add_argument("--dataset", default="dataset.txt", help="Quantization dataset list file")
    parser.add_argument("--no-quant", action="store_true", help="Skip quantization")
    args = parser.parse_args()

    try:
        from rknn.api import RKNN
    except ImportError:
        print("Error: rknn-toolkit2 not installed.")
        print("This script must run on the RK3588 platform with rknn-toolkit2.")
        return

    rknn = RKNN()
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform="rk3588",
    )

    ret = rknn.load_onnx(model=args.onnx)
    if ret != 0:
        print("Load ONNX failed")
        return

    ret = rknn.build(do_quantization=not args.no_quant, dataset=args.dataset)
    if ret != 0:
        print("Build RKNN failed")
        return

    ret = rknn.export_rknn(args.output)
    if ret != 0:
        print("Export RKNN failed")
        return

    rknn.release()
    print(f"RKNN model saved to {args.output}")


if __name__ == "__main__":
    main()

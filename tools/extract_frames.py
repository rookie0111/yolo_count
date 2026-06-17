import argparse
import subprocess
from pathlib import Path


def main():
    """Extract frames from a video file using ffmpeg."""
    parser = argparse.ArgumentParser(description="Extract frames from video for dataset creation")
    parser.add_argument("--input", "-i", required=True, help="Input video file path")
    parser.add_argument("--output", "-o", default="dataset_raw", help="Output directory for frames")
    parser.add_argument("--fps", type=float, default=2, help="Frames per second to extract")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    output_pattern = str(Path(args.output) / "frame_%05d.jpg")
    cmd = ["ffmpeg", "-i", args.input, "-vf", f"fps={args.fps}", output_pattern]

    print(f"Extracting frames at {args.fps} FPS...")
    subprocess.run(cmd)
    print(f"Done. Frames saved to {args.output}")


if __name__ == "__main__":
    main()

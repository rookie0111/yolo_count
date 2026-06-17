# Conveyor Belt Bag Counting System

A computer vision system for real-time bag counting on conveyor belts, built on [Ultralytics YOLO v8.4.21](https://github.com/ultralytics/ultralytics). Supports USB industrial cameras, YOLO detection + ByteTrack tracking + diagonal cross-line counting, with deployment on Orange Pi 5 / RK3588 edge devices.

[中文说明](README.zh-CN.md)

## Features

- **Diagonal counting line**: Supports arbitrary-angle virtual lines via cross-product method, not limited to axis-aligned
- **4 direction rules**: `up_right`, `down_right`, `up_left`, `down_left`, `any` (backward-compatible)
- **Track ID dedup**: Prevents double-counting via `counted_ids` set
- **CSV logging**: Daily auto-rolling log files (`logs/count_YYYYMMDD.csv`)
- **Serial display**: USB digital number display support (optional)
- **Dual runtime**: Python (desktop dev/test) + C++ (RK3588 edge deployment)
- **systemd auto-start**: Deploy as a Linux service with auto-restart

## Architecture

```
Camera / Video
      │
      ▼
┌─────────────┐    ┌───────────┐    ┌──────────────────┐
│ YOLO Detect  │───▶│ ByteTrack  │───▶│ Cross-line Count  │
│ (bag class)  │    │ (track ID) │    │ (diagonal line)   │
└─────────────┘    └───────────┘    └────────┬─────────┘
                                             │
                                    ┌────────┴─────────┐
                                    ▼                  ▼
                              CSV Logger        Serial Display
```

- **Detection**: YOLOv8n/YOLO26n custom bag model
- **Tracking**: ByteTrack (default) or BoT-SORT
- **Counting**: Cross-product `point_line_side()` determines line crossing direction; Shapely `intersects` for trajectory-line intersection check

## Quick Start

### Prerequisites

- Python >= 3.8, PyTorch >= 1.8
- Recommended: Conda environment `bag_counter` with Python 3.10

```bash
conda create -n bag_counter python=3.10
conda activate bag_counter
pip install -e "."
```

### Run with video (test mode)

```bash
python src/main.py --source path/to/video.mp4 --show
```

### Run with camera (production mode)

```bash
python src/main.py --config configs/runtime.yaml
```

### Configuration

Edit `configs/runtime.yaml`:

```yaml
camera:
  source: 0           # Camera index or video path
  width: 1280
  height: 720

model:
  type: yolo26
  weight: runs/detect/bag_data6/weights/best.pt

tracker:
  type: bytetrack

counter:
  line_p1: [1000, 400]   # Counting line start point
  line_p2: [300, 400]    # Counting line end point
  direction: up_right    # Direction rule

runtime:
  target_fps: 10
  show_window: false

display:
  enable: false          # USB serial display
```

## C++ Version (Edge Deployment)

The `counter_cpp/` directory contains a C++ port using ONNX Runtime and OpenCV, optimized for RK3588 / Orange Pi 5.

```bash
cd counter_cpp
./build.sh                    # Build with ONNX
./build.sh -DRKNN=ON          # Build with RKNN support
./build/bag_counter --source video.mp4 --show
./build/bag_counter --config config.yaml --source 0 --show --save
```

Bundled Eigen3 headers in `counter_cpp/eigen3/` provide Kalman filter math for ByteTrack.

## Training Your Own Model

### 1. Extract frames from video

```bash
python tools/extract_frames.py -i video.mp4 -o dataset_raw --fps 2
```

### 2. Label data in YOLO format

Annotate images with tools like LabelImg, place labels in `dataset/labels/train/` and images in `dataset/images/train/`.

### 3. Train

```bash
yolo detect train model=yolov8n.pt data=dataset/data.yaml epochs=100 imgsz=640
```

### 4. Export for edge deployment

```bash
python tools/export_onnx.py --model weights/best.pt
# Then convert to RKNN for RK3588: python tools/convert_rknn.py
```

## Project Structure

```
├── src/                     # Counting application layer
│   ├── main.py             # Main entry point
│   ├── utils.py            # Config loading
│   ├── logger.py           # CSV daily logger
│   └── display.py          # Serial number display
├── configs/
│   └── runtime.yaml        # Runtime configuration
├── counter_cpp/            # C++ port (RK3588/ONNX Runtime)
├── ultralytics/            # YOLO v8.4.21 framework
│   └── solutions/
│       ├── object_counter.py   # Modified: diagonal line support
│       ├── config.py           # Modified: direction fields
│       └── solutions.py        # Modified: total_count field
├── deploy/                 # systemd service for auto-start
├── tools/                  # Helper scripts
├── tests/                  # Unit tests
│   └── test_bag_counter.py
├── AGENT.md                # Development plan (Chinese)
├── Guide_AGENT.md          # Requirements spec (Chinese)
└── AGENTS.md               # AI agent instruction file
```

## Testing

```bash
pip install -e ".[dev]"
pytest tests/test_bag_counter.py    # Counting-specific tests
pytest tests/                        # Full test suite
pytest --slow tests/                 # Include slow tests (GPU training, export)
```

## Deploy to Orange Pi 5

```bash
bash deploy/install_service.sh
```

The service auto-starts on boot, counts from zero on each startup.

## License

[AGPL-3.0](LICENSE) — Based on [Ultralytics YOLO](https://github.com/ultralytics/ultralytics).

## References

- [Ultralytics YOLO Docs](https://docs.ultralytics.com/)
- `AGENT.md` — Developer implementation plan
- `Guide_AGENT.md` — Requirements specification

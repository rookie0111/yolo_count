# AGENTS.md — 袋装货物计数系统

## What this repo is

A fork of [Ultralytics YOLO v8.4.21](https://github.com/ultralytics/ultralytics) with a custom conveyor-belt bag counting system layered on top. The core YOLO framework lives in `ultralytics/` and the counting application in `src/`.

Reference docs: `AGENT.md` (implementation plan, Chinese) and `Guide_AGENT.md` (requirements spec, Chinese).

## Modified files vs upstream Ultralytics

| File | Change |
|------|--------|
| `ultralytics/solutions/object_counter.py` | Added diagonal-line direction support via `point_line_side()` cross-product, `total_count`, `draw_count_overlay()` |
| `ultralytics/solutions/config.py` | Added `direction`, `polygon_mode`, `camera_source` fields |
| `ultralytics/solutions/solutions.py` | Added `SolutionResults.total_count` return field |

All other `ultralytics/` code is stock v8.4.21 — do not modify unless you understand the upstream implications.

## Entry points

| Entry | Command |
|-------|---------|
| Counting app (Python) | `python src/main.py --config configs/runtime.yaml` |
| Counting app (with video) | `python src/main.py --source video.mp4 --show` |
| Counting app (C++) | `cd counter_cpp && ./build.sh && ./build/bag_counter --source video.mp4 --show` |
| Training | `yolo detect train model=yolov8n.pt data=dataset/data.yaml epochs=100 imgsz=640` |
| ONNX export | `python tools/export_onnx.py --model weights/best.pt` |
| Frame extraction | `python tools/extract_frames.py -i video.mp4 -o dataset_raw --fps 2` |

## Tests

```bash
pip install -e ".[dev]"
pytest tests/test_bag_counter.py            # Counting-specific tests only
pytest tests/                                # Full test suite
pytest --slow tests/                         # Include slow tests
```

Test markers: `--slow` (opt-in for expensive tests). Fast suite excludes GPU training and export tests.

## Counting algorithm

- Cross product `point_line_side(px, py, p1, p2)` determines which side of the counting line a point is on
- Track trajectory segment (prev → current) must intersect the counting line (Shapely `intersects`)
- Direction must match the configured `direction` rule (`up_right`, `down_right`, `up_left`, `down_left`, `any`)
- `track_id` deduplication via `counted_ids` set
- Direction rules are keyed as `(entry_side, exit_side)` where `-1` = clockwise side, `+1` = counter-clockwise side

## C++ port (`counter_cpp/`)

The C++ version mirrors the Python counting logic for RK3588/ONNX Runtime edge deployment. Requires OpenCV, Eigen3 (bundled), ONNX Runtime. Optional: RKNN SDK.

```bash
cd counter_cpp && ./build.sh          # ONNX only
./build.sh -DRKNN=ON                  # With RKNN support
```

The Eigen3 headers in `counter_cpp/eigen3/` are bundled third-party code — `.gitignore` excludes this directory from tracking.

## .gitignore

Custom whitelist-based `.gitignore`: starts with `*` (ignore everything), then `!*/` (allow directory traversal), then `!` patterns for text file extensions only. Model weights (`.pt`, `.onnx`, `.rknn`), images, videos, and binary build artifacts are all excluded. Core directories explicitly excluded: `.vscode/`, `counter_cpp/build/`, `counter_cpp/eigen3/`.

## SSH / GitHub

GitHub SSH uses `ssh.github.com` port 443 (port 22 blocked in this environment). The SSH config is in `~/.ssh/config`:
```
Host github.com
    HostName ssh.github.com
    Port 443
    User git
```

## YOLO Pipeline — SAM3 Auto-label → Train → ONNX Export

Automated pipeline for training custom YOLO models with SAM3-generated labels.

### Quick start

```bash
cd ultralytics/labels

# Full pipeline (one command)
python pipeline.py all \
    --images /path/to/images/val \
    --labels /path/to/labels/val \
    --dataset-root /path/to/dataset \
    --task detect --epochs 100 \
    --prompt "object to detect" --names "class_name"

# Or step by step
python pipeline.py precompute --prompt "object to detect"     # 1. Text embeddings
python pipeline.py label --images ... --labels ... --prompt ... # 2. Auto-label
python pipeline.py data --dataset-root ... --names ...          # 3. YAML config
python pipeline.py train --data data.yaml --task detect ...     # 4. Train
python pipeline.py export --weights runs/.../best.pt          # 5. ONNX
```

### Task types

| `--task` | Label format | Model suffix | Label `--label-task` |
|----------|-------------|--------------|----------------------|
| `detect` | `class cx cy w h` | `.pt` | `hbb` |
| `obb` | `class x1 y1 x2 y2 x3 y3 x4 y4` | `-obb.pt` | `obb` |
| `segment` | `class x1 y1 x2 y2 ...` polygon | `-seg.pt` | `hbb` |
| `track` | Same as detect | `.pt` | — |

### Scripts

| Script | Purpose |
|--------|---------|
| `ultralytics/labels/pipeline.py` | Main pipeline with 6 subcommands |
| `ultralytics/labels/SKILL.md` | Skill documentation |
| `ultralytics/labels/precompute_text_embed.py` | Text embedding pre-computation |
| `ultralytics/labels/auto_label_sam3.py` | Standalone SAM3 labeling |
| `ultralytics/labels/visualize_labels.py` | Label visualization |
| `ultralytics/labels/fix_labels.py` | Fix label formatting |
| `ultralytics/sam3_predict.py` | SAM3 Python predictor with visualization |
| `tools/export_sam3_onnx.py` | Export SAM3 vision backbone → ONNX |
| `tools/export_sam3_detector_onnx.py` | Export SAM3 detector core → ONNX |

### Installed skills (opencode)

| Skill | Usage |
|-------|-------|
| `find-skills` / `find-skills-cn` | Discover and install new agent skills |
| `qiaomu-ai-prd` | Generate AI-implementable PRDs from product ideas |
| `customize-opencode` | Edit opencode configuration (agents, skills, permissions) |

To list available skills: `/help` in opencode, or use the `skill` tool with `find-skills`.

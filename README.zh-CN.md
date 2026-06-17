# 输送带袋装货物计数系统

基于 [Ultralytics YOLO v8.4.21](https://github.com/ultralytics/ultralytics) 的输送带袋装货物实时计数系统。支持 USB 工业相机、YOLO 检测 + ByteTrack 跟踪 + 斜向跨线计数，可部署到 Orange Pi 5 / RK3588 边缘设备。

[English](README.md)

## 功能特性

- **斜向计数线**：通过叉积法支持任意角度虚拟计数线，不限于轴对齐
- **4 种方向规则**：`up_right`、`down_right`、`up_left`、`down_left`、`any`（向后兼容）
- **跟踪 ID 去重**：通过 `counted_ids` 集合防止重复计数
- **CSV 日志**：按天自动滚动的日志文件（`logs/count_YYYYMMDD.csv`）
- **串口数字屏**：支持 USB 串口数字显示屏（可选）
- **双语言运行时**：Python（桌面开发/测试）+ C++（RK3588 边缘部署）
- **systemd 自启动**：以 Linux 服务方式部署，支持崩溃自动重启

## 系统架构

```
摄像头 / 视频
      │
      ▼
┌─────────────┐    ┌───────────┐    ┌──────────────────┐
│ YOLO 检测    │───▶│ ByteTrack  │───▶│ 跨线计数          │
│ (bag 类别)   │    │ (跟踪 ID)  │    │ (斜向计数线)      │
└─────────────┘    └───────────┘    └────────┬─────────┘
                                             │
                                    ┌────────┴─────────┐
                                    ▼                  ▼
                              CSV 日志         串口数字屏
```

- **检测**：YOLOv8n/YOLO26n 自定义袋装货物模型
- **跟踪**：ByteTrack（默认）或 BoT-SORT
- **计数**：叉积法 `point_line_side()` 判断轨迹过线方向；Shapely `intersects` 检测轨迹与计数线相交

## 快速开始

### 环境准备

- Python >= 3.8，PyTorch >= 1.8
- 推荐：Conda 环境 `bag_counter`，Python 3.10

```bash
conda create -n bag_counter python=3.10
conda activate bag_counter
pip install -e "."
```

### 视频测试模式

```bash
python src/main.py --source path/to/video.mp4 --show
```

### 摄像头生产模式

```bash
python src/main.py --config configs/runtime.yaml
```

### 配置说明

编辑 `configs/runtime.yaml`：

```yaml
camera:
  source: 0           # 摄像头索引或视频路径
  width: 1280
  height: 720

model:
  type: yolo26
  weight: runs/detect/bag_data6/weights/best.pt

tracker:
  type: bytetrack

counter:
  line_p1: [1000, 400]   # 计数线起点坐标
  line_p2: [300, 400]    # 计数线终点坐标
  direction: up_right    # 计数方向规则

runtime:
  target_fps: 10
  show_window: false

display:
  enable: false          # USB 串口数字屏
```

## C++ 版本（边缘部署）

`counter_cpp/` 目录包含使用 ONNX Runtime 和 OpenCV 的 C++ 移植版本，针对 RK3588 / Orange Pi 5 优化。

```bash
cd counter_cpp
./build.sh                    # 使用 ONNX 构建
./build.sh -DRKNN=ON          # 启用 RKNN 支持
./build/bag_counter --source video.mp4 --show
./build/bag_counter --config config.yaml --source 0 --show --save
```

`counter_cpp/eigen3/` 中的 Eigen3 头文件为 ByteTrack 提供卡尔曼滤波所需矩阵运算。

## 训练自己的模型

### 1. 从视频抽帧

```bash
python tools/extract_frames.py -i video.mp4 -o dataset_raw --fps 2
```

### 2. 标注数据

使用 LabelImg 等工具按 YOLO 格式标注，标签放入 `dataset/labels/train/`，图片放入 `dataset/images/train/`。

### 3. 训练

```bash
yolo detect train model=yolov8n.pt data=dataset/data.yaml epochs=100 imgsz=640
```

### 4. 导出部署

```bash
python tools/export_onnx.py --model weights/best.pt
# RK3588 部署：python tools/convert_rknn.py
```

## 项目结构

```
├── src/                     # 计数应用层
│   ├── main.py             # 主入口
│   ├── utils.py            # 配置加载
│   ├── logger.py           # CSV 日志
│   └── display.py          # 串口数字屏
├── configs/
│   └── runtime.yaml        # 运行时配置
├── counter_cpp/            # C++ 移植版 (RK3588/ONNX Runtime)
├── ultralytics/            # YOLO v8.4.21 框架
│   └── solutions/
│       ├── object_counter.py   # 已修改：斜线方向支持
│       ├── config.py           # 已修改：方向字段
│       └── solutions.py        # 已修改：total_count 字段
├── deploy/                 # systemd 自启动服务
├── tools/                  # 辅助脚本
├── tests/                  # 单元测试
│   └── test_bag_counter.py
├── AGENT.md                # 开发方案
├── Guide_AGENT.md          # 需求规范
└── AGENTS.md               # AI 代理指令文件
```

## 测试

```bash
pip install -e ".[dev]"
pytest tests/test_bag_counter.py    # 计数专用测试
pytest tests/                        # 全部测试
pytest --slow tests/                 # 包含慢测试（GPU 训练、导出等）
```

## Orange Pi 5 部署

```bash
bash deploy/install_service.sh
```

服务开机自启动，每次启动从 0 开始计数。

## 许可证

[AGPL-3.0](LICENSE) — 基于 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)。

## 参考资料

- [Ultralytics YOLO 文档](https://docs.ultralytics.com/)
- `AGENT.md` — 开发者实施方案
- `Guide_AGENT.md` — 需求规范文档

# AGENT.md — 输送带袋装货物计数系统开发方案

> 基于 Ultralytics YOLO v8.4.21 现有 `solutions/object_counter.py` 扩展。
> 需求规范详见 `Guide_AGENT.md`。

---

## 1. 现状分析

| 组件               | 位置                                      | 状态                  |
| ------------------ | ----------------------------------------- | --------------------- |
| YOLO26 检测        | `ultralytics/`                            | ✅ 可用               |
| ByteTrack/BoT-SORT | `ultralytics/trackers/`                   | ✅ 可用               |
| BaseSolution 基类  | `ultralytics/solutions/solutions.py`      | ✅ 可用               |
| ObjectCounter 计数 | `ultralytics/solutions/object_counter.py` | ✅ 已修改（增加斜线） |
| SolutionConfig     | `ultralytics/solutions/config.py`         | ✅ 已修改（增加参数） |
| 运行时配置         | `configs/runtime.yaml`                    | ✅ 已创建             |
| CSV 日志           | `src/logger.py`                           | ✅ 已创建             |
| 数字屏显示         | `src/display.py`                          | ✅ 已创建             |
| 主程序入口         | `src/main.py`                             | ✅ 已创建             |
| Systemd 部署       | `deploy/`                                 | ✅ 已创建             |
| 工具脚本           | `tools/`                                  | ✅ 已创建             |
| 单元测试           | `tests/test_bag_counter.py`               | ✅ 已创建             |

---

## 2. 关键技术决策

### 斜线方向判断

原有 `ObjectCounter.count_objects()` 对 2 点线段仅支持轴对齐方向（水平/垂直），通过比较 x 或 y 坐标判断方向。**修改后**增加 `point_line_side` 叉积法，支持任意角度的计数线方向判断。

方向规则（以 p1→p2 方向的视角）：

- `up_right`: 从顺时针侧(-1) → 逆时针侧(+1) 过线
- `down_right`: 从逆时针侧(+1) → 顺时针侧(-1) 过线
- `up_left`: 从逆时针侧(+1) → 顺时针侧(-1) 过线
- `down_left`: 从顺时针侧(-1) → 逆时针侧(+1) 过线
- `any`: 不限制方向（兼容原有逻辑）

### 向后兼容

当 `direction` 设为 `any`（默认值）时，保留原有水平/垂直线方向判断逻辑，完全向后兼容。

---

## 3. 目录结构

```
ultralytics-main-count/
├── AGENT.md                      # 本文件
├── Guide_AGENT.md                # 需求规范
├── configs/
│   └── runtime.yaml              # 运行时配置
├── src/
│   ├── main.py                   # 主入口
│   ├── utils.py                  # 配置加载
│   ├── logger.py                 # CSV 日志
│   └── display.py                # 数字屏显示
├── deploy/
│   ├── bag_counter.service       # systemd 服务
│   └── install_service.sh        # 安装脚本
├── tools/
│   ├── extract_frames.py         # 视频抽帧
│   ├── export_onnx.py            # ONNX 导出
│   └── convert_rknn.py           # RKNN 转换
├── logs/                         # 日志输出
├── outputs/                      # 结果输出
├── tests/
│   └── test_bag_counter.py       # 单元测试
├── ultralytics/                  # YOLO 框架
│   └── solutions/
│       ├── object_counter.py     # 已修改
│       ├── config.py             # 已修改
│       └── solutions.py          # 已修改（SolutionResults.total_count）
└── ...
```

---

## 4. 关键算法说明

### point_line_side 叉积法

```python
def point_line_side(px, py, p1, p2):
    """叉积：>0 顺时针侧，<0 逆时针侧，=0 在线段上."""
    return (p2[0] - p1[0]) * (py - p1[1]) - (p2[1] - p1[1]) * (px - p1[0])
```

计数触发条件：

1. 轨迹线段（prev_position → current_centroid）与计数线相交（Shapely `intersects`）
2. 过线方向符合配置的 direction 规则
3. track_id 未被计数过（counted_ids 去重）

### 日志格式

`logs/count_YYYYMMDD.csv`:

```
timestamp,count,track_id
2026-06-04 08:30:12,1,5
2026-06-04 08:30:18,2,6
```

**不用于恢复计数**。开机从 0 开始。

---

## 5. 使用方式

### 训练模型

```bash
conda activate bag_counter
yolo detect train model=yolov8n.pt data=dataset/data.yaml epochs=100 imgsz=640
```

### 测试计数（视频文件）

```bash
python src/main.py --source videos/test/test.mp4 --show
```

### 生产运行（摄像头）

```bash
python src/main.py --config configs/runtime.yaml
```

### 部署到 Orange Pi 5

```bash
bash deploy/install_service.sh
```

### 导出 ONNX

```bash
python tools/export_onnx.py --model weights/best.pt
```

### 抽帧

```bash
python tools/extract_frames.py -i videos/raw/conveyor.mp4 -o dataset_raw --fps 2
```

---

## 6. 验收对照

| #   | Guide_AGENT.md 第17节 | 实现                               |
| --- | --------------------- | ---------------------------------- |
| 1   | 读取测试视频          | `src/main.py --source video.mp4`   |
| 2   | 检测袋装货物          | YOLOv8n + 自定义 bag 模型          |
| 3   | 分配 track_id         | ByteTrack via `extract_tracks()`   |
| 4   | 斜向计数线            | point_line_side + direction 配置   |
| 5   | 过线计数+1            | `total_count` 自增                 |
| 6   | 同ID不重复            | `counted_ids` 去重                 |
| 7   | 输出累计数量          | 终端打印 + CSV 日志                |
| 8   | 保存日志              | `logs/count_YYYYMMDD.csv`          |
| 9   | 配置文件调整          | `configs/runtime.yaml`             |
| 10  | Orange Pi 自启动      | `deploy/bag_counter.service`       |
| 11  | 断电从0开始           | `reset_count_on_start`，不恢复历史 |

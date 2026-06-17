````markdown
# 输送带袋装货物计数系统开发说明

> 如需克隆源码，请自行通过 Git 克隆对应代码仓库。
>
> 示例：
>
> ```bash
> git clone https://github.com/username/repository.git
> ```
>
> 本文档只提供开发说明、目录结构、配置示例和关键代码片段，不直接粘贴完整项目源码。

---

## 1. 项目目标

开发一个用于输送带袋装货物自动计数的系统。

系统使用：

```text
USB 工业相机
+
YOLO 检测模型
+
ByteTrack/SORT 跟踪
+
虚拟斜线过线计数
+
Orange Pi 5 / RK3588 边缘部署
```

最终效果：

```text
开机自动启动计数
摄像头自动采集
袋子经过计数线时计数 +1
同一个袋子不重复计数
断电/关机后下次启动从 0 开始
不显示桌面
可选数字屏显示累计数量
```

---

## 2. 必须遵守的开发要求

```text
1. 必须新建 conda 环境开发，不能污染 base 环境
2. 开发、训练、测试都在新 conda 环境中进行
3. 不允许只用 YOLO 每帧检测数量累加
4. 必须使用跟踪 ID 去重
5. 必须使用虚拟线过线计数
6. 必须支持斜线计数线
7. 必须支持配置文件修改参数
8. 开发板只做推理和计数，不在开发板上训练
9. 正式部署不显示桌面
10. 程序需要支持开机自启动
```

---

## 3. 新建 Conda 环境

创建环境：

```bash
conda create -n bag_counter python=3.10 -y
```

激活环境：

```bash
conda activate bag_counter
```

升级 pip：

```bash
pip install --upgrade pip
```

安装基础依赖：

```bash
pip install ultralytics opencv-python numpy scipy filterpy lap pyyaml tqdm matplotlib pandas pyserial
```

如果需要 CPU 版 PyTorch：

```bash
pip install torch torchvision torchaudio
```

如果需要 GPU 版 PyTorch，请根据本机 CUDA 版本安装对应版本。

检查环境：

```bash
python -c "import cv2; print(cv2.__version__)"
python -c "from ultralytics import YOLO; print('ultralytics ok')"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

导出环境：

```bash
conda env export > environment.yml
pip freeze > requirements.txt
```

---

## 4. 推荐项目结构

```text
bag-counter/
├── AGENT.md
├── README.md
├── environment.yml
├── requirements.txt
├── configs/
│   ├── data.yaml
│   ├── runtime.yaml
│   └── line_config.yaml
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   └── data.yaml
├── videos/
│   ├── raw/
│   └── test/
├── weights/
│   ├── best.pt
│   ├── best.onnx
│   └── bag_counter.rknn
├── tools/
│   ├── extract_frames.py
│   ├── split_dataset.py
│   ├── export_onnx.py
│   └── convert_rknn.py
├── src/
│   ├── main.py
│   ├── camera.py
│   ├── detector.py
│   ├── tracker.py
│   ├── counter.py
│   ├── display.py
│   ├── logger.py
│   └── utils.py
├── deploy/
│   ├── bag_counter.service
│   └── install_service.sh
├── logs/
└── outputs/
```

---

## 5. 硬件目标

推荐硬件：

```text
Orange Pi 5 4GB/8GB
1080P USB 工业相机
6mm/8mm 定焦镜头
稳定 5V 4A/5A 电源
散热片 + 风扇
金属摄像头支架
防尘防水外壳
可选 USB 串口数字屏
```

说明：

```text
1. 单路摄像头、无桌面运行时，Orange Pi 5 4GB 可满足需求
2. 预算允许时可选 8GB
3. 正式部署建议使用 Ubuntu/Debian Server
4. 不建议使用桌面系统长期运行
```

---

## 6. 算法方案

必须实现：

```text
YOLO 检测袋子
+
ByteTrack/SORT 跟踪目标 ID
+
虚拟斜线过线计数
+
方向判断
+
重复计数过滤
```

推荐优先级：

```text
YOLOv8n + ByteTrack
YOLOv5n + ByteTrack
YOLOv8n + SORT
YOLOv5n + SORT
```

类别只有一个：

```text
bag
```

不要使用每帧检测数量直接累加。

---

## 7. 数据采集与标注

固定最终摄像头机位后采集数据。

建议采集场景：

```text
正常输送
袋子稀疏
袋子密集
强光反光
阴影变化
空输送带
不同颜色袋子
夜间补光场景，若夜间使用
```

抽帧命令：

```bash
ffmpeg -i videos/raw/conveyor.mp4 -vf fps=2 dataset_raw/frame_%05d.jpg
```

标注数量：

```text
第一版验证：500~800 张
稳定上线版：1000~1500 张
夜间使用：1500~2500 张
```

标注规则：

```text
1. 每个袋子单独画框
2. 类别只标 bag
3. 不要多个袋子框成一个
4. 可见遮挡袋子也标
5. 空输送带图片不标框，作为负样本
6. 不标注地面、支架、车轮、黄布、天空等干扰物
```

数据格式：

```text
YOLO 格式
```

---

## 8. 数据集配置

`dataset/data.yaml`：

```yaml
path: ./dataset
train: images/train
val: images/val

names:
  0: bag
```

---

## 9. YOLO 训练

训练在电脑或云服务器完成，不在 Orange Pi 上训练。

激活环境：

```bash
conda activate bag_counter
```

训练 YOLOv8n：

```bash
yolo detect train model=yolov8n.pt data=dataset/data.yaml epochs=100 imgsz=640 batch=16
```

低性能电脑可用：

```bash
yolo detect train model=yolov8n.pt data=dataset/data.yaml epochs=80 imgsz=416 batch=8
```

训练完成后复制权重：

```text
runs/detect/train/weights/best.pt
```

到：

```text
weights/best.pt
```

验证：

```bash
yolo detect val model=weights/best.pt data=dataset/data.yaml imgsz=640
```

测试视频：

```bash
yolo detect predict model=weights/best.pt source=videos/test/test.mp4 conf=0.35 imgsz=640
```

---

## 10. ONNX 和 RKNN 部署

导出 ONNX：

```bash
yolo export model=weights/best.pt format=onnx imgsz=640
```

输出：

```text
weights/best.onnx
```

RKNN 转换示例：

```python
from rknn.api import RKNN


def main():
    rknn = RKNN()

    rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform="rk3588")

    ret = rknn.load_onnx(model="weights/best.onnx")
    if ret != 0:
        print("Load ONNX failed")
        exit(ret)

    ret = rknn.build(do_quantization=True, dataset="dataset.txt")
    if ret != 0:
        print("Build RKNN failed")
        exit(ret)

    ret = rknn.export_rknn("weights/bag_counter.rknn")
    if ret != 0:
        print("Export RKNN failed")
        exit(ret)

    rknn.release()
    print("RKNN export success")


if __name__ == "__main__":
    main()
```

`dataset.txt` 放 100~300 张现场图片路径，用于量化。

---

## 11. 计数核心逻辑

虚拟计数线由两个点定义：

```python
line_p1 = (x1, y1)
line_p2 = (x2, y2)
```

判断点在线哪一侧：

```python
def point_line_side(px, py, p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    value = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

    if value > 0:
        return 1
    else:
        return -1
```

计数伪代码：

```python
total_count = 0
counted_ids = set()
last_side = {}

for frame in video_stream:
    detections = detector.detect(frame)
    tracks = tracker.update(detections)

    for track in tracks:
        track_id = track.id
        x1, y1, x2, y2 = track.bbox

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        current_side = point_line_side(cx, cy, line_p1, line_p2)

        if track_id in last_side:
            crossed_line = last_side[track_id] != current_side

            if crossed_line and track_id not in counted_ids:
                total_count += 1
                counted_ids.add(track_id)

        last_side[track_id] = current_side
```

需要加入方向判断：

```python
if crossed_line and direction_ok and track_id not in counted_ids:
    total_count += 1
    counted_ids.add(track_id)
```

---

## 12. 运行配置

`configs/runtime.yaml` 示例：

```yaml
camera:
  source: 0
  width: 1280
  height: 720
  fps: 30

model:
  type: yolov8
  weight: weights/best.pt
  rknn_weight: weights/bag_counter.rknn
  input_size: 640
  conf_thres: 0.35
  iou_thres: 0.50

tracker:
  type: bytetrack
  track_thresh: 0.45
  high_thresh: 0.60
  match_thresh: 0.75
  track_buffer: 30

counter:
  line_p1: [180, 430]
  line_p2: [330, 250]
  direction: up_right
  enable_direction_check: true

runtime:
  target_fps: 10
  save_log: true
  save_debug_image: false
  show_window: false
  reset_count_on_start: true

display:
  type: serial
  enable: false
  port: /dev/ttyUSB0
  baudrate: 9600
  digits: 8
```

推荐运行参数：

```text
摄像头采集：1280×720
模型输入：640×640
推理帧率：5~15 FPS
置信度阈值：0.35~0.5
NMS 阈值：0.45~0.6
跟踪算法：ByteTrack 优先
计数方式：斜线过线计数
```

---

## 13. 数字屏显示

如果使用 USB 串口数字屏：

```python
import time

import serial


class NumberDisplay:
    def __init__(self, port="/dev/ttyUSB0", baudrate=9600, digits=8):
        self.digits = digits
        self.set = serial.Serial(port, baudrate, timeout=1)
        time.sleep(1)

    def show_number(self, number):
        text = str(number).zfill(self.digits)
        self.set.write((text + "\n").encode("ascii"))
```

实际通信格式以数字屏商家协议为准。

---

## 14. 开机自启动

`deploy/bag_counter.service`：

```ini
[Unit]
Description=Bag Counter Service
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/orangepi/bag-counter
ExecStart=/usr/bin/python3 /home/orangepi/bag-counter/src/main.py
Restart=always
RestartSec=3
User=orangepi

[Install]
WantedBy=multi-user.target
```

安装服务：

```bash
sudo cp deploy/bag_counter.service /etc/systemd/system/bag_counter.service
sudo systemctl daemon-reload
sudo systemctl enable bag_counter.service
sudo systemctl start bag_counter.service
```

查看状态：

```bash
sudo systemctl status bag_counter.service
```

查看日志：

```bash
journalctl -u bag_counter.service -f
```

---

## 15. 日志要求

保存计数日志，但不用于恢复计数。

日志路径：

```text
logs/count_YYYYMMDD.csv
```

日志格式：

```text
timestamp,count,track_id
2026-06-04 08:30:12,1,5
2026-06-04 08:30:18,2,6
```

开机清零要求：

```python
total_count = 0
```

不要从历史日志恢复计数。

---

## 16. 开发任务顺序

Agent 应按以下顺序开发：

```text
1. 创建 conda 环境 bag_counter
2. 创建项目目录结构
3. 实现配置文件读取
4. 实现摄像头/视频读取模块
5. 实现 YOLO 检测模块
6. 集成 ByteTrack 或 SORT 跟踪
7. 实现斜线过线计数
8. 实现方向判断和重复计数过滤
9. 实现日志保存
10. 实现可选串口数字屏显示
11. 实现测试视频离线计数
12. 导出 ONNX
13. 预留 RKNN 推理接口
14. 编写 systemd 服务文件
15. 部署到 Orange Pi 5 测试
```

---

## 17. 验收标准

PC 测试验收：

```text
1. 能读取本地测试视频
2. 能检测袋装货物
3. 能为每个袋子分配 track_id
4. 能设置斜向计数线
5. 袋子过线后计数 +1
6. 同一个袋子不重复计数
7. 能输出累计数量
8. 能保存计数日志
9. 能通过配置文件调整阈值和计数线
```

现场部署验收：

```text
1. Orange Pi 5 上电后自动启动程序
2. 摄像头自动采集画面
3. 数字屏或日志显示计数结果
4. 断电重启后计数从 0 开始
5. 连续运行 8 小时以上无崩溃
6. 袋子分离明显时计数准确率达到 95% 左右
```

---

## 18. 最终推荐方案

最终落地方案：

```text
Orange Pi 5 4GB/8GB
+
1080P USB 工业相机
+
YOLOv8n 或 YOLOv5n 单类别 bag 检测
+
ByteTrack 跟踪
+
斜向虚拟线过线计数
+
无桌面 systemd 开机自启动
+
可选 USB 串口数字屏显示
```
````

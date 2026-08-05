#!/bin/bash
# YOLO11m-seg 训练脚本
# 针对 RTX 4080 12GB 显存优化配置
# 数据集: xutong_military_yolo_seg (26张图, 2类别)

set -e

# 激活 yolo conda 环境
source /home/finn/miniconda3/etc/profile.d/conda.sh
conda activate yolo

echo "============================================"
echo "YOLO11m-seg 训练"
echo "RTX 4080 12GB 优化配置"
echo "============================================"

# 路径配置
PROJECT_DIR="/home/finn/ultralytics-main-count/ultralytics/yolo-seg"
DATA_YAML="/home/finn/ultralytics-main-count/ultralytics/yolo-seg/data.yaml"
MODEL="yolo11m-seg.pt"
SAVE_DIR="${PROJECT_DIR}/weights"

# 确保目录存在
mkdir -p "${SAVE_DIR}"

echo ""
echo "配置信息:"
echo "  模型: ${MODEL}"
echo "  数据: ${DATA_YAML}"
echo "  输出: ${SAVE_DIR}"
echo "  显存优化: RTX 4080 12GB"
echo "  环境: $(which python)"
echo ""

# 开始训练
# 配置说明:
#   model=yolo11m-seg.pt   - Medium 版本, 27M 参数, 精度与速度平衡
#   imgsz=640              - 输入分辨率, 12GB 显存下的最优选择
#   batch=8                - 批次大小, 12GB 显存可支撑
#   freeze=10              - 冻结前10层(backbone), 防止小数据集过拟合
#   optimizer=AdamW        - 适合小数据集微调
#   lr0=0.001              - 较低学习率, 保护预训练权重
#   lrf=0.01               - 最终学习率 = lr0 * lrf
#   epochs=150             - 训练轮次
#   patience=30            - 早停耐心值, 30轮无提升则停止
#   augment=True           - 数据增强, 小数据集必须开启
#   mosaic=1.0             - Mosaic 增强概率
#   mixup=0.1              - MixUp 增强概率
#   copy_paste=0.1         - 分割任务专用增强
#   workers=4              - 数据加载线程数
#   device=0               - 使用 GPU 0

yolo segment train \
    model="${MODEL}" \
    data="${DATA_YAML}" \
    imgsz=1280 \
    batch=8 \
    epochs=150 \
    patience=30 \
    freeze=10 \
    optimizer=AdamW \
    lr0=0.001 \
    lrf=0.01 \
    momentum=0.937 \
    weight_decay=0.0005 \
    warmup_epochs=3 \
    warmup_momentum=0.8 \
    augment=True \
    mosaic=1.0 \
    mixup=0.1 \
    copy_paste=0.1 \
    workers=4 \
    device=0 \
    project="${SAVE_DIR}" \
    name="yolo11m_seg_military" \
    exist_ok=True \
    pretrained=True \
    verbose=True \
    seed=42 \
    val=True \
    save=True \
    save_period=10 \
    plots=True

echo ""
echo "============================================"
echo "训练完成!"
echo "权重保存在: ${SAVE_DIR}/yolo11m_seg_military/weights/"
echo "============================================"

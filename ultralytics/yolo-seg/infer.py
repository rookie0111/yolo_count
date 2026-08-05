#!/usr/bin/env python3
"""
YOLO11m-seg 推理可视化脚本

功能:
1. 加载训练好的模型
2. 对验证集图片进行推理
3. 保存可视化结果(分割掩码+边界框+类别标签)
4. 生成精度评估报告

使用方法:
    python infer.py
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


def plot_results(image, result, class_names):
    """
    在图片上绘制推理结果
    
    Args:
        image: 原始图片 (BGR)
        result: YOLO 推理结果
        class_names: 类别名称字典
    
    Returns:
        annotated_image: 绘制了结果的图片
    """
    annotated = image.copy()
    
    # 定义颜色 (BGR格式)
    colors = [
        (0, 255, 0),    # 绿色 - 类别0
        (0, 0, 255),    # 红色 - 类别1
        (255, 0, 0),    # 蓝色 - 类别2
        (0, 255, 255),  # 黄色 - 类别3
        (255, 0, 255),  # 紫色 - 类别4
    ]
    
    if result.masks is None:
        return annotated
    
    masks = result.masks.data.cpu().numpy()
    boxes = result.boxes
    
    for i, (mask, box) in enumerate(zip(masks, boxes)):
        # 获取类别和置信度
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        cls_name = class_names.get(cls_id, str(cls_id))
        color = colors[cls_id % len(colors)]
        
        # 绘制半透明掩码
        mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
        mask_binary = (mask_resized > 0.5).astype(np.uint8)
        
        # 创建彩色掩码
        colored_mask = np.zeros_like(image)
        colored_mask[mask_binary == 1] = color
        
        # 混合原图和掩码 (透明度 0.4)
        alpha = 0.4
        mask_region = mask_binary == 1
        annotated[mask_region] = (
            annotated[mask_region] * (1 - alpha) + colored_mask[mask_region] * alpha
        ).astype(np.uint8)
        
        # 绘制掩码轮廓
        contours, _ = cv2.findContours(
            mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(annotated, contours, -1, color, 2)
        
        # 绘制边界框
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # 绘制标签背景
        label = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 5, y1), color, -1)
        
        # 绘制标签文字
        cv2.putText(
            annotated, label, (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
        )
    
    return annotated


def main():
    # ============================================================
    # 路径配置
    # ============================================================
    PROJECT_DIR = "/home/finn/ultralytics-main-count/ultralytics/yolo-seg"
    DATA_DIR = "/mnt/c/Users/finn.zhang/project/Datasets/xutong_military_yolo_seg"
    
    # 自动查找最新训练结果
    weights_dir = Path(PROJECT_DIR) / "weights" / "yolo11m_seg_military"
    model_path = weights_dir / "weights" / "best.pt"
    
    # 如果 best.pt 不存在, 尝试 last.pt
    if not model_path.exists():
        model_path = weights_dir / "weights" / "last.pt"
    
    if not model_path.exists():
        print(f"[错误] 未找到训练权重: {model_path}")
        print(f"请先运行 train.sh 完成训练")
        sys.exit(1)
    
    # 输出目录
    output_dir = Path(PROJECT_DIR) / "results" / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 验证集图片目录
    val_img_dir = Path(DATA_DIR) / "images" / "val"
    train_img_dir = Path(DATA_DIR) / "images" / "train"
    
    # 使用本地 data.yaml（WSL 路径）
    data_yaml = Path(PROJECT_DIR) / "data.yaml"
    
    print("=" * 60)
    print("YOLO11m-seg 推理可视化")
    print("=" * 60)
    print(f"模型: {model_path}")
    print(f"输出: {output_dir}")
    print()
    
    # ============================================================
    # 加载模型
    # ============================================================
    print("[1/4] 加载模型...")
    model = YOLO(str(model_path))
    class_names = model.names
    print(f"  类别: {class_names}")
    
    # ============================================================
    # 在验证集上评估
    # ============================================================
    print("\n[2/4] 验证集评估...")
    metrics = model.val(
        data=str(data_yaml),
        imgsz=640,
        batch=1,
        device=0,
        plots=True,
        project=str(output_dir),
        name="metrics",
        exist_ok=True,
    )
    
    print(f"  mAP50:    {metrics.seg.map50:.4f}")
    print(f"  mAP50-95: {metrics.seg.map:.4f}")
    
    # ============================================================
    # 推理验证集图片并保存可视化
    # ============================================================
    print("\n[3/4] 推理验证集图片...")
    
    val_images = sorted(val_img_dir.glob("*"))
    val_images = [f for f in val_images if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
    
    for img_path in val_images:
        # 读取图片
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  [跳过] 无法读取: {img_path.name}")
            continue
        
        # 推理
        results = model.predict(
            source=image,
            imgsz=640,
            conf=0.75,
            iou=0.5,
            agnostic_nms=True,
            device=0,
            verbose=False,
        )
        
        # 绘制结果
        result = results[0]
        annotated = plot_results(image, result, class_names)
        
        # 保存结果
        save_path = output_dir / f"val_{img_path.stem}.jpg"
        cv2.imwrite(str(save_path), annotated)
        print(f"  [OK] {img_path.name} -> {save_path.name}")
    
    # ============================================================
    # 推理训练集图片并保存可视化
    # ============================================================
    print("\n[4/4] 推理训练集图片...")
    
    train_images = sorted(train_img_dir.glob("*"))
    train_images = [f for f in train_images if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
    
    for img_path in train_images:
        # 读取图片
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  [跳过] 无法读取: {img_path.name}")
            continue
        
        # 推理
        results = model.predict(
            source=image,
            imgsz=640,
            conf=0.75,
            iou=0.5,
            agnostic_nms=True,
            device=0,
            verbose=False,
        )
        
        # 绘制结果
        result = results[0]
        annotated = plot_results(image, result, class_names)
        
        # 保存结果
        save_path = output_dir / f"train_{img_path.stem}.jpg"
        cv2.imwrite(str(save_path), annotated)
        print(f"  [OK] {img_path.name} -> {save_path.name}")
    
    # ============================================================
    # 完成
    # ============================================================
    print()
    print("=" * 60)
    print("推理完成!")
    print("=" * 60)
    print(f"可视化结果保存在: {output_dir}")
    print(f"  - val_*.jpg    验证集推理结果")
    print(f"  - train_*.jpg  训练集推理结果")
    print(f"  - metrics/     精度评估报告")


if __name__ == "__main__":
    main()

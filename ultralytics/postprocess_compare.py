"""
YOLO OBB (旋转目标检测) 和 Box (普通目标检测) 后处理区别.
=========================================================

1. Box后处理: 检测结果是水平框 (x1, y1, x2, y2)
2. OBB后处理: 检测结果是旋转框 (x1, y1, x2, y2, x3, y3, x4, y4) 或 (cx, cy, w, h, angle)
"""

import numpy as np


# ==========================================================
# 1. 普通检测框 (Box) 后处理
# ==========================================================
def postprocess_boxes(preds, conf_thresh=0.25, nms_thresh=0.45):
    """普通检测框后处理.

    输入模型输出: (batch, 4+nc, num_anchors)
    - 4: xywh (中心点x, 中心点y, 宽, 高)
    - nc: 每个类别的置信度

    输出: 每个检测框 [x1, y1, x2, y2, conf, cls]
    """
    # 假设 preds 形状: (batch, 5, num_anchors) -> 4(xywh) + 1(obj) + 0(class)
    # 这里简化处理，只输出 xyxy 格式

    for batch_idx in range(preds.shape[0]):
        pred = preds[batch_idx]  # (5, num_anchors)

        # 1. 应用sigmoid得到obj概率
        obj = 1 / (1 + np.exp(-pred[4, :]))

        # 2. 过滤低置信度
        mask = obj > conf_thresh
        pred_filtered = pred[:, mask]

        if pred_filtered.shape[1] == 0:
            continue

        # 3. xywh 转 xyxy (中心点 -> 左上右下)
        xywh = pred_filtered[:4, :]
        x1 = xywh[0] - xywh[2] / 2  # cx - w/2
        y1 = xywh[1] - xywh[3] / 2  # cy - h/2
        x2 = xywh[0] + xywh[2] / 2  # cx + w/2
        y2 = xywh[1] + xywh[3] / 2  # cy + h/2

        # 4. 构建输出 [x1, y1, x2, y2, conf, cls]
        boxes = np.column_stack([x1, y1, x2, y2, obj[mask]])

        # 5. NMS非极大值抑制
        boxes = nms_boxes(boxes, nms_thresh)

    return boxes


def nms_boxes(boxes, iou_thresh):
    """NMS非极大值抑制.

    原理:
    1. 按置信度排序
    2. 保留置信度最高的框
    3. 删除与最高置信度框IOU重叠过大的框
    4. 重复直到处理完所有框
    """
    if len(boxes) == 0:
        return boxes

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]

        order = order[inds + 1]

    return boxes[keep]


# ==========================================================
# 2. OBB (旋转目标检测) 后处理
# ==========================================================
def postprocess_obb(preds, conf_thresh=0.25, nms_thresh=0.45):
    """OBB旋转框后处理.

    输入模型输出: (batch, 5+nc, num_anchors)
    - 5: xywhr (中心点x, 中心点y, 宽, 高, 角度)
    - nc: 每个类别的置信度

    输出: 每个检测框 [x1, y1, x2, y2, x3, y3, x4, y4, conf, cls]
    - 旋转框的4个顶点坐标
    """
    for batch_idx in range(preds.shape[0]):
        pred = preds[batch_idx]

        # 1. 应用sigmoid得到obj概率
        obj = 1 / (1 + np.exp(-pred[4, :]))

        # 2. 过滤低置信度
        mask = obj > conf_thresh
        pred_filtered = pred[:, mask]

        if pred_filtered.shape[1] == 0:
            continue

        # 3. xywhr 转 4个顶点坐标
        xywhr = pred_filtered[:5, :]  # cx, cy, w, h, angle
        boxes_xyxyxyxy = xywhr_to_xyxyxyxy(xywhr)

        # 4. 构建输出 (包含4个顶点)
        boxes = np.column_stack([boxes_xyxyxyxy, obj[mask]])

        # 5. OBB-NMS (使用旋转IoU)
        boxes = nms_obb(boxes, nms_thresh)

    return boxes


def xywhr_to_xyxyxyxy(xywhr):
    """将 xywhr 转换为 4个顶点坐标 (xyxyxyxy).

    参数:
        xywhr: [cx, cy, w, h, angle] - 中心点x,y + 宽高 + 角度(度)

    返回:
        pts: (4, num_boxes) - 4个顶点的x,y坐标
    """
    cx, cy, w, h, angle = xywhr
    angle_rad = np.deg2rad(angle)

    # 计算旋转矩阵
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # 相对于中心的4个顶点 (未旋转)
    dx = np.array([-w / 2, w / 2, w / 2, -w / 2])
    dy = np.array([-h / 2, -h / 2, h / 2, h / 2])

    # 旋转并平移到绝对坐标
    pts = np.zeros((8, len(cx)))
    for i in range(4):
        pts[2 * i] = cx + dx[i] * cos_a - dy[i] * sin_a
        pts[2 * i + 1] = cy + dx[i] * sin_a + dy[i] * cos_a

    return pts


def nms_obb(boxes, iou_thresh):
    """OBB-NMS (旋转框非极大值抑制).

    与普通NMS的区别:
    - 普通NMS: 计算轴对齐IOU (axis-aligned IOU)
    - OBB-NMS: 计算旋转IOU (rotated IOU)

    旋转IOU计算:
    1. 用最小外接矩形描述两个旋转框
    2. 使用 cv2.minAreaRect 获取矩形
    3. 使用 cv2.rotatedRectangleIntersection 计算交集
    """
    if len(boxes) == 0:
        return boxes

    keep = []
    order = boxes[:, 8].argsort()[::-1]  # 按置信度排序

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        # 计算旋转IOU
        ious = []
        for j in order[1:]:
            iou = rotated_iou(boxes[i], boxes[j])
            ious.append(iou)

        ious = np.array(ious)
        inds = np.where(ious <= iou_thresh)[0]
        order = order[inds + 1]

    return boxes[keep]


def rotated_iou(box1, box2):
    """计算两个旋转框的IoU.

    参数:
        box1, box2: [x1,y1,x2,y2,x3,y3,x4,y4,conf,cls] - 4个顶点+置信度+类别

    返回:
        iou: IoU值
    """
    import cv2

    # 提取4个顶点
    pts1 = box1[:8].reshape(4, 2).astype(np.float32)
    pts2 = box2[:8].reshape(4, 2).astype(np.float32)

    # 获取最小外接矩形
    rect1 = cv2.minAreaRect(pts1)
    rect2 = cv2.minAreaRect(pts2)

    # 获取轮廓
    cv2.boxPoints(rect1)
    cv2.boxPoints(rect2)

    # 计算交集
    try:
        _, contours = cv2.rotatedRectangleIntersection(rect1, rect2)

        if contours is None or len(contours) == 0:
            return 0.0

        intersection = cv2.contourArea(contours[0])

        # 计算并集 (近似: 两矩形面积之和 - 交集)
        area1 = rect1[1][0] * rect1[1][1]
        area2 = rect2[1][0] * rect2[1][1]
        union = area1 + area2 - intersection

        iou = intersection / union if union > 0 else 0
    except:
        iou = 0

    return iou


# ==========================================================
# 3. 坐标转换到原图尺寸
# ==========================================================
def scale_boxes_to_orig(img_shape, boxes, ratio_pad):
    """将归一化的框坐标转换到原图尺寸.

    参数:
        img_shape: (h, w) 预处理后的图像尺寸
        boxes: 检测框 [x1,y1,x2,y2,...] 归一化坐标 (0-1)
        ratio_pad: (gain, (pad_x, pad_y)) 缩放比例和填充

    返回:
        boxes: 原图尺寸的坐标
    """
    gain = ratio_pad[0]
    pad = ratio_pad[1]

    # 去掉padding，乘以缩放比例
    boxes[..., [0, 2]] -= pad[0]  # x轴去padding
    boxes[..., [1, 3]] -= pad[1]  # y轴去padding
    boxes[..., :4] /= gain  # 缩放到原图尺寸

    return boxes


if __name__ == "__main__":
    print("=" * 60)
    print("YOLO 后处理说明")
    print("=" * 60)
    print("""
Box后处理流程:
  1. sigmoid(obj) → 置信度
  2. 过滤 conf < thresh
  3. xywh → xyxy (中心点转左上右下)
  4. NMS (轴对齐IOU)
  5. scale_boxes (去padding，乘gain)

OBB后处理流程:
  1. sigmoid(obj) → 置信度
  2. 过滤 conf < thresh
  3. xywhr → xyxyxyxy (4个顶点)
  4. OBB-NMS (旋转IOU)
  5. scale_boxes (去padding，乘gain)
    """)

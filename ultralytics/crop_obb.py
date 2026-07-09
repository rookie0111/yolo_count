import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image


def order_points(pts):
    """
    对4个点进行排序:
    返回顺序: [top-left, top-right, bottom-right, bottom-left]
    """
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def crop_obb_perspective(image, polygon):
    """
    对 OBB 四点框进行透视裁剪并拉正
    image: cv2读取的BGR图
    polygon: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    """
    pts = order_points(polygon)

    (tl, tr, br, bl) = pts

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    if maxWidth <= 1 or maxHeight <= 1:
        return None

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def crop_obb_bbox(image_pil, polygon):
    """
    OBB四点框的外接水平矩形裁剪
    image_pil: PIL.Image
    polygon: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    """
    pts = np.array(polygon, dtype=np.float32)
    x_min = max(int(np.min(pts[:, 0])), 0)
    y_min = max(int(np.min(pts[:, 1])), 0)
    x_max = min(int(np.max(pts[:, 0])), image_pil.width)
    y_max = min(int(np.max(pts[:, 1])), image_pil.height)

    if x_max <= x_min or y_max <= y_min:
        return None

    return image_pil.crop((x_min, y_min, x_max, y_max))


def crop_single_image(image_path, model, output_folder, conf=0.25, use_obb=True, obb_mode="perspective"):
    """
    只裁剪置信度最高的一个检测框，没有目标则不裁剪

    use_obb=True 时优先使用 OBB 结果裁剪
    obb_mode:
        - 'perspective': 对旋转框做透视矫正裁剪
        - 'bbox': 用旋转框外接矩形裁剪
    """
    img_pil = Image.open(image_path).convert("RGB")
    img_cv = cv2.imread(image_path)  # BGR
    img_name = os.path.basename(image_path)
    name, ext = os.path.splitext(img_name)

    results = model.predict(img_cv, conf=conf, verbose=False, end2end=False)
    saved_count = 0

    for result in results:
        # =========================
        # 1. 优先处理 OBB
        # =========================
        if use_obb and hasattr(result, "obb") and result.obb is not None and len(result.obb) > 0:
            obb = result.obb

            if hasattr(obb, "xyxyxyxy") and obb.xyxyxyxy is not None:
                polygons = obb.xyxyxyxy.cpu().numpy()
            else:
                polygons = None

            cls_list = obb.cls.cpu().numpy().astype(int) if hasattr(obb, "cls") and obb.cls is not None else None

            # 获取所有置信度，找最高的
            if hasattr(obb, "conf") and obb.conf is not None:
                confs = obb.conf.cpu().numpy()
            else:
                confs = np.ones(len(obb))

            if len(confs) == 0:
                print(f"No detections in {img_name}")
                continue

            # 找置信度最高的框
            best_idx = int(np.argmax(confs))
            best_conf = float(confs[best_idx])

            polygon = polygons[best_idx] if polygons is not None else None
            if polygon is None:
                continue

            cls_id = int(cls_list[best_idx]) if cls_list is not None else 0
            cls_name = result.names.get(cls_id, str(cls_id)) if isinstance(result.names, dict) else str(cls_id)

            if obb_mode == "perspective":
                crop = crop_obb_perspective(img_cv, polygon)
                if crop is None:
                    return 0
                save_name = f"{name}.png"
                save_path = os.path.join(output_folder, save_name)
                cv2.imwrite(save_path, crop)
            else:
                crop = crop_obb_bbox(img_pil, polygon)
                if crop is None:
                    return 0
                save_name = f"{name}"
                save_path = os.path.join(output_folder, save_name)
                crop.save(save_path)

            saved_count = 1
            print(f"Saved OBB crop (conf={best_conf:.3f}): {save_path}")
            return saved_count

        # =========================
        # 2. 如果没有 OBB，就处理普通 boxes
        # =========================
        elif hasattr(result, "boxes") and result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes

            # 获取所有置信度，找最高的
            if hasattr(boxes, "conf") and boxes.conf is not None:
                confs = boxes.conf.cpu().numpy()
            else:
                confs = np.ones(len(boxes))

            if len(confs) == 0:
                print(f"No detections in {img_name}")
                continue

            # 找置信度最高的框
            best_idx = int(np.argmax(confs))
            best_conf = float(confs[best_idx])

            box = boxes[best_idx]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = min(x2, img_pil.width)
            y2 = min(y2, img_pil.height)

            if x2 <= x1 or y2 <= y1:
                return 0

            crop = img_pil.crop((x1, y1, x2, y2))

            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]

            save_name = f"{name}_{cls_name}.png"
            save_path = os.path.join(output_folder, save_name)
            crop.save(save_path)

            saved_count = 1
            print(f"Saved box crop (conf={best_conf:.3f}): {save_path}")
            return saved_count
        else:
            print(f"No detections in {img_name}")

    return saved_count


def crop_predictions(input_path, model_path, output_folder, conf=0.25, use_obb=True, obb_mode="perspective", batch_size=8, device="0"):
    """
    批量处理模式，加速推理

    Args:
        batch_size: 每批处理图片数量，越大越快但越耗内存
        device: GPU设备号，"0"表示第一块GPU，"cpu"表示CPU
    """
    os.makedirs(output_folder, exist_ok=True)

    model = YOLO(model_path)

    total = 0
    if os.path.isdir(input_path):
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

        # 收集所有图片路径
        img_paths = []
        for img_file in sorted(os.listdir(input_path)):
            if os.path.splitext(img_file.lower())[1] in extensions:
                img_paths.append(os.path.join(input_path, img_file))

        # 批量处理
        for i in range(0, len(img_paths), batch_size):
            batch_paths = img_paths[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(img_paths)-1)//batch_size + 1} ({len(batch_paths)} images)")

            # 批量推理，device参数传给predict
            results = model.predict(batch_paths, conf=conf, verbose=False, end2end=False, device=device)

            # 逐个处理结果
            for img_path, result in zip(batch_paths, results):
                img_name = os.path.basename(img_path)
                name, ext = os.path.splitext(img_name)
                saved = 0

                img_pil = Image.open(img_path).convert("RGB")
                img_cv = cv2.imread(img_path)

                # 处理 OBB
                if use_obb and hasattr(result, "obb") and result.obb is not None and len(result.obb) > 0:
                    obb = result.obb
                    if hasattr(obb, "conf") and obb.conf is not None:
                        confs = obb.conf.cpu().numpy()
                    else:
                        confs = np.ones(len(obb))

                    if len(confs) > 0:
                        best_idx = int(np.argmax(confs))
                        best_conf = float(confs[best_idx])

                        if hasattr(obb, "xyxyxyxy") and obb.xyxyxyxy is not None:
                            polygons = obb.xyxyxyxy.cpu().numpy()
                            polygon = polygons[best_idx]

                            if obb_mode == "perspective":
                                crop = crop_obb_perspective(img_cv, polygon)
                            else:
                                crop = crop_obb_bbox(img_pil, polygon)

                            if crop is not None:
                                save_name = f"{name}_obb.png"
                                save_path = os.path.join(output_folder, save_name)
                                cv2.imwrite(save_path, crop)
                                saved = 1
                                print(f"  {img_name} -> {save_name} (conf={best_conf:.3f})")

                # 处理普通 boxes
                elif hasattr(result, "boxes") and result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes
                    if hasattr(boxes, "conf") and boxes.conf is not None:
                        confs = boxes.conf.cpu().numpy()
                    else:
                        confs = np.ones(len(boxes))

                    if len(confs) > 0:
                        best_idx = int(np.argmax(confs))
                        best_conf = float(confs[best_idx])
                        box = boxes[best_idx]

                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        x1 = max(x1, 0)
                        y1 = max(y1, 0)
                        x2 = min(x2, img_pil.width)
                        y2 = min(y2, img_pil.height)

                        if x2 > x1 and y2 > y1:
                            crop = img_pil.crop((x1, y1, x2, y2))
                            save_name = f"{name}.png"
                            save_path = os.path.join(output_folder, save_name)
                            crop.save(save_path)
                            saved = 1
                            print(f"  {img_name} -> {save_name} (conf={best_conf:.3f})")

                total += saved

    print(f"Total cropped: {total}")
    return total


if __name__ == "__main__":
    input_path = r"/mnt/c/Users/finn.zhang/project/Datasets/20260626_yellow_180ml/NG"
    model_path = r"/home/finn/ultralytics-main/runs/obb/blue_bottle/weights/best.pt"
    output_folder = r"/mnt/c/Users/finn.zhang/project/Datasets/20260626_yellow_180ml/NG_crop"
    conf = 0.25

    # obb_mode:
    #   "perspective" -> 旋转框拉正裁剪
    #   "bbox"        -> 旋转框外接矩形裁剪
    crop_predictions(
        input_path=input_path,
        model_path=model_path,
        output_folder=output_folder,
        conf=conf,
        use_obb=True,
        obb_mode="perspective",
        batch_size=16,
        device="0"
    )


# if __name__ == "__main__":
#     input_path = r"/mnt/c/Users/finn.zhang/project/Datasets/NG_3/NG"
#     model_path = r"/home/finn/ultralytics-main/runs/obb/train/weights/best.pt"
#     output_folder = r"/mnt/c/Users/finn.zhang/project/Datasets/NG_3/NG_crop_obb"
#     conf = 0.25
#     margin = 0
    
#     crop_predictions(input_path, model_path, output_folder, conf, margin)

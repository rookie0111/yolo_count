import os
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def order_points(pts):
    """
    对4个点排序:
    返回顺序: [top-left, top-right, bottom-right, bottom-left]
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def crop_obb_perspective_cv(image, polygon):
    """
    OBB四点透视拉正裁剪。
    image: OpenCV BGR图像
    polygon: shape [4, 2] 或 [8]
    """
    pts = order_points(polygon)

    tl, tr, br, bl = pts

    width_a = np.hypot(*(br - bl))
    width_b = np.hypot(*(tr - tl))
    max_width = int(max(width_a, width_b))

    height_a = np.hypot(*(tr - br))
    height_b = np.hypot(*(tl - bl))
    max_height = int(max(height_a, height_b))

    if max_width <= 1 or max_height <= 1:
        return None

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )

    M = cv2.getPerspectiveTransform(pts, dst)

    crop = cv2.warpPerspective(
        image,
        M,
        (max_width, max_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    return crop


def crop_obb_bbox_cv(image, polygon):
    """
    OBB四点框外接水平矩形裁剪。
    """
    h, w = image.shape[:2]
    pts = np.asarray(polygon, dtype=np.float32).reshape(4, 2)

    x_min = max(int(np.floor(pts[:, 0].min())), 0)
    y_min = max(int(np.floor(pts[:, 1].min())), 0)
    x_max = min(int(np.ceil(pts[:, 0].max())), w)
    y_max = min(int(np.ceil(pts[:, 1].max())), h)

    if x_max <= x_min or y_max <= y_min:
        return None

    return image[y_min:y_max, x_min:x_max]


def crop_xyxy_cv(image, xyxy):
    """
    普通水平框裁剪。
    """
    h, w = image.shape[:2]

    x1, y1, x2, y2 = xyxy

    x1 = max(int(np.floor(x1)), 0)
    y1 = max(int(np.floor(y1)), 0)
    x2 = min(int(np.ceil(x2)), w)
    y2 = min(int(np.ceil(y2)), h)

    if x2 <= x1 or y2 <= y1:
        return None

    return image[y1:y2, x1:x2]


def save_image_same_format(save_path, image, original_ext, png_compression=1, jpg_quality=95):
    """
    按原图格式保存。

    original_ext:
        .jpg / .jpeg / .png / .bmp / .tif / .tiff
    """
    save_path = str(save_path)
    original_ext = original_ext.lower()

    if original_ext == ".png":
        return cv2.imwrite(
            save_path,
            image,
            [cv2.IMWRITE_PNG_COMPRESSION, png_compression],
        )

    elif original_ext in {".jpg", ".jpeg"}:
        return cv2.imwrite(
            save_path,
            image,
            [cv2.IMWRITE_JPEG_QUALITY, jpg_quality],
        )

    elif original_ext == ".bmp":
        return cv2.imwrite(save_path, image)

    elif original_ext in {".tif", ".tiff"}:
        return cv2.imwrite(save_path, image)

    else:
        # 理论上不会走到这里，因为前面已经过滤了格式
        return cv2.imwrite(save_path, image)


def process_one_result(
    result,
    original_img_path,
    output_folder,
    use_obb=True,
    obb_mode="perspective",
    png_compression=1,
    jpg_quality=95,
):
    """
    处理单张图片预测结果。

    重点:
        不再使用 result.path 生成输出名。
        必须使用 original_img_path，避免 image0.jpg/image1.jpg/image2.jpg 问题。

    Args:
        result: YOLO预测结果
        original_img_path: 原始图片真实路径
        output_folder: 输出文件夹
    """
    original_img_path = Path(original_img_path)

    # 输出文件名严格等于原图片文件名
    save_name = original_img_path.name
    save_path = Path(output_folder) / save_name

    # 原图格式
    original_ext = original_img_path.suffix.lower()

    # 使用 Ultralytics 已加载的原图
    image = result.orig_img

    # 如果 result.orig_img 异常，则用原始路径重新读取
    if image is None:
        image = cv2.imread(str(original_img_path), cv2.IMREAD_COLOR)

    if image is None:
        print(f"Failed to read image: {original_img_path}")
        return 0

    # =========================
    # 1. 优先处理 OBB
    # =========================
    if use_obb and hasattr(result, "obb") and result.obb is not None and len(result.obb) > 0:
        obb = result.obb

        if not hasattr(obb, "xyxyxyxy") or obb.xyxyxyxy is None:
            return 0

        if hasattr(obb, "conf") and obb.conf is not None:
            best_idx = int(torch.argmax(obb.conf).item())
        else:
            best_idx = 0

        polygon = obb.xyxyxyxy[best_idx].detach().cpu().numpy().reshape(4, 2)

        if obb_mode == "perspective":
            crop = crop_obb_perspective_cv(image, polygon)
        elif obb_mode == "bbox":
            crop = crop_obb_bbox_cv(image, polygon)
        else:
            raise ValueError(f"Unsupported obb_mode: {obb_mode}")

        if crop is None:
            return 0

        ok = save_image_same_format(
            save_path=save_path,
            image=crop,
            original_ext=original_ext,
            png_compression=png_compression,
            jpg_quality=jpg_quality,
        )

        return 1 if ok else 0

    # =========================
    # 2. 普通 boxes
    # =========================
    if hasattr(result, "boxes") and result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes

        if hasattr(boxes, "conf") and boxes.conf is not None:
            best_idx = int(torch.argmax(boxes.conf).item())
        else:
            best_idx = 0

        xyxy = boxes.xyxy[best_idx].detach().cpu().numpy()

        crop = crop_xyxy_cv(image, xyxy)

        if crop is None:
            return 0

        ok = save_image_same_format(
            save_path=save_path,
            image=crop,
            original_ext=original_ext,
            png_compression=png_compression,
            jpg_quality=jpg_quality,
        )

        return 1 if ok else 0

    return 0


def collect_image_paths(input_path):
    """
    收集图片路径。
    """
    input_path = Path(input_path)

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
    }

    if input_path.is_dir():
        img_paths = [
            str(p)
            for p in sorted(input_path.iterdir())
            if p.suffix.lower() in extensions
        ]
    elif input_path.is_file() and input_path.suffix.lower() in extensions:
        img_paths = [str(input_path)]
    else:
        img_paths = []

    return img_paths


def crop_predictions(
    input_path,
    model_path,
    output_folder,
    conf=0.25,
    use_obb=True,
    obb_mode="perspective",
    batch_size=8,
    device="0",
    imgsz=640,
    half=False,
    png_compression=1,
    jpg_quality=95,
    print_interval=20,
):
    """
    批量预测并裁剪。

    保证:
        1. 输出文件名与原图完全相同
        2. 输出图片格式与原图完全相同
        3. 不会出现 image0.jpg / image1.jpg / image2.jpg 这类内部名称
    """
    os.makedirs(output_folder, exist_ok=True)

    img_paths = collect_image_paths(input_path)

    print(f"Found images: {len(img_paths)}")

    if len(img_paths) == 0:
        print("No images found.")
        return 0

    print("Loading model...")
    model = YOLO(model_path)
    print("Model loaded.")

    if device != "cpu":
        torch.backends.cudnn.benchmark = True

    total = 0
    processed = 0

    num_batches = (len(img_paths) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(img_paths))
        batch_paths = img_paths[start:end]

        print(f"Predicting batch {batch_idx + 1}/{num_batches}, images {start}~{end - 1}")

        predict_kwargs = dict(
            source=batch_paths,
            conf=conf,
            batch=len(batch_paths),
            device=device,
            verbose=False,
        )

        if imgsz is not None:
            predict_kwargs["imgsz"] = imgsz

        if device != "cpu":
            predict_kwargs["half"] = half

        try:
            results = model.predict(**predict_kwargs)
        except Exception as e:
            print(f"Predict failed at batch {batch_idx + 1}: {e}")
            continue

        # 关键修改:
        # 用 zip(batch_paths, results)，把真实原图路径传入 process_one_result
        for original_img_path, result in zip(batch_paths, results):
            try:
                saved = process_one_result(
                    result=result,
                    original_img_path=original_img_path,
                    output_folder=output_folder,
                    use_obb=use_obb,
                    obb_mode=obb_mode,
                    png_compression=png_compression,
                    jpg_quality=jpg_quality,
                )
            except Exception as e:
                print(f"Process failed: {original_img_path}, error: {e}")
                saved = 0

            total += saved
            processed += 1

            if print_interval and processed % print_interval == 0:
                print(f"Processed {processed}/{len(img_paths)}, cropped {total}")

    print(f"Done. Processed: {processed}, Total cropped: {total}")
    return total


if __name__ == "__main__":
    input_path = r"/mnt/c/Users/finn.zhang/project/Datasets/20260506_yellow_120ml/NG"
    model_path = r"/home/finn/ultralytics-main/runs/obb/train4/weights/best.pt"
    output_folder = r"/mnt/c/Users/finn.zhang/project/Datasets/20260506_yellow_120ml/NG_crop"

    crop_predictions(
        input_path=input_path,
        model_path=model_path,
        output_folder=output_folder,
        conf=0.25,
        use_obb=True,
        obb_mode="perspective",
        batch_size=32,
        device="0",
        imgsz=640,
        half=False,
        png_compression=0,
        jpg_quality=95,
        print_interval=20,
    )

import argparse
import logging
import os
import time

import cv2
import numpy as np
import onnxruntime as ort

logging.basicConfig(level=logging.INFO)

COLORS = [
    [56, 0, 255],
    [226, 255, 0],
    [0, 94, 255],
    [0, 37, 255],
    [0, 255, 94],
    [255, 226, 0],
    [0, 18, 255],
    [255, 151, 0],
    [170, 0, 255],
    [0, 255, 56],
    [255, 0, 75],
    [0, 75, 255],
    [0, 255, 169],
    [255, 0, 207],
    [75, 255, 0],
]


class PostProcess:
    def __init__(
        self, conf_thresh=0.25, nms_thresh=0.45, nc=1, agnostic=False, multi_label=False, max_det=300, task="detect"
    ):
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.nc = nc
        self.agnostic = agnostic
        self.multi_label = multi_label
        self.max_det = max_det
        self.max_wh = 7680
        self.task = task

    def xywh2xyxy(self, x):
        """Convert xywh to xyxy format."""
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y

    def nms(self, boxes, scores):
        """Axis-aligned NMS."""
        if len(boxes) == 0:
            return []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
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

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

            inds = np.where(iou <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep

    def rotated_iou(self, box1, box2):
        """计算两个旋转框 IoU box: [cx, cy, w, h, angle(rad)].
        """
        rect1 = (
            (float(box1[0]), float(box1[1])),
            (max(float(box1[2]), 1e-6), max(float(box1[3]), 1e-6)),
            float(box1[4] * 180.0 / np.pi),
        )
        rect2 = (
            (float(box2[0]), float(box2[1])),
            (max(float(box2[2]), 1e-6), max(float(box2[3]), 1e-6)),
            float(box2[4] * 180.0 / np.pi),
        )

        try:
            _inter_type, inter_pts = cv2.rotatedRectangleIntersection(rect1, rect2)
            if inter_pts is None:
                return 0.0

            inter_pts = cv2.convexHull(inter_pts, returnPoints=True)
            inter_area = cv2.contourArea(inter_pts)

            area1 = float(box1[2] * box1[3])
            area2 = float(box2[2] * box2[3])
            union = area1 + area2 - inter_area + 1e-6
            return inter_area / union
        except Exception:
            return 0.0

    def nms_rotated(self, rboxes, scores):
        """Rotated NMS rboxes: [N, 5] => cx, cy, w, h, angle(rad).
        """
        if len(rboxes) == 0:
            return []

        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break

            remain = order[1:]
            ious = []
            for j in remain:
                iou = self.rotated_iou(rboxes[i], rboxes[j])
                ious.append(iou)
            ious = np.array(ious, dtype=np.float32)

            inds = np.where(ious <= self.nms_thresh)[0]
            order = remain[inds]

        return keep

    def scale_boxes(self, boxes, img0_shape, ratio_pad):
        """Rescale xyxy boxes from letterbox image to original image."""
        if len(boxes) == 0:
            return boxes

        gain = ratio_pad[0]
        pad = ratio_pad[1]

        boxes[:, [0, 2]] -= pad[0]
        boxes[:, [1, 3]] -= pad[1]
        boxes[:, :4] /= gain

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img0_shape[1])
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img0_shape[0])
        return boxes

    def scale_rboxes(self, rboxes, img0_shape, ratio_pad):
        """Rescale xywhr boxes from letterbox image to original image."""
        if len(rboxes) == 0:
            return rboxes

        gain = ratio_pad[0]
        pad = ratio_pad[1]

        rboxes[:, 0] -= pad[0]
        rboxes[:, 1] -= pad[1]
        rboxes[:, 0:4] /= gain

        rboxes[:, 0] = np.clip(rboxes[:, 0], 0, img0_shape[1])
        rboxes[:, 1] = np.clip(rboxes[:, 1], 0, img0_shape[0])
        rboxes[:, 2] = np.clip(rboxes[:, 2], 1e-6, img0_shape[1])
        rboxes[:, 3] = np.clip(rboxes[:, 3], 1e-6, img0_shape[0])

        return rboxes

    def __call__(self, preds_batch, img0_batch, ratio_pad_batch):
        """detect: output -> [x1, y1, x2, y2, conf, cls] obb: output -> [cx, cy, w, h, conf, cls, angle].
        """
        if isinstance(preds_batch, list) and len(preds_batch) == 1:
            preds = preds_batch[0]
        else:
            raise NotImplementedError(f"Unsupported preds type: {type(preds_batch)}")

        bs = preds.shape[0]
        preds = np.transpose(preds, (0, 2, 1))  # [bs, num, channels]

        batch_outs = []

        if self.task == "detect":
            nc = self.nc
            mi = 4 + nc
            xc = preds[:, :, 4:mi].max(axis=2) > self.conf_thresh
            preds[..., :4] = self.xywh2xyxy(preds[..., :4])

            for xi in range(bs):
                pred = preds[xi]
                x = pred[xc[xi]]

                if len(x) == 0:
                    batch_outs.append(np.zeros((0, 6), dtype=np.float32))
                    continue

                box = x[:, :4]
                cls = x[:, 4 : 4 + nc]

                if self.multi_label:
                    i, j = np.where(cls > self.conf_thresh)
                    x = np.concatenate([box[i], x[i, 4 + j, None], j.astype(np.float32)[:, None]], axis=1)
                else:
                    conf = cls.max(axis=1, keepdims=True)
                    j = cls.argmax(axis=1, keepdims=True)
                    filt = conf.flatten() > self.conf_thresh
                    x = np.concatenate([box, conf, j.astype(np.float32)], axis=1)[filt]

                if len(x) == 0:
                    batch_outs.append(np.zeros((0, 6), dtype=np.float32))
                    continue

                if self.agnostic:
                    c = x[:, 5:6] * 0
                else:
                    c = x[:, 5:6] * self.max_wh

                boxes_for_nms = x[:, :4] + np.hstack([c, c, c, c])
                scores = x[:, 4]

                keep = self.nms(boxes_for_nms, scores)[: self.max_det]
                x = x[keep]

                img0_shape = img0_batch[xi]
                ratio_pad = ratio_pad_batch[xi]

                logging.info(f"[DET] gain={ratio_pad[0]}, pad={ratio_pad[1]}, orig={img0_shape[1]}x{img0_shape[0]}")
                logging.info(f"[DET] Before scale: {x[:3, :4]}")

                x[:, :4] = self.scale_boxes(x[:, :4].copy(), img0_shape, ratio_pad)

                logging.info(f"[DET] After scale: {x[:3, :4]}")
                batch_outs.append(x.astype(np.float32))

        elif self.task == "obb":
            # Ultralytics OBB ONNX 通常格式：
            # [cx, cy, w, h, cls1, cls2, ..., angle]
            nc = self.nc
            angle_idx = 4 + nc
            xc = preds[:, :, 4 : 4 + nc].max(axis=2) > self.conf_thresh

            for xi in range(bs):
                pred = preds[xi]
                x = pred[xc[xi]]

                if len(x) == 0:
                    batch_outs.append(np.zeros((0, 7), dtype=np.float32))
                    continue

                box = x[:, :4]  # cx, cy, w, h
                cls = x[:, 4 : 4 + nc]  # class scores
                angle = x[:, angle_idx : angle_idx + 1]  # rad

                if self.multi_label:
                    i, j = np.where(cls > self.conf_thresh)
                    x = np.concatenate([box[i], x[i, 4 + j, None], j.astype(np.float32)[:, None], angle[i]], axis=1)
                else:
                    conf = cls.max(axis=1, keepdims=True)
                    j = cls.argmax(axis=1, keepdims=True)
                    filt = conf.flatten() > self.conf_thresh
                    x = np.concatenate([box, conf, j.astype(np.float32), angle], axis=1)[filt]

                if len(x) == 0:
                    batch_outs.append(np.zeros((0, 7), dtype=np.float32))
                    continue

                # x = [cx, cy, w, h, conf, cls, angle]
                scores = x[:, 4]

                if self.agnostic:
                    c = x[:, 5:6] * 0
                else:
                    c = x[:, 5:6] * self.max_wh

                rboxes_for_nms = np.concatenate([x[:, 0:1] + c, x[:, 1:2] + c, x[:, 2:4], x[:, 6:7]], axis=1)

                keep = self.nms_rotated(rboxes_for_nms, scores)[: self.max_det]
                x = x[keep]

                img0_shape = img0_batch[xi]
                ratio_pad = ratio_pad_batch[xi]

                logging.info(f"[OBB] gain={ratio_pad[0]}, pad={ratio_pad[1]}, orig={img0_shape[1]}x{img0_shape[0]}")
                logging.info(f"[OBB] Before scale: {x[:3, :4]}, angle={x[:3, 6:7]}")

                x[:, :4] = self.scale_rboxes(x[:, :4].copy(), img0_shape, ratio_pad)

                logging.info(f"[OBB] After scale: {x[:3, :4]}, angle={x[:3, 6:7]}")
                batch_outs.append(x.astype(np.float32))
        else:
            raise ValueError(f"Unsupported task: {self.task}")

        return batch_outs


class YOLO26:
    def __init__(self, onnx_path, conf_thresh=0.25, nms_thresh=0.45, task="detect"):
        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_shape = self.session.get_outputs()[0].shape

        self.batch_size = self.input_shape[0] if self.input_shape[0] else 1
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]
        self.task = task

        # detect: channels = 4 + nc
        # obb:    channels = 5 + nc (多一个 angle)
        if self.task == "detect":
            self.nc = self.output_shape[1] - 4
        elif self.task == "obb":
            self.nc = self.output_shape[1] - 5
        else:
            raise ValueError(f"Unsupported task: {self.task}")

        logging.info(f"Load ONNX: {onnx_path}")
        logging.info(f"Task: {self.task}")
        logging.info(f"Input shape: {self.input_shape}")
        logging.info(f"Output shape: {self.output_shape}, nc={self.nc}")

        self.conf_thresh = conf_thresh
        self.postprocess = PostProcess(conf_thresh=conf_thresh, nms_thresh=nms_thresh, nc=self.nc, task=self.task)

        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114)):
        shape = im.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (round(shape[1] * r), round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        logging.info(f"letterbox: shape={shape}, r={r}, new_unpad={new_unpad}, dw={dw}, dh={dh}")
        return im, r, (dw, dh)

    def preprocess(self, ori_img):
        img, ratio, (dw, dh) = self.letterbox(ori_img, (self.net_h, self.net_w))
        img = img[..., ::-1]  # BGR -> RGB
        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32) / 255.0
        return img, ratio, (dw, dh)

    def __call__(self, img_list):
        img0_list, ratio_pad_list, imgs = [], [], []

        for img in img_list:
            h, w = img.shape[:2]
            img0_list.append((h, w))

            start = time.time()
            img_p, ratio, pad = self.preprocess(img)
            self.preprocess_time += time.time() - start

            imgs.append(img_p)
            ratio_pad_list.append((ratio, pad))

        img_num = len(imgs)
        input_img = np.zeros((self.batch_size, 3, self.net_h, self.net_w), dtype=np.float32)
        input_img[:img_num] = np.stack(imgs)

        start = time.time()
        outputs = self.session.run(self.output_names, {self.input_name: input_img})
        self.inference_time += time.time() - start

        start = time.time()
        results = self.postprocess(outputs, img0_list, ratio_pad_list)
        self.postprocess_time += time.time() - start

        return results


def xywhr2xyxyxyxy(box):
    """box: [cx, cy, w, h, angle(rad)] return: 4 points shape [4, 2].
    """
    cx, cy, w, h, angle = box
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    dx = w / 2.0
    dy = h / 2.0

    pts = np.array(
        [
            [-dx, -dy],
            [dx, -dy],
            [dx, dy],
            [-dx, dy],
        ],
        dtype=np.float32,
    )

    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

    pts = pts @ R.T
    pts[:, 0] += cx
    pts[:, 1] += cy
    return pts


def draw_numpy(image, boxes, classes_ids, conf_scores):
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].astype(int)
        cls_id = int(classes_ids[i])
        color = COLORS[cls_id % len(COLORS)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_id}:{conf_scores[i]:.2f}"
        cv2.putText(image, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return image


def draw_numpy_obb(image, rboxes, classes_ids, conf_scores):
    """Rboxes: [N, 5] -> [cx, cy, w, h, angle(rad)]."""
    for i in range(len(rboxes)):
        cls_id = int(classes_ids[i])
        color = COLORS[cls_id % len(COLORS)]
        pts = xywhr2xyxyxyxy(rboxes[i]).astype(np.int32)
        cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)

        x, y = pts[0]
        label = f"{cls_id}:{conf_scores[i]:.2f}"
        cv2.putText(image, label, (int(x), max(0, int(y) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return image


def visualize_and_save(image, detections, output_path, conf_thresh=0.25, task="detect"):
    img = image.copy()

    if detections is None or len(detections) == 0:
        logging.warning("No detections found!")
        cv2.imwrite(output_path, img)
        return img

    if task == "detect":
        keep = detections[:, 4] > conf_thresh
        det = detections[keep]

        if len(det) > 0:
            draw_numpy(img, det[:, :4], classes_ids=det[:, 5], conf_scores=det[:, 4])

            for i in range(len(det)):
                box = det[i, :4]
                cls_id = int(det[i, 5])
                conf = det[i, 4]
                logging.info(
                    f"[DET] Box {i + 1}: cls={cls_id}, conf={conf:.3f}, "
                    f"coords=[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]"
                )

    elif task == "obb":
        keep = detections[:, 4] > conf_thresh
        det = detections[keep]

        if len(det) > 0:
            # 正确提取 xywhr
            rboxes = det[:, [0, 1, 2, 3, 6]]  # cx, cy, w, h, angle
            classes = det[:, 5]
            scores = det[:, 4]

            draw_numpy_obb(img, rboxes, classes_ids=classes, conf_scores=scores)

            for i in range(len(det)):
                box = det[i, [0, 1, 2, 3, 6]]  # 注意这里也要取 angle，不是 conf
                cls_id = int(det[i, 5])
                conf = det[i, 4]
                pts = xywhr2xyxyxyxy(box)

                logging.info(
                    f"[OBB] Box {i + 1}: cls={cls_id}, conf={conf:.3f}, "
                    f"xywhr=[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}, {box[4]:.4f}]"
                )
                logging.info(f"[OBB] Polygon {i + 1}: {pts.tolist()}")

    else:
        raise ValueError(f"Unsupported task: {task}")

    cv2.imwrite(output_path, img)
    logging.info(f"Saved to {output_path}")
    return img


def main(args):
    if not os.path.exists(args.onnx):
        raise FileNotFoundError(args.onnx)
    if not os.path.exists(args.data):
        raise FileNotFoundError(args.data)

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    yolo26 = YOLO26(args.onnx, args.conf_thresh, args.iou_thresh, task=args.task)

    img_paths = []
    if os.path.isdir(args.data):
        for fn in sorted(os.listdir(args.data)):
            if fn.lower().endswith((".jpg", ".png", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")):
                img_paths.append(os.path.join(args.data, fn))
    elif os.path.isfile(args.data):
        img_paths.append(args.data)
    else:
        raise FileNotFoundError(f"{args.data} not found.")

    logging.info(f"Total images to process: {len(img_paths)}")

    imgs, names = [], []
    for cn, path in enumerate(img_paths, 1):
        logging.info(f"{cn}, {path}")
        img = cv2.imread(path)

        if img is None:
            logging.warning(f"Cannot read image: {path}")
            continue

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        imgs.append(img)
        names.append(os.path.basename(path))

        if len(imgs) == yolo26.batch_size:
            results = yolo26(imgs)
            for i, det in enumerate(results):
                output_path = os.path.join(output_dir, names[i])
                visualize_and_save(imgs[i], det, output_path, args.conf_thresh, task=args.task)
            imgs.clear()
            names.clear()

    if imgs:
        results = yolo26(imgs)
        for i, det in enumerate(results):
            output_path = os.path.join(output_dir, names[i])
            visualize_and_save(imgs[i], det, output_path, args.conf_thresh, task=args.task)

    logging.info("-------------- Speed Info --------------")
    logging.info(f"preprocess(ms): {yolo26.preprocess_time * 1000:.2f}")
    logging.info(f"inference(ms): {yolo26.inference_time * 1000:.2f}")
    logging.info(f"postprocess(ms): {yolo26.postprocess_time * 1000:.2f}")


def argsparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="/mnt/c/Users/finn.zhang/project/Datasets/JieMei/test/good/NG_NG_20260311162212348.jpg",
        help="image directory or single image path",
    )
    parser.add_argument(
        "--onnx",
        type=str,
        default="/home/finn/ultralytics-main/runs/obb/train/weights/best.onnx",
        help="onnx model path",
    )
    parser.add_argument("--output", type=str, default="./results/images", help="output directory")
    parser.add_argument("--conf_thresh", type=float, default=0.25)
    parser.add_argument("--iou_thresh", type=float, default=0.7)
    parser.add_argument("--task", type=str, default="obb", choices=["detect", "obb"], help="model task type")
    return parser.parse_args()


if __name__ == "__main__":
    args = argsparser()
    main(args)
    print("all done.")

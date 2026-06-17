import os
import time
import cv2
import argparse
import numpy as np
import onnxruntime as ort
import logging

logging.basicConfig(level=logging.INFO)


COLORS = [
    [56, 0, 255], [226, 255, 0], [0, 94, 255], [0, 37, 255], [0, 255, 94],
    [255, 226, 0], [0, 18, 255], [255, 151, 0], [170, 0, 255], [0, 255, 56],
    [255, 0, 75], [0, 75, 255], [0, 255, 169], [255, 0, 207], [75, 255, 0],
]


class PostProcess:
    def __init__(self, conf_thresh=0.25, nms_thresh=0.45, nc=1, agnostic=False, multi_label=False, max_det=300):
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.nc = nc
        self.agnostic = agnostic
        self.multi_label = multi_label
        self.max_det = max_det
        self.max_wh = 7680

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def xywh2xyxy(self, x):
        """Convert xywh to xyxy format"""
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y

    def nms(self, boxes, scores):
        """Non-maximum suppression, matching ultralytics logic"""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
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
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(iou <= self.nms_thresh)[0]
            order = order[inds + 1]
        return keep

    def scale_boxes(self, img1_shape, boxes, img0_shape, ratio_pad):
        """Rescale boxes from img1_shape to img0_shape, matching ultralytics ops.scale_boxes"""
        gain = ratio_pad[0]
        pad = ratio_pad[1]

        if len(boxes) == 0:
            return boxes

        boxes[:, [0, 2]] -= pad[0]
        boxes[:, [1, 3]] -= pad[1]
        boxes[:, :4] /= gain

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img0_shape[1])
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img0_shape[0])

        return boxes

    def __call__(self, preds_batch, img0_batch, ratio_pad_batch):
        """Postprocess predictions, matching ultralytics non_max_suppression logic"""
        if isinstance(preds_batch, list) and len(preds_batch) == 1:
            preds = preds_batch[0]
        else:
            raise NotImplementedError(f"Unsupported preds type: {type(preds_batch)}")

        bs = preds.shape[0]
        nc = self.nc
        nm = preds.shape[1] - nc - 4
        mi = 4 + nc

        preds = np.transpose(preds, (0, 2, 1))

        xc = preds[:, :, 4:mi].max(axis=2) > self.conf_thresh

        preds[..., :4] = self.xywh2xyxy(preds[..., :4])

        batch_outs = []
        for xi in range(bs):
            pred = preds[xi]

            x = pred[xc[xi]]

            if len(x) == 0:
                batch_outs.append(np.zeros((0, 6 + nm)))
                continue

            box, cls, mask = np.split(x, [4, 4 + nc], axis=1)

            if self.multi_label:
                i, j = np.where(cls > self.conf_thresh)
                x = np.concatenate([box[i], x[i, 4 + j, None], j.astype(np.float32), mask[i]], axis=1)
            else:
                conf = cls.max(axis=1, keepdims=True)
                j = cls.argmax(axis=1, keepdims=True)
                filt = conf.flatten() > self.conf_thresh
                x = np.concatenate([box, conf, j.astype(np.float32), mask], axis=1)[filt]

            if self.agnostic:
                c = x[:, 5:6] * 0
            else:
                c = x[:, 5:6] * self.max_wh

            boxes = x[:, :4] + c
            scores = x[:, 4]

            keep = self.nms(boxes, scores)
            keep = keep[:self.max_det]
            x = x[keep]

            img0_shape = img0_batch[xi]
            ratio_pad = ratio_pad_batch[xi]
            logging.info(f"gain={ratio_pad[0]}, pad={ratio_pad[1]}, orig={img0_shape[1]}x{img0_shape[0]}")
            logging.info(f"Before scale - xyxy: {x[:3, :4]}")

            x[:, :4] = self.scale_boxes((640, 640), x[:, :4].copy(), img0_shape, ratio_pad)
            logging.info(f"After scale - xyxy: {x[:3, :4]}")

            batch_outs.append(x)

        return batch_outs


class YOLO26:
    def __init__(self, onnx_path, conf_thresh=0.25, nms_thresh=0.45):
        providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, providers=providers)

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.input_shape = self.session.get_inputs()[0].shape

        self.output_shape = self.session.get_outputs()[0].shape
        self.batch_size = self.input_shape[0] if self.input_shape[0] else 1
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]

        self.nc = self.output_shape[1] - 4

        logging.info(f"Load ONNX: {onnx_path}")
        logging.info(f"Input shape: {self.input_shape}")
        logging.info(f"Output shape: {self.output_shape}, nc={self.nc}")

        self.conf_thresh = conf_thresh
        self.postprocess = PostProcess(conf_thresh, nms_thresh, nc=self.nc)

        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114)):
        shape = im.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        logging.info(f"letterbox: shape={shape}, r={r}, new_unpad={new_unpad}, dw={dw}, dh={dh}")

        return im, r, (dw, dh)

    def preprocess(self, ori_img):
        img, ratio, (dw, dh) = self.letterbox(ori_img, (self.net_h, self.net_w))

        img = img[..., ::-1]
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


def draw_numpy(image, boxes, classes_ids, conf_scores):
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].astype(int)
        cls_id = int(classes_ids[i])
        color = COLORS[cls_id % len(COLORS)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_id}:{conf_scores[i]:.2f}"
        cv2.putText(image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return image


def visualize_and_save(image, detections, output_path, conf_thresh=0.25):
    img = image.copy()

    if detections is None or len(detections) == 0:
        logging.warning("No detections found!")
        cv2.imwrite(output_path, img)
        return img

    keep = detections[:, -2] > conf_thresh
    det = detections[keep]

    if len(det) > 0:
        draw_numpy(img, det[:, :4], classes_ids=det[:, -1], conf_scores=det[:, -2])

        for i in range(len(det)):
            box = det[i, :4]
            cls_id = int(det[i, -1])
            conf = det[i, -2]
            logging.info(f"Box {i+1}: cls={cls_id}, conf={conf:.3f}, coords=[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")

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

    yolo26 = YOLO26(args.onnx, args.conf_thresh, args.iou_thresh)

    img_paths = []
    if os.path.isdir(args.data):
        for fn in sorted(os.listdir(args.data)):
            if fn.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.webp')):
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
                visualize_and_save(imgs[i], det, output_path, args.conf_thresh)
            imgs.clear()
            names.clear()

    if imgs:
        results = yolo26(imgs)
        for i, det in enumerate(results):
            output_path = os.path.join(output_dir, names[i])
            visualize_and_save(imgs[i], det, output_path, args.conf_thresh)

    logging.info("-------------- Speed Info --------------")
    logging.info(f"preprocess(ms): {yolo26.preprocess_time * 1000:.2f}")
    logging.info(f"inference(ms): {yolo26.inference_time * 1000:.2f}")
    logging.info(f"postprocess(ms): {yolo26.postprocess_time * 1000:.2f}")


def argsparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="/mnt/c/Users/finn.zhang/project/Datasets/JieMei/test/good/NG_NG_20260311162212348.jpg", help='image directory or single image path')
    parser.add_argument('--onnx', type=str, default="/home/finn/ultralytics-main/runs/detect/train/weights/yolo26n_crop.onnx", help='onnx model path')
    parser.add_argument('--output', type=str, default="./results/images", help='output directory')
    parser.add_argument('--conf_thresh', type=float, default=0.25)
    parser.add_argument('--iou_thresh', type=float, default=0.7)
    return parser.parse_args()


if __name__ == "__main__":
    args = argsparser()
    main(args)
    print("all done.")

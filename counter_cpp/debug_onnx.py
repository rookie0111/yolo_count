from ultralytics import YOLO

m = YOLO("/home/finn/ultralytics-main-count/runs/detect/bag_data6/weights/best.pt")
r = m(
    "/mnt/c/Users/finn.zhang/project/Datasets/ObjectCounting.v1i.yolo26/images/val/frame_100_jpg.rf.72707ebc70d741d5b9e1d11546d2e36d.jpg",
    conf=0.35,
)
boxes = r[0].boxes
if boxes is not None and len(boxes) > 0:
    print(f"Detections: {len(boxes)}")
    for i in range(min(len(boxes), 5)):
        b = boxes.xyxy[i]
        print(f"  [{int(b[0])},{int(b[1])},{int(b[2])},{int(b[3])}] conf={boxes.conf[i]:.2f}")
    print(f"Image size: {r[0].orig_shape}")
else:
    print("No detections")

# Also test ONNX directly
import cv2
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("/home/finn/ultralytics-main-count/runs/detect/bag_data6/weights/best.onnx")
print(f"\nONNX input: {session.get_inputs()[0].name} {session.get_inputs()[0].shape}")
print(f"ONNX output: {session.get_outputs()[0].name} {session.get_outputs()[0].shape}")

img = cv2.imread(
    "/mnt/c/Users/finn.zhang/project/Datasets/ObjectCounting.v1i.yolo26/images/val/frame_100_jpg.rf.72707ebc70d741d5b9e1d11546d2e36d.jpg"
)
h, w = img.shape[:2]
img_rgb = cv2.cvtColor(cv2.resize(img, (640, 640)), cv2.COLOR_BGR2RGB)
blob = img_rgb.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...] / 255.0

out = session.run(None, {"images": blob})[0]
print(f"ONNX raw output shape: {out.shape}")
print(f"ONNX output[0][:8]: {out[0, :8, 0]}")
print(f"ONNX output[0][:8]: {out[0, :8, 100]}")
print(f"ONNX output[0][:8]: {out[0, :8, 5000]}")
print(f"ONNX conf max: {out[0, 4:, :].max()}, mean: {out[0, 4:, :].mean():.4f}")

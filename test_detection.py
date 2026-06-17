from ultralytics import YOLO

model = YOLO("runs/detect/bag_data6/weights/best.pt")

val_imgs = [
    r"C:\Users\finn.zhang\project\Datasets\ObjectCounting.v1i.yolo26\images\val\frame_100_jpg.rf.72707ebc70d741d5b9e1d11546d2e36d.jpg",
    r"C:\Users\finn.zhang\project\Datasets\ObjectCounting.v1i.yolo26\images\val\frame_101_jpg.rf.7a4077657f4c3050204ce2aa353deeb6.jpg",
]

for img in val_imgs:
    r = model(img, conf=0.35, verbose=False)
    n = len(r[0].boxes) if r[0].boxes else 0
    confs = r[0].boxes.conf.tolist() if r[0].boxes else []
    print(f"{n} detections, conf={[round(c, 2) for c in confs]}")

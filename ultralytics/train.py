from ultralytics import YOLO
def train():
    model = YOLO("/home/finn/ultralytics-main/ultralytics/yolo26n-obb.pt")
    # model = YOLO("yolo26n-obb.pt")
    results = model.train(data="mydata-obb.yaml", epochs=100, imgsz=640)
    model.val()
    # model.export()
    # Use one-to-one head (default, no NMS required)
    # results = model.predict("/home/finn/ultralytics-main/ultralytics/assets/bus.jpg")  # inference
    # # metrics = model.val(data="coco.yaml")  # validation
    # model.export(format="onnx")  # export

def train_bag():
    model = YOLO("/home/finn/ultralytics-main/ultralytics/yolo26n.pt")
    # model = YOLO("yolo26n-obb.pt")
    results = model.train(data="/home/finn/ultralytics-main-count/ultralytics/cfg/datasets/bag_data.yaml", epochs=100, imgsz=640, name="bag_data")
    model.val()
def predict():
    # model = YOLO("/home/finn/ultralytics-main/runs/detect/train/weights/bfest.pt")
    model = YOLO("/home/finn/ultralytics-main/runs/obb/train3/weights/best.pt")
    results = model.predict("/mnt/c/Users/finn.zhang/project/Datasets/hainanjiahua_20260416/NG_1/NG_20260414162419520.png", end2end=False,save=True)  # inference
    print("end")
def export():
    model = YOLO("/home/finn/ultralytics-main-count/runs/detect/bag_data6/weights/best.pt")
    # Use one-to-many head (requires NMS)
    
    # metrics = model.val(data="coco.yaml", end2end=False)  # validation
    model.export(format="onnx", end2end=False)  # export

if __name__ == "__main__":
    # train()
    # predict()
    export()
    # train_bag()
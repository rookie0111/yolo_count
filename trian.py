from ultralytics import YOLO
def train():
    model = YOLO("/home/finn/ultralytics-main/ultralytics/yolo26n.pt")
    results = model.train(data="mydata.yaml", epochs=100, imgsz=640)
    model.val()
    # model.export()
    # Use one-to-one head (default, no NMS required)
    # results = model.predict("/home/finn/ultralytics-main/ultralytics/assets/bus.jpg")  # inference
    # # metrics = model.val(data="coco.yaml")  # validation
    # model.export(format="onnx")  # export

def predict():
    model = YOLO("/home/finn/ultralytics-main/ultralytics/yolo26n.pt")
    results = model.predict("/home/finn/ultralytics-main/ultralytics/assets/bus.jpg", end2end=False)  # inference

def export():
    model = YOLO("/home/finn/ultralytics-main/ultralytics/yolo26n.pt")
    # Use one-to-many head (requires NMS)
    
    # metrics = model.val(data="coco.yaml", end2end=False)  # validation
    model.export(format="onnx", end2end=False)  # export

if __name__ == "__main__":
    train()
    # predict()
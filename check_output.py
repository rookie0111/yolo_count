from ultralytics import YOLO
import torch

model = YOLO('ultralytics/yolo26n.pt')
img = torch.zeros(1, 3, 640, 640)
with torch.no_grad():
    out = model.model(img)
print('Model output shapes:', [o.shape for o in out] if isinstance(out, (list, tuple)) else out.shape)

# Check with actual image
img2 = model.predict('ultralytics/assets/NG_NG_20260316121648480.jpg', verbose=False)
print('Predict boxes:', img2[0].boxes.shape)

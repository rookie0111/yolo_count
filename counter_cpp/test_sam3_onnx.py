import onnxruntime as ort
import numpy as np
import cv2
import torch
from ultralytics.models.sam.build_sam3 import build_sam3_image_model

model = build_sam3_image_model('/home/finn/ultralytics-main-count/sam3.pt')
model.eval().cuda()
with torch.no_grad():
    model.set_classes(['bag on the conveyor belt'])
lang_feats = model.text_embeddings['language_features'].detach().cpu().numpy()
lang_mask = model.text_embeddings['language_mask'].detach().cpu().numpy()

img_path = '/mnt/c/Users/finn.zhang/project/Datasets/yolo_zhonghua_counter/images/train/frame_0014.jpg'
img = cv2.imread(img_path)
img_resized = cv2.resize(img, (1008, 1008))
img_input = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32)
img_input = (img_input - 127.5) / 127.5
img_input = img_input.transpose(2, 0, 1)[None]

sess_bb = ort.InferenceSession('/home/finn/ultralytics-main-count/sam3_backbone.onnx', providers=['CUDAExecutionProvider'])
bb_out = sess_bb.run(None, {'image': img_input})
bb_names = [o.name for o in sess_bb.get_outputs()]
out_dict = {name: bb_out[i] for i, name in enumerate(bb_names)}

sess_det = ort.InferenceSession('/home/finn/ultralytics-main-count/sam3_detector.onnx', providers=['CUDAExecutionProvider'])
det_inputs = {
    'vision_pos_enc': out_dict['vision_pos_enc_2'].astype(np.float32),
    'backbone_fpn_0': out_dict['backbone_fpn_0'].astype(np.float32),
    'backbone_fpn_1': out_dict['backbone_fpn_1'].astype(np.float32),
    'backbone_fpn_2': out_dict['backbone_fpn_2'].astype(np.float32),
    'language_features': lang_feats.astype(np.float32),
    'language_mask': lang_mask.astype(bool),
    'text_ids': np.array([0], dtype=np.int64),
}
det_out = sess_det.run(None, det_inputs)

pred_boxes = det_out[0]
pred_logits = det_out[1]
pred_masks = det_out[2]
presence = det_out[3]

scores = 1.0 / (1.0 + np.exp(-pred_logits[:, :, 0]))
presence_score = 1.0 / (1.0 + np.exp(-presence[:, 0]))
final_scores = scores * presence_score[:, None]

conf_thres = 0.25
keep = final_scores[0] > conf_thres
print(f'SAM3 ONNX detections above {conf_thres}: {int(keep.sum())}')
boxes_xywh = pred_boxes[0, keep]
scores_keep = final_scores[0, keep]

H, W = img.shape[:2]
for i in range(min(len(scores_keep), 10)):
    cx, cy, w, h = boxes_xywh[i]
    x1 = int((cx - w/2) * W)
    y1 = int((cy - h/2) * H)
    x2 = int((cx + w/2) * W)
    y2 = int((cy + h/2) * H)
    print(f'  box {i}: [{x1},{y1},{x2},{y2}] score={scores_keep[i]:.3f}')

# Save text embeddings for C++ usage
np.savez('/home/finn/ultralytics-main-count/counter_cpp/sam3_text_embeddings.npz',
         language_features=lang_feats,
         language_mask=lang_mask)
print('Saved text embeddings to counter_cpp/sam3_text_embeddings.npz')
print('SAM3 ONNX pipeline test PASSED')

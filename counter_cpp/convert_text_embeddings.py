"""Convert sam3_text_embeddings.npz to .yml format for C++ OpenCV FileStorage.
OpenCV FileStorage doesn't support 3D matrices, so we flatten and store shape separately."""
import numpy as np
import cv2

data = np.load('/home/finn/ultralytics-main-count/counter_cpp/sam3_text_embeddings.npz')
lang_feats = data['language_features']  # (32, 1, 256)
lang_mask = data['language_mask']       # (1, 32)

print(f"language_features: {lang_feats.shape}, dtype={lang_feats.dtype}")
print(f"language_mask: {lang_mask.shape}, dtype={lang_mask.dtype}")

# Flatten language_features to 1D and store shape
lang_feats_flat = lang_feats.flatten().astype(np.float32)  # (8192,)
lang_mask_flat = lang_mask.flatten().astype(np.uint8)       # (32,)

out_path = '/home/finn/ultralytics-main-count/counter_cpp/sam3_text_embeddings.yml'
fs = cv2.FileStorage(out_path, cv2.FileStorage_WRITE)
fs.write('lang_feats_flat', np.ascontiguousarray(lang_feats_flat.reshape(1, -1)))  # (1, 8192)
fs.write('lang_feats_size', np.array([32, 1, 256], dtype=np.int32))
fs.write('lang_mask_flat', np.ascontiguousarray(lang_mask_flat.reshape(1, -1)))    # (1, 32)
fs.write('lang_mask_size', np.array([1, 32], dtype=np.int32))
fs.release()
print(f"Saved to {out_path}")
print(f"  lang_feats_flat: {lang_feats_flat.shape}, total={lang_feats_flat.size}")
print(f"  lang_mask_flat: {lang_mask_flat.shape}, total={lang_mask_flat.size}")

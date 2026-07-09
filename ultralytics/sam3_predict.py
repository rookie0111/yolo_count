import cv2

from ultralytics.models.sam import SAM3SemanticPredictor

# Initialize predictor with configuration
overrides = dict(
    conf=0.25,
    task="segment",
    mode="predict",
    model="/home/finn/ultralytics-main-count/sam3.pt",
    half=False,  # FP32 for compatibility
    save=True,
)
predictor = SAM3SemanticPredictor(overrides=overrides)

# Set image once for multiple queries
predictor.set_image("/mnt/c/Users/finn.zhang/project/Datasets/yolo_zhonghua_counter/images/train/frame_0014.jpg")

# Object detection via text prompts
results = predictor(text=["filled with fertilizer bags",])

# Visualize detection results: numbered bounding boxes only, no labels
for r in results:
    if r.boxes is not None:
        n_det = r.boxes.data.shape[0]
        print(f"\nDetected {n_det} objects:")
        for i, (box, cls, conf) in enumerate(zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf)):
            name = r.names[int(cls)]
            x1, y1, x2, y2 = box.int().tolist()
            print(f"  {i + 1}. {name}: conf={conf:.3f}, box=[{x1},{y1},{x2},{y2}]")

        from ultralytics.utils.plotting import Annotator, colors

        annotator = Annotator(r.orig_img.copy(), line_width=2)
        for i, box in enumerate(r.boxes.xyxy):
            c = colors(i, True)
            x1, y1, x2, y2 = box.int().tolist()
            annotator.box_label(box, str(i + 1), color=c, txt_color=(255, 255, 255))

        annotated = annotator.result()

        cv2.imshow("SAM3 Object Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        out_path = "sam3_detection_output.jpg"
        cv2.imwrite(out_path, annotated)
        print(f"\nSaved annotated image to {out_path}")

# Works with descriptive phrases
results = predictor(text=["person with red cloth", "person with blue cloth"])

# Query with a single concept
results = predictor(text=["a person"])
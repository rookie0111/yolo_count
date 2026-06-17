import os

from PIL import Image

from ultralytics import YOLO


def crop_single_image(image_path, model, output_folder):
    img = Image.open(image_path)
    img_name = os.path.basename(image_path)
    name, ext = os.path.splitext(img_name)

    results = model.predict(image_path, conf=conf, end2end=False)
    saved_count = 0

    for result in results:
        boxes = result.boxes
        for j, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            crop = img.crop((x1, y1, x2, y2))

            cls_id = int(box.cls[0])
            cls_name = result.names[cls_id]

            save_name = f"{name}_{cls_name}_{j}{ext}"
            save_path = os.path.join(output_folder, save_name)
            crop.save(save_path)
            saved_count += 1
            print(f"Saved: {save_path}")

    return saved_count


def crop_predictions(input_path, model_path, output_folder):
    # global conf
    os.makedirs(output_folder, exist_ok=True)

    model = YOLO(model_path)
    # conf = conf

    total = 0
    if os.path.isfile(input_path):
        total = crop_single_image(input_path, model, output_folder)
    elif os.path.isdir(input_path):
        extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        for img_file in os.listdir(input_path):
            if os.path.splitext(img_file.lower())[1] in extensions:
                img_path = os.path.join(input_path, img_file)
                print(f"Processing: {img_file}")
                total += crop_single_image(img_path, model, output_folder)

    print(f"Total cropped: {total}")
    return total


if __name__ == "__main__":
    input_path = "/mnt/c/Users/finn.zhang/project/Datasets/JieMei/test/good2"
    model_path = "/home/finn/ultralytics-main/runs/detect/train/weights/best.pt"
    output_folder = "/mnt/c/Users/finn.zhang/project/Datasets/JieMei/test/good2_test_crop"
    conf = 0.25

    crop_predictions(input_path, model_path, output_folder)

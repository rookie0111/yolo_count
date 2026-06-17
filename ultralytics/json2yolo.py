import glob
import json
import os


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def normalize_point(x, y, w, h):
    return x / w, y / h


def convert_one_json(json_path, output_txt_path, class_map=None):
    """将单个 json 转为 yolo obb txt.

    参数：
    - json_path: 输入json路径
    - output_txt_path: 输出txt路径
    - class_map: 类别映射字典，例如 {"0":0, "defect":1}
                 如果为None，则尝试将 label 直接转 int
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    img_w = data.get("imageWidth", None)
    img_h = data.get("imageHeight", None)

    if img_w is None or img_h is None:
        print(f"[跳过] 缺少 imageWidth/imageHeight: {json_path}")
        return

    lines = []

    shapes = data.get("shapes", [])
    for shape in shapes:
        label = str(shape.get("label", "")).strip()
        points = shape.get("points", [])
        shape_type = shape.get("shape_type", "")

        # 只处理4点旋转框
        if len(points) != 4:
            print(f"[跳过] 不是4点标注: {json_path}")
            continue

        if shape_type not in ["rotation", "polygon", "obb", "rotated_box", "rectangle"]:
            # 不强制限制也可以，这里保守一点
            pass

        # 类别映射
        if class_map is not None:
            if label not in class_map:
                print(f"[跳过] 未在类别映射中找到 label={label}, 文件={json_path}")
                continue
            cls_id = class_map[label]
        else:
            try:
                cls_id = int(label)
            except:
                print(f"[跳过] label无法转int，且未提供class_map: label={label}, 文件={json_path}")
                continue

        norm_pts = []
        for pt in points:
            x, y = pt
            xn, yn = normalize_point(float(x), float(y), img_w, img_h)
            norm_pts.extend([xn, yn])

        line = str(cls_id) + " " + " ".join(f"{v:.6f}" for v in norm_pts)
        lines.append(line)

    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def batch_convert_json_to_yolo_obb(json_dir, output_dir, class_map=None, recursive=False):
    """批量转换文件夹下所有json."""
    ensure_dir(output_dir)

    if recursive:
        pattern = os.path.join(json_dir, "**", "*.json")
        json_files = glob.glob(pattern, recursive=True)
    else:
        pattern = os.path.join(json_dir, "*.json")
        json_files = glob.glob(pattern)

    if not json_files:
        print(f"未找到json文件: {json_dir}")
        return

    count_ok = 0
    count_fail = 0

    for json_path in json_files:
        try:
            base_name = os.path.splitext(os.path.basename(json_path))[0]
            out_txt = os.path.join(output_dir, base_name + ".txt")
            convert_one_json(json_path, out_txt, class_map=class_map)
            print(f"[成功] {json_path} -> {out_txt}")
            count_ok += 1
        except Exception as e:
            print(f"[失败] {json_path}, 错误: {e}")
            count_fail += 1

    print("\n========== 转换完成 ==========")
    print(f"成功: {count_ok}")
    print(f"失败: {count_fail}")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    # ======== 这里改成你的路径 ========
    json_dir = r"/mnt/c/Users/finn.zhang/project/Datasets/20260515_transparent_80ml/yolo_dataset/images/train"
    output_dir = r"/mnt/c/Users/finn.zhang/project/Datasets/20260515_transparent_80ml/yolo_dataset/labels/train"

    # 如果你的 label 本身就是 "0"、"1"、"2"，可以设为 None
    class_map = None

    # 如果你的标签是字符串类别名，例如：
    # class_map = {
    #     "defect": 0,
    #     "scratch": 1,
    #     "dirty": 2
    # }

    batch_convert_json_to_yolo_obb(json_dir=json_dir, output_dir=output_dir, class_map=class_map, recursive=False)

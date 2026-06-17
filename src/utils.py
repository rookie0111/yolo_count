from pathlib import Path

import yaml


def load_runtime_config(config_path="configs/runtime.yaml"):
    """Load runtime configuration from a YAML file.

    Args:
        config_path (str): Path to the runtime YAML config file.

    Returns:
        (dict): Configuration dictionary.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def build_solution_kwargs(config):
    """Convert runtime config dict to ObjectCounter kwargs.

    Args:
        config (dict): Loaded runtime configuration.

    Returns:
        (dict): Keyword arguments for ObjectCounter initialization.
    """
    model_cfg = config.get("model", {})
    tracker_cfg = config.get("tracker", {})
    counter_cfg = config.get("counter", {})
    runtime_cfg = config.get("runtime", {})
    camera_cfg = config.get("camera", {})

    region = None
    poly_keys = [f"poly_p{i}" for i in range(1, 5)]
    if all(k in counter_cfg for k in poly_keys):
        region = [tuple(counter_cfg[k]) for k in poly_keys]
    elif "line_p1" in counter_cfg and "line_p2" in counter_cfg:
        region = [tuple(counter_cfg["line_p1"]), tuple(counter_cfg["line_p2"])]

    tracker_map = {"bytetrack": "bytetrack.yaml", "botsort": "botsort.yaml"}

    kwargs = {
        "model": model_cfg.get("weight", "yolo26n.pt"),
        "conf": model_cfg.get("conf_thres", 0.35),
        "iou": model_cfg.get("iou_thres", 0.50),
        "region": region,
        "tracker": tracker_map.get(tracker_cfg.get("type", "bytetrack"), "bytetrack.yaml"),
        "direction": counter_cfg.get("direction", "any"),
        "polygon_mode": counter_cfg.get("polygon_mode", "enter_exit"),
        "show": runtime_cfg.get("show_window", False),
        "verbose": True,
        "camera_source": camera_cfg.get("source", 0),
        "camera_width": camera_cfg.get("width", 1280),
        "camera_height": camera_cfg.get("height", 720),
        "camera_fps": camera_cfg.get("fps", 30),
        "log_dir": runtime_cfg.get("log_dir", "logs"),
    }

    return kwargs


def ensure_dir(path):
    """Ensure a directory exists, creating it if necessary.

    Args:
        path (str): Directory path.
    """
    Path(path).mkdir(parents=True, exist_ok=True)

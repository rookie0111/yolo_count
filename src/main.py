import argparse
import sys
import time
from pathlib import Path

import cv2

# "/mnt/d/download/vidu-video-3321313083087446.mp4"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from display import NumberDisplay
from logger import CountLogger
from ultralytics.solutions.object_counter import ObjectCounter
from utils import build_solution_kwargs, load_runtime_config


def main():
    """Main entry point for the bag counter system."""
    parser = argparse.ArgumentParser(description="Conveyor Belt Bag Counter")
    parser.add_argument("--config", type=str, default="configs/runtime.yaml", help="Path to runtime config YAML")
    parser.add_argument("--source", type=str, default=None, help="Video source (camera index or video path)")
    parser.add_argument("--show", action="store_true", help="Show output window")
    parser.add_argument("--no-display", action="store_true", help="Disable serial display")
    args = parser.parse_args()

    cfg = load_runtime_config(args.config)
    kwargs = build_solution_kwargs(cfg)

    if args.source is not None:
        kwargs["camera_source"] = args.source
    if args.show:
        kwargs["show"] = True

    camera_source = kwargs.pop("camera_source", 0)
    camera_width = kwargs.pop("camera_width", 1280)
    camera_height = kwargs.pop("camera_height", 720)
    camera_fps = kwargs.pop("camera_fps", 30)
    log_dir = kwargs.pop("log_dir", "logs")

    counter = ObjectCounter(**kwargs)

    logger = CountLogger(log_dir) if cfg.get("runtime", {}).get("save_log", True) else None

    display_cfg = cfg.get("display", {})
    number_display = None
    if display_cfg.get("enable", False) and not args.no_display:
        number_display = NumberDisplay(
            port=display_cfg.get("port", "/dev/ttyUSB0"),
            baudrate=display_cfg.get("baudrate", 9600),
            digits=display_cfg.get("digits", 8),
        )

    cap = cv2.VideoCapture(camera_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
    cap.set(cv2.CAP_PROP_FPS, camera_fps)

    if not cap.isOpened():
        print(f"Error: Cannot open video source: {camera_source}")
        sys.exit(1)

    print("Bag Counter started. Press Ctrl+C to stop.")
    last_count = 0
    target_fps = cfg.get("runtime", {}).get("target_fps", 10)
    frame_interval = 1.0 / target_fps if target_fps > 0 else 0

    try:
        while True:
            loop_start = time.time()

            ret, frame = cap.read()
            if not ret:
                print("End of video stream.")
                break

            result = counter.process(frame)

            if result.total_count != last_count:
                last_count = result.total_count
                print(f"Count: {result.total_count}")

                if logger is not None:
                    logger.log_count(result.total_count, len(counter.counted_ids))

                if number_display is not None:
                    number_display.show_number(result.total_count)

            elapsed = time.time() - loop_start
            if frame_interval > 0 and elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        cap.release()
        if number_display is not None:
            number_display.close()
        cv2.destroyAllWindows()
        print(f"Final count: {last_count}")


if __name__ == "__main__":
    main()

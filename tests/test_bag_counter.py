import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from ultralytics.solutions.object_counter import ObjectCounter


class TestPointLineSide:
    """Tests for the point_line_side cross product method."""

    def test_points_on_opposite_sides(self):
        p1, p2 = (0, 0), (100, 100)
        r1 = ObjectCounter.point_line_side(50, 0, p1, p2)
        r2 = ObjectCounter.point_line_side(0, 50, p1, p2)
        assert r1 != 0
        assert r2 != 0
        assert (r1 > 0) != (r2 > 0)

    def test_point_on_line(self):
        p1, p2 = (0, 0), (100, 100)
        result = ObjectCounter.point_line_side(50, 50, p1, p2)
        assert result == 0

    def test_diagonal_line_different_quadrant(self):
        p1, p2 = (180, 430), (330, 250)
        result_above = ObjectCounter.point_line_side(200, 200, p1, p2)
        result_below = ObjectCounter.point_line_side(300, 500, p1, p2)
        assert result_above < 0
        assert result_below > 0

    def test_opposite_sides(self):
        p1, p2 = (100, 0), (100, 200)
        result_right = ObjectCounter.point_line_side(150, 100, p1, p2)
        result_left = ObjectCounter.point_line_side(50, 100, p1, p2)
        assert (result_right > 0 and result_left < 0) or (result_right < 0 and result_left > 0)


class TestDirectionRules:
    """Tests for direction rule validation."""

    def test_up_right_rule_exists(self):
        assert "up_right" in ObjectCounter.DIRECTION_RULES
        assert ObjectCounter.DIRECTION_RULES["up_right"] == (-1, 1)

    def test_down_right_rule_exists(self):
        assert "down_right" in ObjectCounter.DIRECTION_RULES
        assert ObjectCounter.DIRECTION_RULES["down_right"] == (1, -1)

    def test_all_directions_defined(self):
        for d in ["up_right", "down_right", "up_left", "down_left"]:
            assert d in ObjectCounter.DIRECTION_RULES


class TestObjectCounterInit:
    """Tests for ObjectCounter initialization with direction config."""

    def test_default_direction_is_any(self):
        counter = ObjectCounter(model="yolo26n.pt", show=False)
        assert counter.direction == "any"

    def test_custom_direction(self):
        counter = ObjectCounter(model="yolo26n.pt", direction="up_right", show=False)
        assert counter.direction == "up_right"

    def test_total_count_starts_at_zero(self):
        counter = ObjectCounter(model="yolo26n.pt", show=False)
        assert counter.total_count == 0


class TestConfigUtils:
    """Tests for configuration utility functions."""

    def test_load_runtime_config_defaults(self):
        from src.utils import build_solution_kwargs, load_runtime_config

        cfg = load_runtime_config("configs/runtime.yaml")
        kwargs = build_solution_kwargs(cfg)

        assert kwargs["direction"] == "up_right"
        assert kwargs["model"] == "runs/detect/bag_data6/weights/best.pt"
        assert kwargs["conf"] == 0.35
        assert kwargs["region"] is not None
        assert len(kwargs["region"]) == 2

    def test_logger_csv_format(self):
        from src.logger import CountLogger

        logger = CountLogger("logs")
        logger.log_count(5, 1)
        log_file = logger.log_file
        assert log_file.exists()

        with open(log_file) as f:
            header = f.readline().strip()
            assert header == "timestamp,count,track_id"

        logger.log_file.unlink()


class TestDisplayModule:
    """Tests for NumberDisplay (without real serial hardware)."""

    def test_display_graceful_failure(self):

        from src.display import NumberDisplay

        display = NumberDisplay(port="/dev/nonexistent", baudrate=9600, digits=8)
        assert display.set is None
        display.show_number(123)
        display.close()

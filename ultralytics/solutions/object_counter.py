# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from collections import defaultdict
from typing import Any

import cv2

from ultralytics.solutions.solutions import BaseSolution, SolutionAnnotator, SolutionResults
from ultralytics.utils.plotting import colors


class ObjectCounter(BaseSolution):
    """A class to manage the counting of objects in a real-time video stream based on their tracks.

    This class extends the BaseSolution class and provides functionality for counting objects moving in and out of a
    specified region in a video stream. It supports both polygonal and linear regions for counting.

    Attributes:
        in_count (int): Counter for objects moving inward.
        out_count (int): Counter for objects moving outward.
        counted_ids (list[int]): List of IDs of objects that have been counted.
        classwise_count (dict[str, dict[str, int]]): Dictionary for counts, categorized by object class.
        region_initialized (bool): Flag indicating whether the counting region has been initialized.
        show_in (bool): Flag to control display of inward count.
        show_out (bool): Flag to control display of outward count.
        margin (int): Margin for background rectangle size to display counts properly.

    Methods:
        count_objects: Count objects within a polygonal or linear region based on their tracks.
        display_counts: Display object counts on the frame.
        process: Process input data and update counts.

    Examples:
        >>> counter = ObjectCounter()
        >>> frame = cv2.imread("frame.jpg")
        >>> results = counter.process(frame)
        >>> print(f"Inward count: {counter.in_count}, Outward count: {counter.out_count}")
    """

    DIRECTION_RULES = {
        "up_right": (-1, 1),
        "down_right": (1, -1),
        "up_left": (1, -1),
        "down_left": (-1, 1),
    }

    @staticmethod
    def point_line_side(px: float, py: float, p1: tuple[int, int], p2: tuple[int, int]) -> float:
        """Compute the cross product to determine which side of the line a point lies on.

        Args:
            px (float): x-coordinate of the point.
            py (float): y-coordinate of the point.
            p1 (tuple[int, int]): First point defining the line.
            p2 (tuple[int, int]): Second point defining the line.

        Returns:
            (float): Positive if point is on clockwise side, negative if on counter-clockwise side.
        """
        return (p2[0] - p1[0]) * (py - p1[1]) - (p2[1] - p1[1]) * (px - p1[0])

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the ObjectCounter class for real-time object counting in video streams."""
        super().__init__(**kwargs)

        self.total_count = 0
        self.in_count = 0  # Counter for objects moving inward
        self.out_count = 0  # Counter for objects moving outward
        self.counted_ids = []  # List of IDs of objects that have been counted
        self.entered_ids = set()  # Set of track_ids that have entered polygon but not yet exited
        self.classwise_count = defaultdict(lambda: {"IN": 0, "OUT": 0})  # Dictionary for counts, categorized by class
        self.region_initialized = False  # Flag indicating whether the region has been initialized

        self.show_in = self.CFG["show_in"]
        self.show_out = self.CFG["show_out"]
        self.direction = self.CFG.get("direction", "any")
        self.polygon_mode = self.CFG.get("polygon_mode", "enter_exit")  # "enter_exit" or "enter"
        self.margin = self.line_width * 2  # Scales the background rectangle size to display counts properly

    def count_objects(
        self,
        current_centroid: tuple[float, float],
        track_id: int,
        prev_position: tuple[float, float] | None,
        cls: int,
    ) -> None:
        """Count objects within a polygonal or linear region based on their tracks.

        Args:
            current_centroid (tuple[float, float]): Current centroid coordinates (x, y) in the current frame.
            track_id (int): Unique identifier for the tracked object.
            prev_position (tuple[float, float], optional): Last frame position coordinates (x, y) of the track.
            cls (int): Class index for classwise count updates.

        Examples:
            >>> counter = ObjectCounter()
            >>> track_line = {1: [100, 200], 2: [110, 210], 3: [120, 220]}
            >>> box = [130, 230, 150, 250]
            >>> track_id_num = 1
            >>> previous_position = (120, 220)
            >>> class_to_count = 0  # In COCO model, class 0 = person
            >>> counter.count_objects((140, 240), track_id_num, previous_position, class_to_count)
        """
        if len(self.region) == 2:  # Linear region: cross-line counting
            if prev_position is None or track_id in self.counted_ids:
                return
            if self.r_s.intersects(self.LineString([prev_position, current_centroid])):
                if self.direction != "any":
                    p1, p2 = self.region[0], self.region[1]
                    prev_side = 1 if self.point_line_side(*prev_position, p1, p2) > 0 else -1
                    curr_side = 1 if self.point_line_side(*current_centroid, p1, p2) > 0 else -1
                    expected = self.DIRECTION_RULES.get(self.direction)
                    if expected and prev_side == expected[0] and curr_side == expected[1]:
                        self.total_count += 1
                        self.in_count += 1
                        self.classwise_count[self.names[cls]]["IN"] += 1
                else:
                    if abs(self.region[0][0] - self.region[1][0]) < abs(self.region[0][1] - self.region[1][1]):
                        if current_centroid[0] > prev_position[0]:  # Moving right
                            self.in_count += 1
                            self.classwise_count[self.names[cls]]["IN"] += 1
                        else:  # Moving left
                            self.out_count += 1
                            self.classwise_count[self.names[cls]]["OUT"] += 1
                    elif current_centroid[1] > prev_position[1]:  # Moving downward
                        self.in_count += 1
                        self.classwise_count[self.names[cls]]["IN"] += 1
                    else:  # Moving upward
                        self.out_count += 1
                        self.classwise_count[self.names[cls]]["OUT"] += 1
                    self.total_count = self.in_count
                self.counted_ids.append(track_id)

        elif len(self.region) > 2:  # Polygonal region: enter+exit counting
            if track_id in self.counted_ids:
                return
            inside = self.r_s.contains(self.Point(current_centroid))

            if self.polygon_mode == "enter_exit":
                if inside:
                    self.entered_ids.add(track_id)
                elif track_id in self.entered_ids:
                    self.total_count += 1
                    self.in_count += 1
                    self.classwise_count[self.names[cls]]["IN"] += 1
                    self.counted_ids.append(track_id)
                    self.entered_ids.discard(track_id)
            else:
                if inside and prev_position is not None:
                    region_width = max(p[0] for p in self.region) - min(p[0] for p in self.region)
                    region_height = max(p[1] for p in self.region) - min(p[1] for p in self.region)
                    if (region_width < region_height and current_centroid[0] > prev_position[0]) or (
                        region_width >= region_height and current_centroid[1] > prev_position[1]
                    ):
                        self.in_count += 1
                        self.classwise_count[self.names[cls]]["IN"] += 1
                    else:
                        self.out_count += 1
                        self.classwise_count[self.names[cls]]["OUT"] += 1
                    self.total_count = self.in_count
                    self.counted_ids.append(track_id)

    def draw_count_overlay(self, plot_im) -> None:
        """Draw the total count as a large overlay on the frame.

        Args:
            plot_im (np.ndarray): The image or frame to draw on.
        """
        label = f"COUNT: {self.total_count}"
        font_scale = max(1.5, plot_im.shape[1] / 800)
        thickness = max(3, int(font_scale * 2))
        (tw, th), _baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        x, y = 20, 50 + th
        cv2.rectangle(plot_im, (x - 10, y - th - 15), (x + tw + 10, y + 15), (0, 0, 0), -1)
        cv2.putText(plot_im, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)

    def display_counts(self, plot_im) -> None:
        """Display object counts on the input image or frame.

        Args:
            plot_im (np.ndarray): The image or frame to display counts on.

        Examples:
            >>> counter = ObjectCounter()
            >>> frame = cv2.imread("image.jpg")
            >>> counter.display_counts(frame)
        """
        labels_dict = {
            str.capitalize(key): f"{'IN ' + str(value['IN']) if self.show_in else ''} "
            f"{'OUT ' + str(value['OUT']) if self.show_out else ''}".strip()
            for key, value in self.classwise_count.items()
            if (value["IN"] != 0 and self.show_in) or (value["OUT"] != 0 and self.show_out)
        }
        if self.total_count > 0:
            labels_dict["Total"] = str(self.total_count)
        if labels_dict:
            self.annotator.display_analytics(plot_im, labels_dict, (104, 31, 17), (255, 255, 255), self.margin)

    def process(self, im0) -> SolutionResults:
        """Process input data (frames or object tracks) and update object counts.

        This method initializes the counting region, extracts tracks, draws bounding boxes and regions, updates object
        counts, and displays the results on the input image.

        Args:
            im0 (np.ndarray): The input image or frame to be processed.

        Returns:
            (SolutionResults): Contains processed image `im0`, 'in_count' (int, count of objects entering the region),
                'out_count' (int, count of objects exiting the region), 'classwise_count' (dict, per-class object
                count), and 'total_tracks' (int, total number of tracked objects).

        Examples:
            >>> counter = ObjectCounter()
            >>> frame = cv2.imread("path/to/image.jpg")
            >>> results = counter.process(frame)
        """
        if not self.region_initialized:
            self.initialize_region()
            self.region_initialized = True

        self.extract_tracks(im0)  # Extract tracks
        self.annotator = SolutionAnnotator(im0, line_width=self.line_width)  # Initialize annotator

        self.annotator.draw_region(
            reg_pts=self.region, color=(104, 0, 123), thickness=self.line_width * 2
        )  # Draw region

        # Iterate over bounding boxes, track ids and classes index
        for box, track_id, cls, conf in zip(self.boxes, self.track_ids, self.clss, self.confs):
            # Draw bounding box and counting region
            self.annotator.box_label(box, label=self.adjust_box_label(cls, conf, track_id), color=colors(cls, True))
            self.store_tracking_history(track_id, box)  # Store track history

            # Store previous position of track for object counting
            prev_position = None
            if len(self.track_history[track_id]) > 1:
                prev_position = self.track_history[track_id][-2]
            self.count_objects(self.track_history[track_id][-1], track_id, prev_position, cls)  # object counting

        plot_im = self.annotator.result()
        self.draw_count_overlay(plot_im)  # Draw large count number on frame
        self.display_counts(plot_im)  # Display the counts on the frame
        self.display_output(plot_im)  # Display output with base class function

        # Return SolutionResults
        return SolutionResults(
            plot_im=plot_im,
            in_count=self.in_count,
            out_count=self.out_count,
            classwise_count=dict(self.classwise_count),
            total_tracks=len(self.track_ids),
            total_count=self.total_count,
        )

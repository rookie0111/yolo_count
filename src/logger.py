import csv
import os
from datetime import datetime
from pathlib import Path


class CountLogger:
    """Logs counting events to a CSV file.

    Attributes:
        log_dir (Path): Directory for log files.
        log_file (Path): Path to the current day's CSV log.

    Examples:
        >>> logger = CountLogger("logs")
        >>> logger.log_count(5, 1)  # Count 5, track_id 1
    """

    def __init__(self, log_dir="logs"):
        """Initialize the CountLogger.

        Args:
            log_dir (str): Directory to store log files.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = None
        self._init_log_file()

    def _get_log_filename(self):
        """Generate log filename based on current date.

        Returns:
            (Path): Log file path.
        """
        date_str = datetime.now().strftime("%Y%m%d")
        return self.log_dir / f"count_{date_str}.csv"

    def _init_log_file(self):
        """Initialize the CSV log file for the current day."""
        self.log_file = self._get_log_filename()
        file_exists = self.log_file.exists()
        with open(self.log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "count", "track_id"])

    def log_count(self, count, track_id):
        """Log a counting event.

        Args:
            count (int): Current total count.
            track_id (int): ID of the tracked object that triggered the count.
        """
        current_file = self._get_log_filename()
        if current_file != self.log_file:
            self._init_log_file()

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, count, track_id])

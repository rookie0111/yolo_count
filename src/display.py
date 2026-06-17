import time


class NumberDisplay:
    """Controls a USB serial number display for showing count values.

    Attributes:
        digits (int): Number of digits on the display.
        ser: Serial connection object.

    Examples:
        >>> display = NumberDisplay(port="/dev/ttyUSB0", baudrate=9600, digits=8)
        >>> display.show_number(12345)
    """

    def __init__(self, port="/dev/ttyUSB0", baudrate=9600, digits=8):
        """Initialize the number display via serial port.

        Args:
            port (str): Serial port path.
            baudrate (int): Baud rate for serial communication.
            digits (int): Number of digits supported by the display.
        """
        self.digits = digits
        self.ser = None
        try:
            import serial

            self.ser = serial.Serial(port, baudrate, timeout=1)
            time.sleep(1)
        except Exception:
            self.ser = None

    def show_number(self, number):
        """Send a number to the display.

        Args:
            number (int): The number to display.
        """
        if self.ser is None:
            return
        text = str(int(number)).zfill(self.digits)
        self.ser.write((text + "\n").encode("ascii"))

    def close(self):
        """Close the serial connection."""
        if self.ser is not None:
            self.ser.close()
            self.ser = None

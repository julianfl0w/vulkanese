# Window name in which image is displayed
from screeninfo import get_monitors
import cv2
import sys


class FullScreen:
    def __init__(self):

        # Get the monitor refresh rate?
        self.display = get_monitors()[0]

        self.window_name = "Image"

        # Create a fullscreen window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        # Set the window properties to fullscreen   
        cv2.setWindowProperty(
            self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )

    def write(self, image):
        # Displaying the image
        cv2.imshow(self.window_name, image)
        key = int(cv2.waitKey(1))
        return key

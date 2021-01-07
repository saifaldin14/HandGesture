import numpy as np
import cv2
from typing import Tuple

def recognize(img_gray):
    """
    Recognizes hand gesture in a single-channel depth image
            This method estimates the number of extended fingers based on
            a single-channel depth image showing a hand and arm region.
            :param img_gray: single-channel depth image
            :returns: (num_fingers, img_draw) The estimated number of
                       extended fingers and an annotated RGB image
    """

    # segment arm region
    segment = segment_arm(img_gray)

    # find the hull of the segmented area, and based on that find the
    # convexity defects
    (contour, defects) = find_hull_defects(segment)

    # detect the number of fingers depending on the contours and convexity
    # defects, then draw defects that belong to fingers green, others red
    img_draw = cv2.cvtColor(segment, cv2.COLOR_GRAY2RGB)
    (num_fingers, img_draw) = detect_num_fingers(contour, defects, img_draw)

    return (num_fingers, img_draw)

def segment_arm(frame: np.ndarray, abs_depth_dev: int = 14) -> np.ndarray:
    """
    Segments arm region
            This method accepts a single-channel depth image of an arm and
            hand region and extracts the segmented arm region.
            It is assumed that the hand is placed in the center of the image.
            :param frame: single-channel depth image
            :returns: binary image (mask) of segmented arm region, where
                      arm=255, else=0
    """

import cv2
import numpy as np
from utils import *

def run():

    left = cv2.imread("images/task7/left.png")
    right = cv2.imread("images/task7/right.png")

    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)

    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    h, w = left_gray.shape

    disparity = np.zeros((h, w), np.uint8)

    block = 5
    max_disp = 64

    for y in range(block, h - block):

        for x in range(block + max_disp, w - block):

            best_offset = 0
            min_error = 1e9

            template = left_gray[y-block:y+block, x-block:x+block]

            for offset in range(max_disp):

                compare = right_gray[y-block:y+block, x-block-offset:x+block-offset]

                error = np.sum(np.abs(template.astype(np.float32) - compare.astype(np.float32)))

                if error < min_error:
                    min_error = error
                    best_offset = offset

            disparity[y, x] = best_offset * 4

    # SAVE RESULTS

    save("output/task7/left.png", left)

    save("output/task7/right.png", right)

    save("output/task7/depth_map.png", disparity)

    # SHOW RESULTS

    show("Left", left)

    show("Right", right)

    show("Depth Map", disparity)


if __name__ == "__main__":
    run()
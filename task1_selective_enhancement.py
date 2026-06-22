import cv2
import numpy as np
from utils import *

def run():
    img = cv2.imread("images/task1/flowers.jpg")
    img = resize(img, 800)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 100, 100])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([160, 100, 100])
    upper2 = np.array([179, 255, 255])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel_morph = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_morph)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_morph)

    temp_sharp = unsharp_mask(img)
    sharp_img = sharpen(temp_sharp)

    blur_img = cv2.GaussianBlur(img, (31, 31), 0)
    
    foreground = cv2.bitwise_and(sharp_img, sharp_img, mask=mask)

    inv_mask = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(blur_img, blur_img, mask=inv_mask)

    result = cv2.add(foreground, background)
    
    save("output/task1/original.png", img)
    save("output/task1/mask.png", mask)
    save("output/task1/output.png", result)
    
    show("Original", img)
    show("Result", result)

if __name__ == "__main__":
    run()
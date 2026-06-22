import cv2
import numpy as np

from utils import *

def run():
    images = [
        "images/task4/doc1.png",
        "images/task4/doc2.png",
        "images/task4/doc3.png"
    ]

    for i, path in enumerate(images):

        print(f"Processing image {i + 1}")

        img = cv2.imread(path)

        img = resize(img, 700)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # METHOD 1
        # Background Correction

        method1 = cv2.medianBlur(gray, 3)

        background = cv2.GaussianBlur(method1, (51, 51),0)

        method1 = cv2.divide(method1,background,scale=255)

        method1 = normalize(method1)

        # METHOD 2
        # Adaptive Threshold

        method2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,51,15)

        # METHOD 3
        # Otsu Threshold

        blur = cv2.GaussianBlur(gray,(3, 3),0)

        _, method3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # SAVE RESULTS

        save(f"output/task4/image{i+1}_original.png",gray)

        save(f"output/task4/image{i+1}_method1.png",method1)

        save(f"output/task4/image{i+1}_method2.png",method2)

        save(f"output/task4/image{i+1}_method3.png",method3)

        # SHOW RESULTS

        show(f"Original {i+1}",gray)

        show(f"Method 1 - Background Correction {i+1}",method1)

        show(f"Method 2 - Adaptive Threshold {i+1}",method2)

        show(f"Method 3 - Otsu Threshold {i+1}",method3)


if __name__ == "__main__":
    run()
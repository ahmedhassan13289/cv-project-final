import cv2
import numpy as np
from utils import *

def enhance_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mean = np.mean(gray)
    std = np.std(gray)

    result = img.copy()

    if mean < 80:
        result = cv2.convertScaleAbs(img, alpha=1.2, beta=40)

    elif mean > 180:
        result = cv2.convertScaleAbs(img, alpha=0.8, beta=-30)

    elif std < 40:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        l, a, b = cv2.split(lab)

        l = histogram_equalization(l)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

        l = clahe.apply(l)

        lab = cv2.merge((l, a, b))

        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return result, mean, std


def run():
    paths = [
        "images/task3/input1.png",
        "images/task3/input2.png",
        "images/task3/input3.png",
        # "images/task3/good.png",
    ]

    for i, path in enumerate(paths):

        img = cv2.imread(path)

        result, mean, std = enhance_image(img)

        print(f"\nImage {i+1}")
        print("Mean Intensity =", mean)
        print("Standard Deviation =", std)

        # SAVE RESULTS

        save(f"output/task3/image{i+1}_original.png", img)

        save(f"output/task3/image{i+1}_enhanced.png", result)

        # SHOW RESULTS

        show(f"Original {i+1}", img)

        show(f"Enhanced {i+1}", result)


if __name__ == "__main__":
    run()
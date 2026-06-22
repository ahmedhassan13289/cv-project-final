import cv2
import numpy as np

from utils import *


def run():
    images = [
        "images/task2/xray1.png",
        "images/task2/xray2.png",
        "images/task2/xray3.png"
    ]

    for i, path in enumerate(images):

        print(f"Processing image {i + 1}")

        img = cv2.imread(path, 0)
        img = resize(img, 700)

        # PIPELINE 1
        # Contrast + Sharpen

        pipeline1 = normalize(img)
        pipeline1 = sharpen(pipeline1)

        # PIPELINE 2
        # CLAHE + Unsharp Mask

        clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8, 8))

        pipeline2 = clahe.apply(img)
        pipeline2 = unsharp_mask(pipeline2)

        # PIPELINE 3
        # Denoise + Laplacian

        pipeline3 = cv2.GaussianBlur(img, (3, 3), 0)

        lap = cv2.Laplacian(pipeline3, cv2.CV_64F)
        lap = cv2.convertScaleAbs(lap)

        pipeline3 = cv2.addWeighted(pipeline3,1.2,lap,0.6,0)

        pipeline3 = normalize(pipeline3)

        # SAVE RESULTS

        save(f"output/task2/image{i+1}_original.png", img)

        save(f"output/task2/image{i+1}_pipeline1.png", pipeline1)

        save(f"output/task2/image{i+1}_pipeline2.png", pipeline2)

        save(f"output/task2/image{i+1}_pipeline3.png", pipeline3)

        # SHOW RESULTS

        show(f"Original {i+1}", img)

        show(f"Pipeline 1 - Image {i+1}",pipeline1)

        show(f"Pipeline 2 - Image {i+1}",pipeline2)

        show(f"Pipeline 3 - Image {i+1}",pipeline3)


if __name__ == "__main__":
    run()

import cv2
import numpy as np
from utils import *


def compute_radiance(images, exposures):

    images = [
        np.power(img.astype(np.float32) / 255.0, 2.2)
        for img in images
    ]

    eps = 1e-6

    hdr_log = np.zeros_like(images[0], dtype=np.float32)
    weight_sum = np.zeros_like(images[0], dtype=np.float32)

    for img, t in zip(images, exposures):

        w = 4.0 * img * (1.0 - img) + eps

        hdr_log += w * (
            np.log(img + eps) - np.log(t)
        )

        weight_sum += w

    hdr = np.exp(hdr_log / weight_sum)

    return hdr


def tone_map(hdr):

    key = 0.22

    avg_lum = np.exp(
        np.mean(np.log(hdr + 1e-6))
    )

    hdr_scaled = (key / avg_lum) * hdr

    ldr = hdr_scaled / (1.0 + hdr_scaled)

    ldr = np.power(ldr, 1 / 2.2)

    low, high = np.percentile(ldr, (0.5, 99.5))

    ldr = (ldr - low) / (high - low + 1e-6)

    return np.clip(ldr * 255, 0, 255).astype(np.uint8)


def run():

    img_dark = cv2.imread("images/task8/dark.jpg")
    img_mid = cv2.imread("images/task8/normal.jpg")
    img_bright = cv2.imread("images/task8/bright.jpg")

    exposures = np.array(
        [1/125.0, 1/60.0, 1/30.0],
        dtype=np.float32
    )

    hdr = compute_radiance(
        [img_dark, img_mid, img_bright],
        exposures
    )

    result = tone_map(hdr)

    # SAVE RESULTS

    save("output/task8/dark_input.png", img_dark)

    save("output/task8/normal_input.png", img_mid)

    save("output/task8/bright_input.png", img_bright)

    save("output/task8/hdr_result.png", result)

    # SHOW RESULTS

    show("Dark Input", img_dark)

    show("Normal Input", img_mid)

    show("Bright Input", img_bright)

    show("HDR Result", result)


if __name__ == "__main__":
    run()
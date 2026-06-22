import cv2
import numpy as np
import matplotlib.pyplot as plt


def show(title, img, cmap=None):
    plt.figure(figsize=(8, 6))

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)

    plt.title(title)
    plt.axis("off")
    plt.show()

def save(path, img):
    cv2.imwrite(path, img)


def resize(img, width=800):
    h, w = img.shape[:2]

    ratio = width / w
    new_h = int(h * ratio)

    return cv2.resize(img, (width, new_h))


def sharpen(img):
    kernel = np.array([
        [0, -1, 0],
        [-1, 5,-1],
        [0, -1, 0]
    ])

    return cv2.filter2D(img, -1, kernel)


def unsharp_mask(img):
    blur = cv2.GaussianBlur(img, (5,5), 1.5)
    sharp = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
    return sharp


def normalize(img):
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)


def histogram_equalization(gray):
    return cv2.equalizeHist(gray)
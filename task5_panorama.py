import cv2
import numpy as np
from utils import *

def stitch_panorama(img1, img2):
    img1 = resize(img1, 700)
    img2 = resize(img2, 700)

    orb = cv2.ORB_create(3000)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        raise ValueError("Not enough features")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.knnMatch(des1, des2, k=2)

    good = []
    for m_n in matches:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 4:
        raise ValueError("Not enough matches")

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(dst, src, cv2.RANSAC, 5.0)

    if H is None:
        raise ValueError("Homography failed")

    h = max(img1.shape[0], img2.shape[0])
    w = img1.shape[1] + img2.shape[1]

    warp = cv2.warpPerspective(img2, H, (w, h))
    
    mask1 = np.zeros((h, w), dtype=np.float32)
    mask1[0:img1.shape[0], 0:img1.shape[1]] = 1.0
    
    mask2 = (cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY) > 0).astype(np.float32)
    
    overlap = (mask1 > 0) & (mask2 > 0)
    
    pano = warp.astype(np.float32)
    img1_f = np.zeros((h, w, 3), dtype=np.float32)
    img1_f[0:img1.shape[0], 0:img1.shape[1]] = img1.astype(np.float32)

    if np.any(overlap):
        overlap_cols = np.where(overlap.any(axis=0))[0]
        min_x, max_x = overlap_cols[0], overlap_cols[-1]
        
        alpha = np.ones((h, w), dtype=np.float32)
        for x in range(min_x, max_x + 1):
            alpha[:, x] = 1.0 - (x - min_x) / (max_x - min_x)
            
        for c in range(3):
            pano[:, :, c] = np.where(overlap, 
                                     img1_f[:, :, c] * alpha + pano[:, :, c] * (1.0 - alpha), 
                                     pano[:, :, c])
            
        only1 = (mask1 > 0) & ~overlap
        pano[only1] = img1_f[only1]
    
    pano = np.clip(pano, 0, 255).astype(np.uint8)

    gray = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if cnts:
        x, y, w_box, h_box = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        pano = pano[y:y+h_box, x:x+w_box]

    return pano


def run():
    img1 = cv2.imread("images/task5/left.jpg")
    img2 = cv2.imread("images/task5/right.jpg")

    result = stitch_panorama(img1, img2)

    save("output/task5/left.png", resize(img1, 700))
    save("output/task5/right.png", resize(img2, 700))
    save("output/task5/panorama.png", result)

    show("Left", img1)
    show("Right", img2)
    show("Panorama", result)

if __name__ == "__main__":
    run()
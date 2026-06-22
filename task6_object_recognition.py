import cv2
import numpy as np
from utils import *

def extract_features(img):

    orb = cv2.ORB_create(nfeatures=1500, scaleFactor=1.2, nlevels=8)

    keypoints, descriptors = orb.detectAndCompute(img, None)

    return keypoints, descriptors


def match_features(des1, des2):

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    knn = bf.knnMatch(des1, des2, k=2)

    good = []

    for pair in knn:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < 0.8 * n.distance:
            good.append(m)

    good = sorted(good, key=lambda x: x.distance)

    return good


def run():

    obj = cv2.imread("images/task6/object.jpg", 0)

    scene = cv2.imread("images/task6/scene.png", 0)

    kp1, des1 = extract_features(obj)
    kp2, des2 = extract_features(scene)

    if des1 is None or des2 is None:
        print("No descriptors found")
        return

    matches = match_features(des1, des2)

    matches = matches[:80]

    result = cv2.drawMatches(obj, kp1, scene, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    print("Object keypoints:", len(kp1))
    print("Scene keypoints:", len(kp2))
    print("Good matches:", len(matches))

    # SAVE RESULTS

    save("output/task6/object.png", obj)

    save("output/task6/scene.png", scene)

    save("output/task6/matches.png", result)

    # SHOW RESULTS

    show("Object", obj)

    show("Scene", scene)

    show("Recognition", result)


if __name__ == "__main__":
    run()
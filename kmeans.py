import numpy as np
import cv2
from scipy.cluster.vq import *


def main():
    # load image
    img = cv2.imread('./image_in.jpg')
    # convert to CIE LUV
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    Z = img2.reshape((-1, 3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria stop, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    K = 9
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(img2.shape)
    # convert to gray scale
    gray = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
    # convert gray scale to binary
    ret, thresh1 = cv2.threshold(gray, 112, 255, cv2.THRESH_TOZERO)
    # save output
    cv2.imwrite('image_out.jpg', thresh1)


if __name__ == '__main__':
    main()

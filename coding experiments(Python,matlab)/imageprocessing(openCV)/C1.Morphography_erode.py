# morphological transformation : erode and dilate
import cv2 as cv
import numpy as np
img = cv.imread('coins.png')
#element structure (circle(d=9) square a=7 circle (d=5))
kernel=cv.getStructuringElement(cv.MORPH_ELLIPSE,(9,9))
kernel1 = np.ones((7,7), np.uint8)
kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
#erode and dilate twice
cv.imshow('cuaso', np.hstack([img, img]))
cv.waitKey(0)

img1 = cv.erode(img, kernel1, iterations=2)
cv.imshow('cuaso', np.hstack([img, img1]))
cv.waitKey(0)

img1 = cv.dilate(img1, kernel2, iterations=2)
cv.imshow('cuaso', np.hstack([img, img1]))
cv.waitKey(0)
cv.destroyAllWindows()
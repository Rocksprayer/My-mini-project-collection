import cv2 as cv
import numpy as np

img=cv.imread('origin.png')
res=cv.resize(img,(0,0),None,0.2,0.2,interpolation = cv.INTER_CUBIC)
cv.imshow('display window',res)
cv.imshow('origin',img)

k=cv.waitKey(0)
if k==ord("s"):
    cv.imwrite("resize.png",res)

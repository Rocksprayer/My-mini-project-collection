import numpy as np
import cv2
def adjust_gamma(image, gamma=1):
    invgamma = 1/gamma
    brighter_image = np.array(np.power((image/255), invgamma)*255, dtype=np.uint8)
    return brighter_image

image1 = cv2.imread('shapes.png',1)
gray=cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
kernel=cv2.getStructuringElement(cv2.MORPH_RECT,[3,3])
gray=adjust_gamma(gray,gamma=2)
# adaptive thresh contour
thresh=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,13,5)
ret,otsu=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
otsu=~otsu
thresh=~thresh
thresh=cv2.morphologyEx(~otsu,cv2.MORPH_OPEN,kernel,iterations=1)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print("Number of Contours found = " + str(len(contours)))
# for c in contours:
#      x,y,w,h = cv2.boundingRect(c)
#      cv2.rectangle(image1, (x, y), (x + w, y + h), (36,255,12), 2)

cv2.drawContours(image1,contours,-1,(0,200,200),3)
cv2.imshow('Contour', image1)
cv2.imshow('Contours', thresh)
cv2.imshow('Contourss', otsu)
cv2.waitKey(0)


# color segment thresh contour
# image = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
# lower = np.array([170, 70, 50], dtype="uint8")
# upper = np.array([180, 255, 255], dtype="uint8")
# mask = cv2.inRange(image, lower, upper)
#
# cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(image1,contours,-1, (0,255,0), 3)
#
#
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
#
# for c in cnts:
#     x,y,w,h = cv2.boundingRect(c)
#     cv2.rectangle(image1, (x, y), (x + w, y + h), (36,255,12), 2)
#
# cv2.imshow('mask', otsu)
# cv2.imshow('original', image1)
cv2.waitKey()
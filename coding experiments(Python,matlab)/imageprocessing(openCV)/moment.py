import numpy as np
import cv2
def adjust_gamma(image, gamma=1):
    invgamma = 1/gamma
    brighter_image = np.array(np.power((image/255), invgamma)*255, dtype=np.uint8)
    return brighter_image

image1 = cv2.imread('hand.png',1)
cv2.imshow('original',image1)
# gray scale
gray=cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
# gamma adjust gam<1 darken gam>1 bright
gray=adjust_gamma(gray,gamma=0.5)
cv2.imshow('adjust',gray)
# black and white extract method otsu addaptive thresh canny
thresh=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,9,5)
ret,otsu=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU,cv2.THRESH_BINARY_INV)
canny=cv2.Canny(gray,50,255)
cv2.imshow('gray',canny)

contours,hierachy=cv2.findContours(canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
#pick out the n contour of contours[n]  and etract moment

for i in contours:
    if cv2.contourArea(i) >50:
        cnt=i
        M=cv2.moments(cnt)
        #centroid
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        #draw centroid
        cv2.circle(image1,(cx,cy),3,(0,255,0),5)
        # cv2.line(image1,cnt[0][0],cnt[10][0],(0,0,255),3)
        cv2.drawContours(image1,cnt,-1,(0,0,255),3)
        print(cv2.contourArea(i))

        epsilon = 0.02*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        for i in range(len(approx)-1):
            cv2.line(image1,approx[i][0],approx[i+1][0],(0,255,0),3)
        hull = cv2.convexHull(cnt)

        #draw hull
        for i in range(len(hull)-1):
            cv2.line(image1,hull[i][0],hull[i+1][0],(255,0,0),3)
    else:
        i+=i




cv2.imshow('center',image1)
cv2.waitKey()
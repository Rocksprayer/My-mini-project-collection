import cv2 as cv
import numpy as np
def nothing(x):
    pass
cap = cv.VideoCapture(0)
cv.namedWindow('TrackBars')
cv.createTrackbar("Hue Min", "TrackBars", 0, 255, nothing)
cv.createTrackbar("Hue Max", "TrackBars", 0, 255,nothing)
cv.createTrackbar("Sat Min", "TrackBars", 0, 255,nothing)
cv.createTrackbar("Sat Max", "TrackBars", 0, 255, nothing)
cv.createTrackbar("Val Min", "TrackBars", 0, 255,nothing)
cv.createTrackbar("Val Max", "TrackBars", 0, 255, nothing)



while(1):
    # Take each frame
    _, frame1 = cap.read()
    frame=cv.GaussianBlur(frame1,(5,5),1)
    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # define range of blue color in HSV
    h_min = cv.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv.getTrackbarPos("Val Max", "TrackBars")

    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    # Threshold the HSV image to get only blue colors
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv.inRange(hsv, lower, upper)
    contours,hierachy=cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
    cv.drawContours(frame,contours,-1,(0,200,200),3)
    #mask = cv.inRange(hsv, lower_blue, upper_blue)
    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame,frame,mask=mask)

    cv.imshow('frame',frame)
    cv.imshow('mask',mask)
    cv.imshow('res',res)
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break
cv.destroyAllWindows()
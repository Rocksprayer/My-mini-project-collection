import cv2
import numpy as np
import imutils
frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

def empty(a):
    pass

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters",640,240)
cv2.createTrackbar("Threshold1","Parameters",50,255,empty)
cv2.createTrackbar("Threshold2","Parameters",50,255,empty)
cv2.createTrackbar("Area","Parameters",5000,30000,empty)

def findColor(img,areaMin, x, y, w, h):

    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #define range of color
    #red
    red_lower = np.array([0,102,32], np.uint8)
    red_upper = np.array([7,255,255], np.uint8)

    #blue
    blue_lower = np.array([103,110,0], np.uint8)
    blue_upper = np.array([141,240,255], np.uint8)
    #yellow
    yellow_lower = np.array([14,85,32], np.uint8)
    yellow_upper = np.array([46,255,255], np.uint8)

    #finding range of three color
    redr = cv2.inRange(imgHSV, red_lower, red_upper)
    bluer = cv2.inRange(imgHSV, blue_lower, blue_upper)
    yellowr = cv2.inRange(imgHSV, yellow_lower, yellow_upper)

    #Morpho
    kernel = np.ones((5,5),"uint8")
    red = cv2.dilate(redr, kernel)
    res = cv2.bitwise_and(img, img, mask = red)

    blue = cv2.dilate(bluer, kernel)
    res1 = cv2.bitwise_and(img, img, mask = blue)

    yellow = cv2.dilate(yellowr, kernel)
    res2 = cv2.bitwise_and(img, img, mask = yellow)

    #find red objects

    contours, hierarchy = cv2.findContours(red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > areaMin:
            cv2.putText(imgContour, "Red",
                        (x + w + 60, y + 65), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        (0, 255, 0), 2)

     #find blue objects

    contours, hierarchy = cv2.findContours(blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > areaMin:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.putText(imgContour, "Blue",
                        (x + w + 70, y + 75), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        (0, 255, 0), 2)

     #find yellow objects
    contours, hierarchy = cv2.findContours(yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > areaMin:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.putText(imgContour, "yellow",
                        (x + w + 70, y + 75), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        (0, 255, 0), 2)
    cv2.imshow("yellow", res2)
def getContours(img, imgContour):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areaMin = cv2.getTrackbarPos("Area", "Parameters")
        if area > areaMin:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 5)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            print(len(approx))
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 6)

            cv2.putText(imgContour, "Points: " + str(len(approx)), (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        (0, 255, 0), 2)
            cv2.putText(imgContour, "Area: " + str(int(area)), (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                        (0, 255, 0), 2)
    #find the centroid
    cnts = cv2.findContours(imgDil, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        area = cv2.contourArea(c)
        areaMin = cv2.getTrackbarPos("Area", "Parameters")
        if area > areaMin:
            area = cv2.contourArea(c)
            M = cv2.moments(c)

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(imgContour, (cx, cy), 4, (65, 150, 255), -1)

while True:
    success, img = cap.read()
    imgContour = img.copy()

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
    imgCanny = cv2.Canny(imgBlur,threshold1,threshold2)

    kernel = np.ones((5, 5))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
    areaMin = cv2.getTrackbarPos("Area", "Parameters")
    getContours(imgDil,imgContour)
    findColor(img,areaMin, x, y, w, h)
    # getContours(imgDil,imgContour)

    #Show results

    cv2.imshow("blur", imgBlur)
    cv2.imshow("img", img)
    cv2.imshow("imgCanny", imgCanny)
    cv2.imshow("imgDil", imgDil)
    cv2.imshow("imgContour", imgContour)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
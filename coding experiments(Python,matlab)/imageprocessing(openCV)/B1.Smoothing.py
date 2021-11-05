import numpy as np
import cv2 as cv
def nothing(x):
    pass
from matplotlib import pyplot as plt
img = cv.imread('noodles.png')
cv.namedWindow('image')
# create trackbars for color change
cv.createTrackbar('color','image',0,10000,nothing)
cv.createTrackbar('space','image',0,10,nothing)

kernel = np.ones((5,5),np.float32)/25
dst = cv.filter2D(img,-1,kernel)
th1=cv.GaussianBlur(img,(5,5),1)


median = cv.medianBlur(img,5)

titles = ['Original Image', 'kernel',
            'AVG','blur','median']
images2 = [img, dst, th1,median]
while(1):
    # get current positions of four trackbars
    sigmaColor = cv.getTrackbarPos('color','image')
    sigmaSpace = cv.getTrackbarPos('space','image')
    blur = cv.bilateralFilter(img,5,sigmaColor=sigmaColor,sigmaSpace=sigmaSpace)
    cv.imshow('blur',blur)
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break


# cv.imshow('bilateral filter',blur)
cv.waitKey(0)
cv.destroyAllWindows()
# for i in range(5):
#     plt.subplot(2,3,i+1),
#     image = cv.cvtColor(images2[i], cv.COLOR_BGR2RGB)
#     plt.imshow(image)
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])
# plt.show()
# plt.subplot(221),plt.imshow(img),plt.title('Original')
# plt.xticks([]), plt.yticks([])
# plt.subplot(222),plt.imshow(dst),plt.title('kernel')
# plt.xticks([]), plt.yticks([])
# plt.subplot(223),plt.imshow(th1),plt.title('avg')
# plt.xticks([]), plt.yticks([])
# plt.subplot(224),plt.imshow(blur),plt.title('bilateral')
# plt.xticks([]), plt.yticks([])

# plt.show()
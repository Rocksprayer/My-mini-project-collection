import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
img = cv.imread('noisy.png',0)
# img = cv.GaussianBlur(img,[3,3],5)

ret,th1 = cv.threshold(img,125,255,cv.THRESH_BINARY)
img1 = cv.GaussianBlur(img,[5,5],1)
th2 = cv.adaptiveThreshold(img1,255,cv.ADAPTIVE_THRESH_MEAN_C,
            cv.THRESH_BINARY,5,3)

# th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,
#             cv.THRESH_BINARY,11,2)
# ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
# # Otsu's thresholding after Gaussian filtering
blur = cv.GaussianBlur(img,(5,5),0)
ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
# plot all the images and their histograms
titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Otsu Thresholding']
images = [img, th1, th2, th3]
for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
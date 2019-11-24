import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
from PIL import Image

src1 = cv2.imread('../HSCluster1.png')
# cv2.imshow('Imagen Cluster 1', src1)

src2 = cv2.imread('../HSCluster2.png')
# cv2.imshow('Imagen Cluster 2', src2)

blur1 = cv2.GaussianBlur(src1,(7,7),0)
threshOp1 = cv2.threshold(blur1, 100, 255, cv2.THRESH_BINARY)[1]
# cv2.imshow("Gaussian Cluster 1", threshOp1)

blur2 = cv2.GaussianBlur(src2,(7,7),0)
threshOp2 = cv2.threshold(blur2, 100, 255, cv2.THRESH_BINARY)[1]
# cv2.imshow("Gaussian Cluster 2", threshOp2)

imageGrayRGB1 = cv2.cvtColor(src1,cv2.COLOR_RGB2GRAY)
ret1, thresh1 = cv2.threshold(imageGrayRGB1, 127, 255, 0)
contours1, hierarchy1 = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
imageRGBThreshColor1 = cv2.cvtColor(imageGrayRGB1,cv2.COLOR_GRAY2RGB)
cv2.drawContours(imageRGBThreshColor1, contours1, -1, (255,255,255), 3)
# cv2.imshow("Filled Cluster 1", imageRGBThreshColor1)

imageGrayRGB2 = cv2.cvtColor(src2,cv2.COLOR_RGB2GRAY)
ret2, thresh2 = cv2.threshold(imageGrayRGB2, 127, 255, 0)
contours2, hierarchy2 = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
imageRGBThreshColor2 = cv2.cvtColor(imageGrayRGB2,cv2.COLOR_GRAY2RGB)
cv2.drawContours(imageRGBThreshColor2, contours2, -1, (255,255,255), 3)
# cv2.imshow("Filled Cluster 2", imageRGBThreshColor2)

blur1 = cv2.GaussianBlur(imageRGBThreshColor1,(13,13),0)
threshOp1 = cv2.threshold(blur1, 100, 255, cv2.THRESH_BINARY)[1]
# cv2.imshow("Final image Cluster 1", threshOp1)
threshOpGray1 = cv2.cvtColor(threshOp1,cv2.COLOR_RGB2GRAY)
contours1, hierarchy1 = cv2.findContours(threshOpGray1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
ImageFinal1 = threshOp1.copy()
cv2.drawContours(threshOp1, contours1, -1, (0,255,0), 1)
# cv2.imshow("Contours Cluster 1", threshOp1)
for contour in contours1:
    print(cv2.contourArea(contour))
# cv2.waitKey(0)

plt.figure()
plt.suptitle('Detección de parámetros cluster 1')
plt.subplot(2,2,1)
plt.imshow(src1)
plt.title("Imagen original")
plt.xlabel("Horizontal Pixels")
plt.ylabel("Vertical Pixels")
plt.subplot(2,2,2)
plt.imshow(ImageFinal1)
plt.title("Imagen Final sin Contornos")
plt.xlabel("Horizontal Pixels")
plt.ylabel("Vertical Pixels")
plt.subplot(2,2,3)
plt.imshow(threshOp1)
plt.title("Imagen Final con Contornos")
plt.xlabel("Horizontal Pixels")
plt.ylabel("Vertical Pixels")

cv2.imwrite("ImagenFinalCluster1.png",threshOp1)

blur2 = cv2.GaussianBlur(imageRGBThreshColor2,(13,13),0)
threshOp2 = cv2.threshold(blur2, 100, 255, cv2.THRESH_BINARY)[1]
# cv2.imshow("Final image Cluster 2", threshOp2)
threshOpGray2 = cv2.cvtColor(threshOp2,cv2.COLOR_RGB2GRAY)
contours2, hierarchy2 = cv2.findContours(threshOpGray2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
ImageFinal2 = threshOp2.copy()
cv2.drawContours(threshOp2, contours2, -1, (0,255,0), 1)
# cv2.imshow("Contours Cluster 2", threshOp2)
for contour in contours2:
    print(cv2.contourArea(contour))
# cv2.waitKey(0)


plt.figure()
plt.suptitle('Detección de parámetros cluster 2')
plt.subplot(2,2,1)
plt.imshow(src2)
plt.title("Imagen original")
plt.xlabel("Horizontal Pixels")
plt.ylabel("Vertical Pixels")
plt.subplot(2,2,2)
plt.imshow(ImageFinal2)
plt.title("Imagen Final sin Contornos")
plt.xlabel("Horizontal Pixels")
plt.ylabel("Vertical Pixels")
plt.subplot(2,2,3)
plt.imshow(threshOp2)
plt.title("Imagen Final con Contornos")
plt.xlabel("Horizontal Pixels")
plt.ylabel("Vertical Pixels")
plt.show()

cv2.imwrite("ImagenFinalCluster2.png",threshOp2)
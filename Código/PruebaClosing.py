import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
from PIL import Image

src = cv2.imread('../Cluster.png')
cv2.imshow('', src)
height, width, depth = src.shape

# kernel = np.zeros((5,5),np.uint8)
# kernel[:] = np.asarray(([0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0,],[0,0,1,0,0]))
# kernelO = np.asarray(([0,1,0],[1,1,1],[0,1,0]),dtype=np.uint8)
# ImageClose = cv2.morphologyEx(src,cv2.MORPH_OPEN,kernelO)
# cv2.imshow('Image Closed O',ImageClose)

blur = cv2.GaussianBlur(src,(7,7),0)
threshOp = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)[1]
cv2.imshow("Opening", threshOp)


# CloseCoef = 5
# ImageClose = np.zeros((height,width,depth,CloseCoef+1),dtype = np.uint8)
# ImageClose[:,:,:,0] = src
# for i in range(1,CloseCoef+1):
#     ImageClose[:,:,:,i] = cv2.morphologyEx(ImageClose[:,:,:,i-1],cv2.MORPH_CLOSE,kernel)
#     cv2.imshow('Image Closed ' + str(i),ImageClose[:,:,:,i])
# cv2.waitKey(0)
kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3));  
ImageErosion = cv2.erode(src,kernel2,iterations=1)
cv2.imshow("Erosion", ImageErosion)

#These lines find and draw figure's contours
imageGrayRGB = cv2.cvtColor(src,cv2.COLOR_RGB2GRAY)
ret, thresh = cv2.threshold(imageGrayRGB, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
imageRGBThreshColor = cv2.cvtColor(imageGrayRGB,cv2.COLOR_GRAY2RGB)
cv2.drawContours(imageRGBThreshColor, contours, -1, (255,255,255), 5)
cv2.imshow("Contours", imageRGBThreshColor)
# for contour in contours:
#     print(cv2.contourArea(contour))

blur = cv2.GaussianBlur(imageRGBThreshColor,(13,13),0)
threshOp = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)[1]
cv2.imshow("Opening 2", threshOp)
cv2.waitKey(0)

#These lines find and draw figure's contours
imageGrayRGB = cv2.cvtColor(threshOp,cv2.COLOR_RGB2GRAY)
ret, thresh = cv2.threshold(imageGrayRGB, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
imageRGBThreshColor = cv2.cvtColor(imageGrayRGB,cv2.COLOR_GRAY2RGB)
cv2.drawContours(imageRGBThreshColor, contours, -1, (255,255,255), 5)
cv2.imshow("Contours 2", imageRGBThreshColor)
# for contour in contours:0
#     print(cv2.contourArea(contour))

blur = cv2.GaussianBlur(imageRGBThreshColor,(13,13),0)
threshOp = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)[1]
cv2.imshow("Opening 3", threshOp)
threshOpGray = cv2.cvtColor(threshOp,cv2.COLOR_RGB2GRAY)
contours, hierarchy = cv2.findContours(threshOpGray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(threshOp, contours, -1, (0,255,0), 1)
cv2.imshow("Contours 3", threshOp)
for contour in contours:
    print(cv2.contourArea(contour))
cv2.waitKey(0)

# kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5));  
# ImageErosion = cv2.erode(imageRGBThreshColor,kernel2,iterations=2)
# cv2.imshow("Erosion 2", ImageErosion)
# cv2.waitKey(0)

# kernelO = cv2.getStructuringElement(cv2.MORPH_CROSS, (9,9)); 
# ImageOp = cv2.morphologyEx(imageRGBThreshColor,cv2.MORPH_OPEN,kernelO)
# cv2.imshow('Image Open',ImageOp)
# cv2.waitKey(0)
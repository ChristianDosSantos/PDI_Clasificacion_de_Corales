import cv2
import numpy as np
import matplotlib.pyplot as pyplot


src = cv2.imread('../circleAndlines.png')
cv2.imshow('Example',src)

kernel = np.zeros((5,5),np.uint8)
kernel[:] = np.asarray(([0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0,],[0,0,1,0,0]))

# for i in range(0,5,1):
#     procImage = cv2.erode(src,kernel,iterations=i+1)
#     cv2.imshow('Imagen procesada ' + str(i),procImage)

procImage = cv2.erode(src,kernel,iterations=2)
cv2.imshow('Imagen procesada',procImage)

openImage = cv2.morphologyEx(procImage,cv2.MORPH_OPEN,kernel)
cv2.imshow('Imagen Abierta',openImage)

openImage2 = cv2.morphologyEx(openImage,cv2.MORPH_OPEN,kernel)
cv2.imshow('Imagen Abierta 2',openImage2)
cv2.waitKey(0)

# imgray = cv2.cvtColor(openImage2, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(imgray, 127, 255, 0)
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# final = cv2.drawContours(thresh, contours, -1, (0,255,0), 3)
# cv2.imshow('Contours',final)
# cv2.imshow('imgray2',imgray)

imageFinal = cv2.dilate(openImage2,kernel,iterations=2)
cv2.imshow('Imagen Final', imageFinal)
cv2.waitKey(0)
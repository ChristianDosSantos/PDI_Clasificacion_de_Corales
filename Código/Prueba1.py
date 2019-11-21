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

closeImage = cv2.morphologyEx(openImage,cv2.MORPH_CLOSE,kernel)
cv2.imshow('Imagen Cerrada 2',closeImage)

imgray = cv2.cvtColor(closeImage, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
cv2.imshow('imgray2',imgray)
if cv2.__version__.startswith("3."):
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
else:
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(contours)
## Draw contours 
cv2.drawContours(closeImage, contours, -1, (0, 255, 0), 1)
cv2.imshow('Contours',closeImage)


imageFinal = cv2.dilate(openImage2,kernel,iterations=2)
cv2.imshow('Imagen Final', imageFinal)
cv2.waitKey(0)
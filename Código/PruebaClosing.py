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

kernel = np.zeros((5,5),np.uint8)
kernel[:] = np.asarray(([0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0,],[0,0,1,0,0]))
kernelO = np.asarray(([0,1,0],[1,1,1],[0,1,0]),dtype=np.uint8)
ImageClose = cv2.morphologyEx(src,cv2.MORPH_OPEN,kernelO)
cv2.imshow('Image Closed O',ImageClose)




CloseCoef = 5
ImageClose = np.zeros((height,width,depth,CloseCoef+1),dtype = np.uint8)
ImageClose[:,:,:,0] = src
for i in range(1,CloseCoef+1):
    ImageClose[:,:,:,i] = cv2.morphologyEx(ImageClose[:,:,:,i-1],cv2.MORPH_CLOSE,kernel)
    cv2.imshow('Image Closed ' + str(i),ImageClose[:,:,:,i])
cv2.waitKey(0)
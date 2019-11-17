import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
from PIL import Image

# Constants
colors = [[0, 255, 255], [255, 255, 0], [0, 255, 0], [255, 0, 0], [0, 0, 255], [255, 0, 255], [255, 128, 0], [
    128, 128, 128], [153, 51, 255], [255, 255, 255]]  # Cyan, yellow, green, red, blue, magenta, orange, gray, purple, white

# These lines read the image and convert it to various forms
imageBGR = cv2.imread("../FotoCoral1.jpg")
imageRGB = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)
imageHSV = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2HSV)
imageGray = cv2.cvtColor(imageBGR, cv2.cv2.COLOR_BGR2GRAY)

# This portion of code calculates the LBP of the image
height, width = imageGray.shape
imageLBP = np.zeros((height, width), dtype=np.uint8)
for i in range(0, height, 1):
    for j in range(0, width, 1):
        pot = 1
        for x in range(i-1, i+2):
            for y in range(j-1, j+2, 1):
                if x < 0 or y < 0 or x >= height or y >= width:
                    pot *= 2
                    continue
                elif x == i and y == j:
                    continue
                elif imageGray[x, y] >= imageGray[i, j]:
                    imageLBP[i, j] += pot
                    pot *= 2
                else:
                    pot *= 2
imageLBP2 = Image.fromarray(imageLBP, 'L')
# imageLBP2.show()

# these lines use k-means with Hue, Saturation and Texture as his features
# This portion prepares the features matrix
features3D = imageHSV
features3D[:, :, 2] = imageLBP
features2D = np.zeros((height*width, 3))
for i in range(0, height, 1):
    for j in range(0, width, 1):
        features2D[(i)*width + j, 0] = features3D[i, j, 0]
        features2D[(i)*width + j, 1] = features3D[i, j, 1]
        features2D[(i)*width + j, 2] = features3D[i, j, 2]
features2D = np.float32(features2D)

# This portion calculates the result using k-means
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
attempts = 10
result_image = np.zeros((height, width, 3, 6), dtype=np.uint8)
for K in range(3, 9, 1):
    ret, label, center = cv2.kmeans(
        features2D, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    # Hdiv = 255//K
    # result_image = img
    # print(label)
    # for i in range(0,height,1):
    #     for j in range(0,width,1):
    #         result_image[i,j,0] = Hdiv*(label[(i)*width + j]+1)
    # result_image[:,:,1:2] = 128
    # result_image = cv2.merge((result_image[:,:,0],result_image[:,:,1],img[:,:,2]))
    for i in range(0, height, 1):
        for j in range(0, width, 1):
            result_image[i, j, :, K-3] = colors[label[(i)*width + j, 0]]
    # result_image[:,:,1:2] = 128
    # result_image = cv2.merge((result_image[:,:,0],result_image[:,:,1],img[:,:,2]))
    # cv2.imshow("K-means with K=%i" % K,cv2.cvtColor(result_image,cv2.COLOR_HSV2BGR))

plt.figure()
plt.suptitle("Image Segmentation usign K-means (HS + LBP)")
plt.subplot(3, 2, 1)
plt.imshow(result_image[:, :, :, 0])
plt.subplot(3, 2, 2)
plt.imshow(result_image[:, :, :, 1])
plt.subplot(3, 2, 3)
plt.imshow(result_image[:, :, :, 2])
plt.subplot(3, 2, 4)
plt.imshow(result_image[:, :, :, 3])
plt.subplot(3, 2, 5)
plt.imshow(result_image[:, :, :, 4])
plt.subplot(3, 2, 6)
plt.imshow(result_image[:, :, :, 5])
plt.show()

# These lines draw in category shape in white for better viewing
K = 7
ret, label, center = cv2.kmeans(
        features2D, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
imageRGBThresh = imageRGB
imageRGBThresh1 = np.zeros((height,width,3,K), dtype = np.uint8)
for i in range(0, height, 1):
    for j in range(0, width, 1):
        imageRGBThresh[i,j,:] = colors[label[(i)*width + j, 0]]
        for y in range(0,K,1):
            if (imageRGBThresh[i,j,:] == colors[y]).all():
                imageRGBThresh1[i,j,:,y] = [255,255,255]
            else:
                imageRGBThresh1[i,j,:,y] = [0,0,0]

plt.figure()
for i in range(0,K,1):
    if i > 3:
        plt.subplot(2,2,i-3)
        plt.imshow(imageRGBThresh1[:,:,:,i])
    else:
        plt.subplot(2,2,i+1)
        plt.imshow(imageRGBThresh1[:,:,:,i])
    if i == 3:
        plt.figure()
plt.show()

# This implements K-means for RGB
features2D = imageRGB.reshape((-1, 3))
features2D = np.float32(features2D)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
attempts = 10
result_image = np.zeros((height, width, 3, 6), dtype=np.uint8)
for K in range(3, 9, 1):
    ret, label, center = cv2.kmeans(
        features2D, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image[:, :, :, K-3] = res.reshape((imageRGB.shape))
    # cv2.imshow("K-means with K=%i" % K,result_image)

plt.figure()
plt.suptitle("Image Segmentation usign K-means (RGB)")
plt.subplot(3, 2, 1)
plt.imshow(result_image[:, :, :, 0])
plt.subplot(3, 2, 2)
plt.imshow(result_image[:, :, :, 1])
plt.subplot(3, 2, 3)
plt.imshow(result_image[:, :, :, 2])
plt.subplot(3, 2, 4)
plt.imshow(result_image[:, :, :, 3])
plt.subplot(3, 2, 5)
plt.imshow(result_image[:, :, :, 4])
plt.subplot(3, 2, 6)
plt.imshow(result_image[:, :, :, 5])
plt.show()

K = 7
ret, label, center = cv2.kmeans(
        features2D, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
imageRGBThresh = imageRGB
imageRGBThresh1 = np.zeros((height,width,3,K), dtype = np.uint8)
for i in range(0, height, 1):
    for j in range(0, width, 1):
        imageRGBThresh[i,j,:] = colors[label[(i)*width + j, 0]]
        for y in range(0,K,1):
            if (imageRGBThresh[i,j,:] == colors[y]).all():
                imageRGBThresh1[i,j,:,y] = [255,255,255]
            else:
                imageRGBThresh1[i,j,:,y] = [0,0,0]

plt.figure()
for i in range(0,K,1):
    if i > 3:
        plt.subplot(2,2,i-3)
        plt.imshow(imageRGBThresh1[:,:,:,i])
    else:
        plt.subplot(2,2,i+1)
        plt.imshow(imageRGBThresh1[:,:,:,i])
    if i == 3:
        plt.figure()
plt.show()
# cv2.imshow('Thresh',imageRGBThresh1) 
# cv2.waitKey(0)

# This implements K-means for BGR
features2D = imageBGR.reshape((-1, 3))
features2D = np.float32(features2D)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
attempts = 10
result_image = np.zeros((height, width, 3, 6), dtype=np.uint8)
for K in range(3, 9, 1):
    ret, label, center = cv2.kmeans(
        features2D, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image[:, :, :, K-3] = res.reshape((imageRGB.shape))
    # cv2.imshow("K-means with K=%i" % K,result_image)

plt.figure()
plt.suptitle("Image Segmentation usign K-means (BGR)")
plt.subplot(3, 2, 1)
plt.imshow(result_image[:, :, :, 0])
plt.subplot(3, 2, 2)
plt.imshow(result_image[:, :, :, 1])
plt.subplot(3, 2, 3)
plt.imshow(result_image[:, :, :, 2])
plt.subplot(3, 2, 4)
plt.imshow(result_image[:, :, :, 3])
plt.subplot(3, 2, 5)
plt.imshow(result_image[:, :, :, 4])
plt.subplot(3, 2, 6)
plt.imshow(result_image[:, :, :, 5])
plt.show()

#This implements K-means for HS
features2D = imageHSV[:,:,:2].reshape((-1,2))
features2D = np.float32(features2D)

result_image = np.zeros((height, width, 3, 6), dtype=np.uint8)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
attempts=10
for K in range(3,9,1):
    ret,label,center=cv2.kmeans(features2D,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image[:,:,:2,K-3] = res.reshape((height,width,2))
    result_image[:,:,:,K-3] = cv2.merge((result_image[:,:,0,K-3],result_image[:,:,1,K-3],imageHSV[:,:,2]))
    result_image[:,:,:,K-3] = cv2.cvtColor(result_image[:,:,:,K-3], cv2.COLOR_HSV2RGB)
    # cv2.imshow("K-means with K=%i" % K,cv2.cvtColor(result_image,cv2.COLOR_HSV2BGR))

plt.figure()
plt.suptitle("Image Segmentation usign K-means (HS)")
plt.subplot(3, 2, 1)
plt.imshow(result_image[:, :, :, 0])
plt.subplot(3, 2, 2)
plt.imshow(result_image[:, :, :, 1])
plt.subplot(3, 2, 3)
plt.imshow(result_image[:, :, :, 2])
plt.subplot(3, 2, 4)
plt.imshow(result_image[:, :, :, 3])
plt.subplot(3, 2, 5)
plt.imshow(result_image[:, :, :, 4])
plt.subplot(3, 2, 6)
plt.imshow(result_image[:, :, :, 5])
plt.show()

K = 7
ret, label, center = cv2.kmeans(
        features2D, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
imageRGBThresh = imageRGB
imageRGBThresh1 = np.zeros((height,width,3,K), dtype = np.uint8)
for i in range(0, height, 1):
    for j in range(0, width, 1):
        imageRGBThresh[i,j,:] = colors[label[(i)*width + j, 0]]
        for y in range(0,K,1):
            if (imageRGBThresh[i,j,:] == colors[y]).all():
                imageRGBThresh1[i,j,:,y] = [255,255,255]
            else:
                imageRGBThresh1[i,j,:,y] = [0,0,0]

plt.figure()
for i in range(0,K,1):
    if i > 3:
        plt.subplot(2,2,i-3)
        plt.imshow(imageRGBThresh1[:,:,:,i])
    else:
        plt.subplot(2,2,i+1)
        plt.imshow(imageRGBThresh1[:,:,:,i])
    if i == 3:
        plt.figure()
plt.show()

#This implements K-means with LBP
features2D = imageLBP.reshape((-1,1))
features2D = np.float32(features2D)

result_image = np.zeros((height, width, 3, 6), dtype=np.uint8)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
attempts=10
for K in range(3,9,1):
    ret,label,center=cv2.kmeans(features2D,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    for i in range(0, height, 1):
        for j in range(0, width, 1):
            result_image[i, j, :, K-3] = colors[label[(i)*width + j, 0]]
    # cv2.imshow("K-means with K=%i" % K,cv2.cvtColor(result_image,cv2.COLOR_HSV2BGR))

plt.figure()
plt.suptitle("Image Segmentation usign K-means (LBP)")
plt.subplot(3, 2, 1)
plt.imshow(result_image[:, :, :, 0])
plt.subplot(3, 2, 2)
plt.imshow(result_image[:, :, :, 1])
plt.subplot(3, 2, 3)
plt.imshow(result_image[:, :, :, 2])
plt.subplot(3, 2, 4)
plt.imshow(result_image[:, :, :, 3])
plt.subplot(3, 2, 5)
plt.imshow(result_image[:, :, :, 4])
plt.subplot(3, 2, 6)
plt.imshow(result_image[:, :, :, 5])
plt.show()

K = 7
ret, label, center = cv2.kmeans(
        features2D, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
imageRGBThresh = imageRGB
imageRGBThresh1 = np.zeros((height,width,3,K), dtype = np.uint8)
for i in range(0, height, 1):
    for j in range(0, width, 1):
        imageRGBThresh[i,j,:] = colors[label[(i)*width + j, 0]]
        for y in range(0,K,1):
            if (imageRGBThresh[i,j,:] == colors[y]).all():
                imageRGBThresh1[i,j,:,y] = [255,255,255]
            else:
                imageRGBThresh1[i,j,:,y] = [0,0,0]

plt.figure()
for i in range(0,K,1):
    if i > 3:
        plt.subplot(2,2,i-3)
        plt.imshow(imageRGBThresh1[:,:,:,i])
    else:
        plt.subplot(2,2,i+1)
        plt.imshow(imageRGBThresh1[:,:,:,i])
    if i == 3:
        plt.figure()
plt.show()
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
    128, 128, 128], [153, 51, 255], [255, 255, 255], [0, 0, 0], [255, 102, 178]]  # Cyan, yellow, green, red, blue, magenta, orange, gray, purple, white, black, pink

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
plt.figure()
plt.suptitle('Imagen original y su imagen de textura')
plt.subplot(1,2,1)
plt.imshow(imageRGB)
plt.title('Imagen Original')
plt.ylabel('Píxeles Verticales')
plt.xlabel('Píxeles Horizontales')
plt.subplot(1,2,2)
plt.imshow(imageLBP2, cmap='gray', vmin=0, vmax=255)
plt.title('Imagen de Textura (LBP)')
plt.ylabel('Píxeles Verticales')
plt.xlabel('Píxeles Horizontales')
plt.show()

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
Kmax = 12
Kini = 3
result_image = np.zeros((height, width, 3, Kmax+1-Kini), dtype=np.uint8)
for K in range(Kini, Kmax+1, 1):
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
for i in range(0,Kmax-Kini+1,1):
    if i > 7:
        plt.subplot(2,2,i-7)
        plt.imshow(result_image[:, :, :, i])
        plt.title('Image Segmentation for K=' + str(i+Kini))
        plt.xlabel('Horizontal Pixels')
        plt.ylabel('Vertical Pixels')
    elif i > 3:
        plt.subplot(2,2,i-3)
        plt.imshow(result_image[:, :, :, i])
        plt.title('Image Segmentation for K=' + str(i+Kini))
        plt.xlabel('Horizontal Pixels')
        plt.ylabel('Vertical Pixels')
    else:
        plt.subplot(2,2,i+1)
        plt.imshow(result_image[:, :, :, i])
        plt.title('Image Segmentation for K=' + str(i+Kini))
        plt.xlabel('Horizontal Pixels')
        plt.ylabel('Vertical Pixels')
    if i == 3 or i == 7:
        plt.figure()
        plt.suptitle("Image Segmentation usign K-means (HS + LBP)")
plt.show()

# These lines implement Elbow's method
# distortions = [] 
# inertias = [] 
# mapping1 = {} 
# mapping2 = {} 
# K = range(1,10) 
  
# for k in K: 
#     #Building and fitting the model 
#     kmeanModel = KMeans(n_clusters=k).fit(features2D) 
#     kmeanModel.fit(features2D)     
      
#     distortions.append(sum(np.min(cdist(features2D, kmeanModel.cluster_centers_, 
#                       'euclidean'),axis=1)) / features2D.shape[0]) 
#     inertias.append(kmeanModel.inertia_) 
  
#     mapping1[k] = sum(np.min(cdist(features2D, kmeanModel.cluster_centers_, 
#                  'euclidean'),axis=1)) /features2D.shape[0] 
#     mapping2[k] = kmeanModel.inertia_ 

# plt.figure()
# plt.plot(K, distortions, 'bx-') 
# plt.xlabel('Values of K') 
# plt.ylabel('Distortion') 
# plt.title('The Elbow Method using Distortion') 
# plt.show() 

# These lines draw in category shape in white for better viewing
K = 12
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
plt.suptitle('Image cluster for K=' + str(K))
for i in range(0,K,1):
    if i > 7:
        plt.subplot(2,2,i-7)
        plt.imshow(imageRGBThresh1[:,:,:,i])
        plt.title('Cluster ' + str(i+1))
        plt.xlabel('Horizontal Pixels')
        plt.ylabel('Vertical Pixels')
    elif i > 3 and i<=7:
        plt.subplot(2,2,i-3)
        plt.imshow(imageRGBThresh1[:,:,:,i])
        
        plt.title('Cluster ' + str(i+1))
        plt.xlabel('Horizontal Pixels')
        plt.ylabel('Vertical Pixels')
    else:
        plt.subplot(2,2,i+1)
        plt.imshow(imageRGBThresh1[:,:,:,i])
        plt.title('Cluster ' + str(i+1))
        plt.xlabel('Horizontal Pixels')
        plt.ylabel('Vertical Pixels')
    if i == 3 or i == 7:
        plt.figure()
        plt.suptitle('Image cluster for K=' + str(K))
plt.show()

# This implements K-means for RGB
features2D = imageRGB.reshape((-1, 3))
features2D = np.float32(features2D)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
attempts = 10
Kmax = 12
Kini = 3
result_image = np.zeros((height, width, 3, Kmax+1-Kini), dtype=np.uint8)
for K in range(Kini, Kmax+1, 1):
    ret, label, center = cv2.kmeans(
        features2D, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image[:, :, :, K-3] = res.reshape((imageRGB.shape))
    # cv2.imshow("K-means with K=%i" % K,result_image)

plt.figure()
plt.suptitle("Image Segmentation usign K-means (RGB)")
for i in range(0,Kmax-Kini+1,1):
    if i > 7:
        plt.subplot(2,2,i-7)
        plt.imshow(result_image[:, :, :, i])
        plt.title('Image Segmentation for K=' + str(i+Kini))
        plt.xlabel('Horizontal Pixels')
        plt.ylabel('Vertical Pixels')
    elif i > 3:
        plt.subplot(2,2,i-3)
        plt.imshow(result_image[:, :, :, i])
        plt.title('Image Segmentation for K=' + str(i+Kini))
        plt.xlabel('Horizontal Pixels')
        plt.ylabel('Vertical Pixels')
    else:
        plt.subplot(2,2,i+1)
        plt.imshow(result_image[:, :, :, i])
        plt.title('Image Segmentation for K=' + str(i+Kini))
        plt.xlabel('Horizontal Pixels')
        plt.ylabel('Vertical Pixels')
    if i == 3 or i == 7:
        plt.figure()
        plt.suptitle("Image Segmentation usign K-means (RGB)")
plt.show()

# These lines implement Elbow's method
# distortions = [] 
# inertias = [] 
# mapping1 = {} 
# mapping2 = {} 
# K = range(1,10) 
  
# for k in K: 
#     #Building and fitting the model 
#     kmeanModel = KMeans(n_clusters=k).fit(features2D) 
#     kmeanModel.fit(features2D)     
      
#     distortions.append(sum(np.min(cdist(features2D, kmeanModel.cluster_centers_, 
#                       'euclidean'),axis=1)) / features2D.shape[0]) 
#     inertias.append(kmeanModel.inertia_) 
  
#     mapping1[k] = sum(np.min(cdist(features2D, kmeanModel.cluster_centers_, 
#                  'euclidean'),axis=1)) /features2D.shape[0] 
#     mapping2[k] = kmeanModel.inertia_ 

# plt.figure()
# plt.plot(K, distortions, 'bx-') 
# plt.xlabel('Values of K') 
# plt.ylabel('Distortion') 
# plt.title('The Elbow Method using Distortion') 
# plt.show() 

# These lines plot each cluster in white
K = 12
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
plt.suptitle('Image cluster for K=' + str(K))
for i in range(0,K,1):
    if i > 7:
        plt.subplot(2,2,i-7)
        plt.imshow(imageRGBThresh1[:,:,:,i])
        plt.title('Cluster ' + str(i+1))
        plt.xlabel('Horizontal Pixels')
        plt.ylabel('Vertical Pixels')
    elif i > 3:
        plt.subplot(2,2,i-3)
        plt.imshow(imageRGBThresh1[:,:,:,i])
        plt.title('Cluster ' + str(i+1))
        plt.xlabel('Horizontal Pixels')
        plt.ylabel('Vertical Pixels')
    else:
        plt.subplot(2,2,i+1)
        plt.imshow(imageRGBThresh1[:,:,:,i])
        plt.title('Cluster ' + str(i+1))
        plt.xlabel('Horizontal Pixels')
        plt.ylabel('Vertical Pixels')
    if i == 3 or i == 7:
        plt.figure()
        plt.suptitle('Image cluster for K=' + str(K))
# plt.show()
# cv2.imshow('Thresh',imageRGBThresh1) 
# cv2.waitKey(0)

Ksel = 3
kernel = np.zeros((5,5),np.uint8)
kernel[:] = np.asarray(([0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0,],[0,0,1,0,0]))
#These lines find and draw figure's contours
imageGrayRGB = cv2.cvtColor(imageRGBThresh1[:,:,:,Ksel],cv2.COLOR_RGB2GRAY)
ret, thresh = cv2.threshold(imageGrayRGB, 127, 255, 0)
# threshOpen = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel)
# cv2.imshow('Imagen Abierta 2',threshOpen)
threshClose = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel)
# cv2.imshow('Imagen Cerrada 2',threshClose)
contours, hierarchy = cv2.findContours(threshClose, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
imageRGBThreshColor = cv2.cvtColor(imageGrayRGB,cv2.COLOR_GRAY2RGB)
cv2.drawContours(imageRGBThreshColor, contours, -1, (0,255,0), 1)
# cv2.imshow("Contours", imageRGBThreshColor)
# for contour in contours:
#     print(cv2.contourArea(contour))
# cv2.waitKey(0)

plt.figure()
plt.suptitle('Cluster Image and Contours (Cluster=' + str(Ksel) + ')')
plt.subplot(1,2,1)
plt.imshow(threshClose, cmap='gray', vmin=0, vmax=255)
plt.title('Cluster Image Closed')
plt.xlabel('Horizontal Pixels')
plt.ylabel('Vertical Pixels')
plt.subplot(1,2,2)
plt.imshow(imageRGBThreshColor)
plt.title('Cluster Image with Contours')
plt.xlabel('Horizontal Pixels')
plt.ylabel('Vertical Pixels')
cv2.imwrite('Cluster.png',thresh)
plt.show()

# This implements K-means for BGR
features2D = imageBGR.reshape((-1, 3))
features2D = np.float32(features2D)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
attempts = 10
Kmax = 12
Kini = 3
result_image = np.zeros((height, width, 3,Kmax+1-Kini), dtype=np.uint8)
for K in range(Kini, Kmax+1, 1):
    ret, label, center = cv2.kmeans(
        features2D, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image[:, :, :, K-3] = res.reshape((imageRGB.shape))
    # cv2.imshow("K-means with K=%i" % K,result_image)

plt.figure()
plt.suptitle("Image Segmentation usign K-means (BGR)")
for i in range(0,Kmax-Kini+1,1):
    if i > 7:
        plt.subplot(2,2,i-7)
        plt.imshow(result_image[:, :, :, i])
        plt.title('Image Segmentation for K=' + str(i+Kini))
        plt.xlabel('Horizontal Pixels')
        plt.ylabel('Vertical Pixels')
    elif i > 3:
        plt.subplot(2,2,i-3)
        plt.imshow(result_image[:, :, :, i])
        plt.title('Image Segmentation for K=' + str(i+Kini))
        plt.xlabel('Horizontal Pixels')
        plt.ylabel('Vertical Pixels')
    else:
        plt.subplot(2,2,i+1)
        plt.imshow(result_image[:, :, :, i])
        plt.title('Image Segmentation for K=' + str(i+Kini))
        plt.xlabel('Horizontal Pixels')
        plt.ylabel('Vertical Pixels')
    if i == 3 or i == 7:
        plt.figure()
        plt.suptitle("Image Segmentation usign K-means (BGR)")
plt.show()

#This implements K-means for HS
features2D = imageHSV[:,:,:2].reshape((-1,2))
features2D = np.float32(features2D)

Kmax = 12
Kini = 3
result_image = np.zeros((height, width, 3,Kmax+1-Kini), dtype=np.uint8)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
attempts=10
for K in range(Kini, Kmax+1, 1):
    ret,label,center=cv2.kmeans(features2D,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image[:,:,:2,K-3] = res.reshape((height,width,2))
    result_image[:,:,:,K-3] = cv2.merge((result_image[:,:,0,K-3],result_image[:,:,1,K-3],imageHSV[:,:,2]))
    result_image[:,:,:,K-3] = cv2.cvtColor(result_image[:,:,:,K-3], cv2.COLOR_HSV2RGB)
    # cv2.imshow("K-means with K=%i" % K,cv2.cvtColor(result_image,cv2.COLOR_HSV2BGR))

plt.figure()
plt.suptitle("Image Segmentation usign K-means (HS)")
for i in range(0,Kmax-Kini+1,1):
    if i > 7:
        plt.subplot(2,2,i-7)
        plt.imshow(result_image[:, :, :, i])
        plt.title('Image Segmentation for K=' + str(i+Kini))
        plt.xlabel('Horizontal Pixels')
        plt.ylabel('Vertical Pixels')
    elif i > 3:
        plt.subplot(2,2,i-3)
        plt.imshow(result_image[:, :, :, i])
        plt.title('Image Segmentation for K=' + str(i+Kini))
        plt.xlabel('Horizontal Pixels')
        plt.ylabel('Vertical Pixels')
    else:
        plt.subplot(2,2,i+1)
        plt.imshow(result_image[:, :, :, i])
        plt.title('Image Segmentation for K=' + str(i+Kini))
        plt.xlabel('Horizontal Pixels')
        plt.ylabel('Vertical Pixels')
    if i == 3 or i == 7:
        plt.figure()
        plt.suptitle("Image Segmentation usign K-means (HS)")
plt.show()

# These lines implement Elbow's method
# distortions = [] 
# inertias = [] 
# mapping1 = {} 
# mapping2 = {} 
# K = range(1,10) 
  
# for k in K: 
#     #Building and fitting the model 
#     kmeanModel = KMeans(n_clusters=k).fit(features2D) 
#     kmeanModel.fit(features2D)     
      
#     distortions.append(sum(np.min(cdist(features2D, kmeanModel.cluster_centers_, 
#                       'euclidean'),axis=1)) / features2D.shape[0]) 
#     inertias.append(kmeanModel.inertia_) 
  
#     mapping1[k] = sum(np.min(cdist(features2D, kmeanModel.cluster_centers_, 
#                  'euclidean'),axis=1)) /features2D.shape[0] 
#     mapping2[k] = kmeanModel.inertia_ 

# plt.figure()
# plt.plot(K, distortions, 'bx-') 
# plt.xlabel('Values of K') 
# plt.ylabel('Distortion') 
# plt.title('The Elbow Method using Distortion') 
# plt.show() 

# These lines plot each cluster in white
K = 12
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
plt.suptitle('Image cluster for K=' + str(K))
for i in range(0,K,1):
    if i > 7:
        plt.subplot(2,2,i-7)
        plt.imshow(imageRGBThresh1[:,:,:,i])
        plt.title('Cluster ' + str(i+1))
        plt.xlabel('Horizontal Pixels')
        plt.ylabel('Vertical Pixels')
    elif i > 3:
        plt.subplot(2,2,i-3)
        plt.imshow(imageRGBThresh1[:,:,:,i])
        plt.title('Cluster ' + str(i+1))
        plt.xlabel('Horizontal Pixels')
        plt.ylabel('Vertical Pixels')
    else:
        plt.subplot(2,2,i+1)
        plt.imshow(imageRGBThresh1[:,:,:,i])
        plt.title('Cluster ' + str(i+1))
        plt.xlabel('Horizontal Pixels')
        plt.ylabel('Vertical Pixels')
    if i == 3 or i == 7:
        plt.figure()
        plt.suptitle('Image cluster for K=' + str(K))
plt.show()


#This implements K-means with LBP
features2D = imageLBP.reshape((-1,1))
features2D = np.float32(features2D)

Kmax = 12
Kini = 3
result_image = np.zeros((height, width, 3, Kmax+1-Kini), dtype=np.uint8)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
attempts=10
for K in range(Kini, Kmax+1, 1):
    ret,label,center=cv2.kmeans(features2D,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    for i in range(0, height, 1):
        for j in range(0, width, 1):
            result_image[i, j, :, K-3] = colors[label[(i)*width + j, 0]]
    # cv2.imshow("K-means with K=%i" % K,cv2.cvtColor(result_image,cv2.COLOR_HSV2BGR))

plt.figure()
plt.suptitle("Image Segmentation usign K-means (LBP)")
for i in range(0,Kmax-Kini+1,1):
    if i > 7:
        plt.subplot(2,2,i-7)
        plt.imshow(result_image[:, :, :, i])
        plt.title('Image Segmentation for K=' + str(i+Kini))
        plt.xlabel('Horizontal Pixels')
        plt.ylabel('Vertical Pixels')
    elif i > 3:
        plt.subplot(2,2,i-3)
        plt.imshow(result_image[:, :, :, i])
        plt.title('Image Segmentation for K=' + str(i+Kini))
        plt.xlabel('Horizontal Pixels')
        plt.ylabel('Vertical Pixels')
    else:
        plt.subplot(2,2,i+1)
        plt.imshow(result_image[:, :, :, i])
        plt.title('Image Segmentation for K=' + str(i+Kini))
        plt.xlabel('Horizontal Pixels')
        plt.ylabel('Vertical Pixels')
    if i == 3 or i == 7:
        plt.figure()
        plt.suptitle("Image Segmentation usign K-means (LBP)")
plt.show()

# These lines implement Elbow's method
# distortions = [] 
# inertias = [] 
# mapping1 = {} 
# mapping2 = {} 
# K = range(1,10) 
  
# for k in K: 
#     #Building and fitting the model 
#     kmeanModel = KMeans(n_clusters=k).fit(features2D) 
#     kmeanModel.fit(features2D)     
      
#     distortions.append(sum(np.min(cdist(features2D, kmeanModel.cluster_centers_, 
#                       'euclidean'),axis=1)) / features2D.shape[0]) 
#     inertias.append(kmeanModel.inertia_) 
  
#     mapping1[k] = sum(np.min(cdist(features2D, kmeanModel.cluster_centers_, 
#                  'euclidean'),axis=1)) /features2D.shape[0] 
#     mapping2[k] = kmeanModel.inertia_ 

# plt.figure()
# plt.plot(K, distortions, 'bx-') 
# plt.xlabel('Values of K') 
# plt.ylabel('Distortion') 
# plt.title('The Elbow Method using Distortion') 
# plt.show() 

# These lines plot each cluster in white
K = 12
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
plt.suptitle('Image cluster for K=' + str(K))
for i in range(0,K,1):
    if i > 7:
        plt.subplot(2,2,i-7)
        plt.imshow(imageRGBThresh1[:,:,:,i])
        plt.title('Cluster ' + str(i+1))
        plt.xlabel('Horizontal Pixels')
        plt.ylabel('Vertical Pixels')
    elif i > 3:
        plt.subplot(2,2,i-3)
        plt.imshow(imageRGBThresh1[:,:,:,i])
        plt.title('Cluster ' + str(i+1))
        plt.xlabel('Horizontal Pixels')
        plt.ylabel('Vertical Pixels')
    else:
        plt.subplot(2,2,i+1)
        plt.imshow(imageRGBThresh1[:,:,:,i])
        plt.title('Cluster ' + str(i+1))
        plt.xlabel('Horizontal Pixels')
        plt.ylabel('Vertical Pixels')
    if i == 3 or i == 7:
        plt.figure()
        plt.suptitle('Image cluster for K=' + str(K))
plt.show()
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import numpy as np
from sklearn.cluster import KMeans 
from sklearn import metrics 
from scipy.spatial.distance import cdist 
from PIL import Image

img2 = cv2.imread("../FotoCoral1.jpg")
img = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
imgGray = cv2.cvtColor(img2, cv2.cv2.COLOR_BGR2GRAY)
#img = img2

# r, g, b = cv2.split(img)
# r = r.flatten()
# g = g.flatten()
# b = b.flatten()

# h, s, v = cv2.split(img)
# h = h.flatten()
# s = s.flatten()
# v = v.flatten()

#plotting 
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(r, g, b)
# plt.show()

# vectorized = img[:,:,:2].reshape((-1,2))
# vectorized = np.float32(vectorized)

# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
# K = 7
# attempts=10
# for K in range(1,11,1):
#     ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
#     center = np.uint8(center)
#     res = center[label.flatten()]
#     result_image = res.reshape((480,640,2))
#     result_image = cv2.merge((result_image[:,:,0],result_image[:,:,1],img[:,:,2]))
#     cv2.imshow("K-means with K=%i" % K,cv2.cvtColor(result_image,cv2.COLOR_HSV2BGR))

# figure_size = 15
# plt.figure(figsize=(figure_size,figure_size))
# plt.subplot(1,2,1),plt.imshow(cv2.cvtColor(img2,cv2.COLOR_BGR2RGB))
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(1,2,2),plt.imshow(cv2.cvtColor(result_image,cv2.COLOR_HSV2RGB))
# plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])
# plt.show()


# distortions = [] 
# inertias = [] 
# mapping1 = {} 
# mapping2 = {} 
# K = range(1,10) 
  
# for k in K: 
#     #Building and fitting the model 
#     kmeanModel = KMeans(n_clusters=k).fit(vectorized) 
#     kmeanModel.fit(vectorized)     
      
#     distortions.append(sum(np.min(cdist(vectorized, kmeanModel.cluster_centers_, 
#                       'euclidean'),axis=1)) / vectorized.shape[0]) 
#     inertias.append(kmeanModel.inertia_) 
  
#     mapping1[k] = sum(np.min(cdist(vectorized, kmeanModel.cluster_centers_, 
#                  'euclidean'),axis=1)) /vectorized.shape[0] 
#     mapping2[k] = kmeanModel.inertia_ 

# plt.figure()
# plt.plot(K, distortions, 'bx-') 
# plt.xlabel('Values of K') 
# plt.ylabel('Distortion') 
# plt.title('The Elbow Method using Distortion') 
# plt.show() 


height, width = imgGray.shape
imageLBP = np.zeros((height,width),dtype = np.uint8)
for i in range(0,height,1):
    for j in range(0,width,1):
        pot = 1
        for x in range(i-1,i+2):
            for y in range(j-1,j+2,1):
                if x < 0 or y < 0 or x >= height or y >= width:
                    pot *= 2
                    continue
                elif x == i and y == j:
                    continue
                elif imgGray[x,y] >= imgGray[i,j]:
                    imageLBP[i,j] += pot
                    pot *= 2
                else:
                    pot *= 2
imageLBP2 = Image.fromarray(imageLBP, 'L')
# cv2.imshow("LBP Image", imageLBP)
# cv2.waitKey(0)
imageLBP2.show()

vectorized3 = img
vectorized3[:,:,2] = imageLBP
vectorized32 = np.zeros((height*width,3))
for i in range(0,height,1):
    for j in range(0,width,1):
        vectorized32[(i)*width + j,0] = vectorized3[i,j,0]
        vectorized32[(i)*width + j,1] = vectorized3[i,j,1]
        vectorized32[(i)*width + j,2] = vectorized3[i,j,2]
vectorized32 = np.float32(vectorized32)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 7
attempts=10
for K in range(3,6,1):
    ret,label,center=cv2.kmeans(vectorized32,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    # center = np.uint8(center)
    # res = center[label.flatten()]
    # result_image = res.reshape((height,width,3))
    label.reshape(-1,2)
    Hdiv = 255//K
    result_image = img
    print(label)
    for i in range(0,height,1):
        for j in range(0,width,1):
            result_image[i,j,0] = Hdiv*(label[(i)*width + j]+1)
    result_image[:,:,1:2] = 128
    # result_image = cv2.merge((result_image[:,:,0],result_image[:,:,1],img[:,:,2]))
    cv2.imshow("K-means with K=%i" % K,cv2.cvtColor(result_image,cv2.COLOR_HSV2BGR))

cv2.waitKey(0)
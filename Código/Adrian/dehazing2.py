import cv2
import numpy as np 
import matplotlib.pyplot as plt
import math
from PIL import Image
from cv2.ximgproc import createGuidedFilter
from cv2.ximgproc import guidedFilter
# from guided_filter_pytorch.guided_filter import GuidedFilter
#from guided_filter_tf.guided_filter import guided_filter

src = cv2.imread("ImagenProyecto.jpg")   #Reading image
b, g, r = cv2.split(src)    #Spliting RGB channels
b = np.array(b,dtype = int)
g = np.array(g,dtype = int)
r = np.array(r,dtype = int)

# Bright Channel Prior
bNew = [255 - x for x in b]     #Complement of b and g channels
gNew = [255 - x for x in g]
rNew = r
bNew = np.array(bNew,dtype = int)   #Converting lists to np array
gNew = np.array(gNew,dtype = int)

# #Step 1: Bright channel image
height, width, depth = src.shape 
# patchSize = math.sqrt(height*width)//50     #Maximum-filter size
# patchHalfSize = int(patchSize/2)

# Ibright = np.zeros((height,width))

# for i in range(0,height):
#     for j in range(0,width):
#         for k in range(i - patchHalfSize, i + patchHalfSize):
#             for l in range(j - patchHalfSize, j + patchHalfSize):
#                 if k < 0 or l < 0 or k >= height or l >= width:
#                     continue
#                 else:
#                     Ibright[i,j] = max(max(bNew[k,l], gNew[k,l], rNew[k,l]),Ibright[i,j])

# cv2.imwrite('BrightChannel.png', Ibright)
# cv2.imshow('BrightChannel', Ibright)
# cv2.waitKey(0)

# # Step 2: Maximum color difference image

# MCD = np.zeros((height,width))
# for i in range(0,height):
#     for j in range(0,width):
#         Cmax = max(b[i,j], g[i,j], r[i,j])
#         Cmin = min(b[i,j], g[i,j], r[i,j])

#         if Cmax == b[i,j]:
#             if Cmin == g[i,j]:
#                 Cmid = r[i,j]
#             elif Cmin == r[i,j]:
#                 Cmid = g[i,j]
#         elif Cmax == g[i,j]:
#             if Cmin == b[i,j]:
#                 Cmid = r[i,j]
#             elif Cmin == r[i,j]:
#                 Cmid = b[i,j]
#         elif Cmax == r[i,j]:
#             if Cmin == b[i,j]:
#                 Cmid = g[i,j]
#             elif Cmin == g[i,j]:
#                 Cmid = b[i,j]

#         MCD[i,j] = 255 - max(max(Cmax - Cmin, 0), max(Cmid - Cmin,0))

# MCD = np.array(MCD, dtype = np.uint8)
# cv2.imwrite('MCD.png', MCD)
# cv2.imshow('MCD', MCD)
# cv2.waitKey(0)

# # Step 3: Bright channel rectification

# MCD = cv2.imread('MCD.png',0)
# MCD = np.array(MCD, dtype = np.uint8)
# Ibright = cv2.imread('BrightChannel.png',0)
# Ibright = np.array(Ibright, dtype = np.uint8)

# srcHSV = cv2.cvtColor(src,cv2.COLOR_BGR2HSV)
# h, s, v = cv2.split(srcHSV)

# hsvCoef = 0
# for i in range(0,height):
#     for j in range(0,width):
#         hsvCoef = max(s[i,j],hsvCoef)

# hsvCoef = float(hsvCoef)/255

# Ibright2 = np.zeros((height,width))

# for i in range(0,height):
#     for j in range(0,width):
#         Ibright2[i,j] = int(hsvCoef*float(Ibright[i,j]) + (1-hsvCoef)*float(MCD[i,j]))

# Ibright2 = np.array(Ibright2, dtype = np.uint8)
# cv2.imwrite('Ibright2.png', Ibright2)
# cv2.imshow('Ibright2',Ibright2)
# cv2.waitKey(0)

# Step 4: Atmospheric ligth estimation

# Variance Image Calculation

srcGray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
srcGray = np.array(srcGray,dtype = int)
patchSize = math.sqrt(height*width)//50 
patchHalfSize = int(patchSize/2)

varIm = np.zeros((height,width))

for i in range(0,height):
    for j in range(0,width):
        patch = srcGray[max(i-patchHalfSize,0):min(i+patchHalfSize,height),max(j-patchHalfSize,0):min(j+patchHalfSize,width)]
        varIm[i,j] = int(patch.var())

varIm = np.array(varIm, dtype = np.uint8)
cv2.imwrite('varIm.png', varIm)
cv2.imshow('varIm', varIm)
cv2.waitKey(0)

# Finding 1% darker pixels of brigth channel image
Ibright = cv2.imread('BrightChannel.png',0)
Ibright = np.array(Ibright, dtype = np.uint8)
histIbright = cv2.calcHist(Ibright,[0],None,[256],[0,256])
histIbright = np.array(histIbright,dtype = int)

acc = 0
for i in range(0,len(histIbright)):
    acc += histIbright[i]

onePercent = int(acc/100)
histIbrightOnePercent = np.zeros(len(histIbright))

acc = 0
for i in range(0,255):
    acc += histIbright[i]
    histIbrightOnePercent[i] = histIbright[i]
    if acc >= onePercent:
        histIbrightOnePercent[i] = 0


nonZero = [idx for idx, val in enumerate(histIbrightOnePercent) if val != 0]    # Determine the indices in list with non-zero value
IbrightOnePercent = np.zeros((height,width))

# Calculate the image resulting from obtaining 1% of darker pixels in bright channel image
minVar = 255
for i in range(0,height):
    for j in range(0,width):
        for k in nonZero:
            if Ibright[i,j] == k:
                IbrightOnePercent[i,j] = Ibright[i,j]
                if varIm[i,j] < minVar:
                    minVar = varIm[i,j]
                    atmLightpx = [i,j]        # Pixel with lowest variance. Thus, Atmospheric Light coordinates

IbrightOnePercent = np.array(IbrightOnePercent, dtype = np.uint8)
cv2.imwrite('IbrightOnePercent.png', IbrightOnePercent)
# Atmospherin Light 
aB = b[atmLightpx[0],atmLightpx[1]]
aG = g[atmLightpx[0],atmLightpx[1]]
aR = r[atmLightpx[0],atmLightpx[1]]
print(aB,aG,aR)

plt.figure()
plt.subplot(2,2,1)
cv2.imshow('Ibright', Ibright)
plt.subplot(2,2,2)
cv2.imshow('IbrightOnePercent', IbrightOnePercent)
plt.subplot(2,2,3)
plt.plot(histIbright)
plt.subplot(2,2,4)
plt.plot(histIbrightOnePercent)
plt.show()

# Step 5: Transmittance image calculation and refinement

Ibright2 = cv2.imread('Ibright2.png',0)
t = np.zeros((height,width))

for i in range(0,height):
    for j in range(0,width):
        for k in [aB, aG, aR]:
            t[i,j] += 255*(Ibright2[i,j] - k) / (255 - k)
        t[i,j] = int(t[i,j] / 3)

t = np.array(t, dtype = np.uint8)
cv2.imwrite('Transmittance.png', t)
cv2.imshow('Transmittance',t)
cv2.waitKey(0)

# Apply filter
tFiltered = np.zeros((height,width))
srcGray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
t = cv2.imread('Transmittance.png', 0)

tFiltered = guidedFilter(srcGray, t, 500,200)
tFiltered = np.array(tFiltered, dtype = np.uint8)
cv2.imwrite('TransmittanceFiltered.png', tFiltered)
cv2.imshow('TransmittanceFiltered',tFiltered)
cv2.waitKey(0)

# # Step 6: Image restoration

jB = np.zeros((height,width))
jG = np.zeros((height,width))
jR = np.zeros((height,width))
for i in range(0,height):
    for j in range(0,width):
        jR[i,j] = (rNew[i,j] - aR) / tFiltered[i,j] + aR
        jG[i,j] = 255 - int((gNew[i,j] - (255 - aG)*(255 - tFiltered[i,j])) / tFiltered[i,j])
        jB[i,j] = 255 - int((bNew[i,j] - (255 - aB)*(255 - tFiltered[i,j])) / tFiltered[i,j])

jB = np.array(jB, dtype = np.uint8)
jG = np.array(jG, dtype = np.uint8)
jR = np.array(jR, dtype = np.uint8)

# Final image

result = cv2.merge((jB,jG,jR))

cv2.imwrite('Result.png', result)
cv2.imshow('Result', result)
cv2.waitKey(0)
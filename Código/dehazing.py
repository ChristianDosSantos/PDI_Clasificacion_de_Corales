import cv2
import numpy as np 
import matplotlib.pyplot as plt

# def convolve(image, kernel):
# 	# grab the spatial dimensions of the image, along with
# 	# the spatial dimensions of the kernel
# 	(iH, iW) = image.shape[:2]
# 	(kH, kW) = kernel.shape[:2]
 
# 	# allocate memory for the output image, taking care to
# 	# "pad" the borders of the input image so the spatial
# 	# size (i.e., width and height) are not reduced
# 	pad = (kW - 1) // 2
# 	image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
# 		cv2.BORDER_REPLICATE)
# 	output = np.zeros((iH, iW), dtype="float32")
#     	# loop over the input image, "sliding" the kernel across
# 	# each (x, y)-coordinate from left-to-right and top to
# 	# bottom
# 	for y in np.arange(pad, iH + pad):
# 		for x in np.arange(pad, iW + pad):
# 			# extract the ROI of the image by extracting the
# 			# *center* region of the current (x, y)-coordinates
# 			# dimensions
# 			roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
 
# 			# perform the actual convolution by taking the
# 			# element-wise multiplicate between the ROI and
# 			# the kernel, then summing the matrix
# 			k = (roi * kernel).sum()
 
#             # store the convolved value in the output (x,y)-
#             # coordinate of the output image
#             output[y - pad, x - pad] = k
#     # rescale the output image to be in the range [0, 255]
#     output = rescale_intensity(output, in_range=(0, 255))
#     output = (output * 255).astype("uint8")

#     # return the output image
#     return output

src = cv2.imread("../FotoCoral1.jpg")

src = cv2.cvtColor(src,cv2.COLOR_BGR2RGB)
r, g, b = cv2.split(src)
b = np.array(b,dtype = int)
g = np.array(g,dtype = int)
r = np.array(r,dtype = int)
height, width, depth = src.shape

srcHsv = cv2.cvtColor(src,cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(srcHsv)
coefHsv = 0

Cmin = 0
Cmax = 0
Cmed = 0
Imax = 0
Ibright = np.zeros((height,width))
MCD = np.zeros((height,width))
Ibright2 = np.zeros((height,width))

for i in range(0,height,1):
    for j in range(0,width,1):
        # if r[i,j] >= g[i,j]:
        #     Cmax = r[i,j]
        #     Cmin = g[i,j]
        # else:
        #     Cmax = g[i,j]
        #     Cmin = r[i,j]
        # if b[i,j] >= Cmax:
        #     Cmed = Cmax
        #     Cmax = b[i,j]
        # else:
        #     if b[i,j] <= Cmin:
        #         Cmed = Cmin
        #         Cmin = b[i,j]
        Cmax = max(r[i,j],b[i,j],g[i,j])
        Cmin = min(r[i,j],b[i,j],g[i,j])
        Cmed = r[i,j] + g[i,j] + b[i,j] - Cmax - Cmin
        MCD[i,j] = 1 - max(max(Cmax - Cmin, 0),max(Cmed - Cmin, 0))
        for  y in range(i-1,i+2,1):
            for x in range (j-1,j+2,1):
                if x < 0 or y < 0 or x >= width or y >= height:
                    continue
                else:
                    Ibright[i,j] = max(max(b[y,x], g[y,x], r[y,x]),Ibright[i,j])
        coefHsv = max(srcHsv[i,j,1],coefHsv)


MCD = np.array(MCD,dtype = np.uint8)
#print (MCD)
#print(Ibright)
Imagen_dif_color = np.zeros((height,width,1))
Imagen_dif_color[:,:,0] = MCD
# Ibright = np.array(Ibright,dtype = np.uint8)
# cv2.imshow("Imagen Final", Ibright)
# cv2.waitKey(0)

for i in range(0,height,1):
    for j in range(0,width,1):
        Ibright2[i,j] = coefHsv*Ibright[i,j] + (1-coefHsv)*MCD[i,j]
        if Ibright2[i,j] > 255:
            Ibright2[i,j] = 255 

Ibright2 = np.array(Ibright2,dtype = np.uint8)
cv2.imshow("Imagen Final 2", Ibright2)

kernel = [[-1,-1,-1],[1,0,-1],[1,1,1]]
kernel = np.asarray(kernel)
Image_variance = cv2.filter2D(src,-1,kernel)
cv2.imshow('Color r',Image_variance[:,:,0])
cv2.imshow('Color g',Image_variance[:,:,1])
cv2.imshow('Color b',Image_variance[:,:,2])
cv2.waitKey(0)

Imax = 0

for i in range(0,height,1):
    for j in range(0,width,1):
        if Ibright2[i,j] > Imax:
            Imax = Ibright2[i,j]

I1per = Imax//100
PixVarMaxr = Image_variance[0,0,0]
PixVarMaxg = Image_variance[0,0,1]
PixVarMaxb = Image_variance[0,0,2]
AtmR = 0
AtmG = 0
AtmB = 0


for i in range(0,height,1):
    for j in range(0,width,1):
        if Ibright2[i,j] <= I1per:
            if Image_variance[i,j,0] < PixVarMaxr:
                PixVarMaxr = Image_variance[i,j,0]
                AtmR = src[i,j,0]
            if Image_variance[i,j,1] < PixVarMaxg:
                PixVarMaxg = Image_variance[i,j,1]
                AtmG = src[i,j,1]
            if Image_variance[i,j,2] < PixVarMaxb:
                PixVarMaxb = Image_variance[i,j,2]
                AtmB = src[i,j,2]

transR = np.zeros((height,width))
transG = np.zeros((height,width))
transB = np.zeros((height,width))

Ibright2 = np.array(Ibright2,dtype = int) 
AtmR = np.array(AtmR,dtype = int) 
AtmG = np.array(AtmG,dtype = int) 
AtmB = np.array(AtmB,dtype = int) 


for i in range(0,height,1):
    for j in range(0,width,1):
        transR[i,j] = (Ibright2[i,j] - AtmR)//(1 - AtmR)
        transG[i,j] = (Ibright2[i,j] - AtmG)//(1 - AtmG)
        transB[i,j] = (Ibright2[i,j] - AtmB)//(1 - AtmB)
        if transR[i,j] > 255:
            transR[i,j] = 255
        if transG[i,j] > 255:
            transG[i,j] = 255
        if transB[i,j] > 255:
            transB[i,j] = 255

# AtmR = np.array(AtmR,dtype = np.uint8) 
# AtmG = np.array(AtmG,dtype = np.uint8) 
# AtmB = np.array(AtmB,dtype = np.uint8) 

# transR = np.array(transR,dtype = np.uint8) 
# transG = np.array(transG,dtype = np.uint8) 
# transB = np.array(transB,dtype = np.uint8) 

Image_trans = cv2.merge((transR,transG,transB))

cv2.imshow("Transmitance Image R",transR)
cv2.imshow("Transmitance Image G",transG)
cv2.imshow("Transmitance Image B",transB)
cv2.waitKey(0)

Image_dehaze = np.zeros((height,width,3))

for i in range(0,height,1):
    for j in range(0,width,1):
        Image_dehaze[i,j,0] = (r[i,j] - AtmR)//transR[i,j] + AtmR
        Image_dehaze[i,j,1] = (g[i,j] - AtmG)//transG[i,j] + AtmG
        Image_dehaze[i,j,2] = (b[i,j] - AtmB)//transB[i,j] + AtmB
        if Image_dehaze[i,j,0] > 255:
            Image_dehaze[i,j,0] = 255
        if Image_dehaze[i,j,1] > 255:
            Image_dehaze[i,j,1] = 255
        if Image_dehaze[i,j,2] > 255:
            Image_dehaze[i,j,2] = 255

Image_dehaze = np.array(Image_dehaze,dtype = np.uint8) 
cv2.imshow("Image dehaze",Image_dehaze)
cv2.waitKey(0)








            
       

        


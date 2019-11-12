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
print (MCD)
print(Ibright)
Imagen_dif_color = np.zeros((height,width,1))
Imagen_dif_color[:,:,0] = MCD
# cv2.imshow("Imagen Final", Ibright)
# cv2.waitKey(0)

for i in range(0,height,1):
    for j in range(0,width,1):
        Ibright2[i,j] = coefHsv*Ibright[i,j] + (1-coefHsv)*MCD[i,j]

kernel = [[-1,-1,-1],[1,0,-1],[1,1,1]]
Image_variance = cv2.filter2D(src,-1,kernel)
cv2.imshow('Color r',Image_variance[:,:,0])
cv2.imshow('Color g',Image_variance[:,:,1])
cv2.imshow('Color b',Image_variance[:,:,2])
cv2.waitKey(0)



            
       

        


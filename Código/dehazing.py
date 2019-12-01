import cv2
import numpy as np 
import matplotlib.pyplot as plt
import math
from PIL import Image
from cv2.ximgproc import guidedFilter
from guided_filter_tf.guided_filter import guided_filter

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

# src = cv2.imread("../fotoEjemplo2.png")
src = cv2.imread("../ImagenProyecto.jpg")

src = cv2.cvtColor(src,cv2.COLOR_BGR2RGB)
r, g, b = cv2.split(src)
b = np.array(b,dtype = int)
g = np.array(g,dtype = int)
r = np.array(r,dtype = int)
srcGray = cv2.cvtColor(src,cv2.COLOR_RGB2GRAY)
height, width, depth = src.shape

srcHsv = cv2.cvtColor(src,cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(srcHsv)
coefHsv = 0.0
patchSize = math.sqrt(height*width)//50

Cmin = 0
Cmax = 0
Cmed = 0
Imax = 0
Ibright = np.zeros((height,width))
MCD = np.zeros((height,width))
Ibright2 = np.zeros((height,width))

histR = cv2.calcHist(src,[0],None,[256],[0,256])
histG = cv2.calcHist(src,[1],None,[256],[0,256])
histB = cv2.calcHist(src,[2],None,[256],[0,256])
plt.figure()
plt.suptitle("Imagen Original y sus histogramas")
plt.subplot(2,2,1)
plt.imshow(src)
plt.title("Imagen Original")
plt.xlabel('Píxeles Horizontales')
plt.ylabel("Píxeles Verticales")
plt.subplot(2,2,2)
plt.plot(histR,"r")
plt.title("Histograma Rojo")
plt.ylabel('Número de Píxeles')
plt.xlabel("Intensidad")
plt.subplot(2,2,3)
plt.plot(histG,"g")
plt.title("Histograma Verde")
plt.ylabel('Número de Píxeles')
plt.xlabel("Intensidad")
plt.subplot(2,2,4)
plt.plot(histB,"b")
plt.title("Histograma Azul")
plt.ylabel('Número de Píxeles')
plt.xlabel("Intensidad")
plt.show()
Racc = 0
Gacc = 0
Bacc = 0
for i in range(0,256,1):
    Racc += int((i+1)*histR[i])
    Gacc += int((i+1)*histG[i])
    Bacc += int((i+1)*histB[i])
if Racc >= Gacc:
    Canalmax = 0
    Canalmin = 1
    if Bacc >= Racc:
        Canalmed = Canalmax
        Canalmax = 2
    else:
        if Bacc <= Gacc:
            Canalmed = Canalmin
            Canalmin = 2
        else:
            Canalmed = 2
else:
    Canalmax = 1
    Canalmin = 0
    if Bacc >= Gacc:
        Canalmed = Canalmax
        Canalmax = 2
    else:
        if Bacc <= Racc:
            Canalmed = Canalmin
            Canalmin = 2
        else:
            Canalmed = 2

print(Racc)
print(Gacc)
print(Bacc)
print(Canalmax)
print(Canalmed)
print(Canalmin)
# patchCalc = int(math.sqrt(patchSize)/2)
patchCalc = int(patchSize/2)

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
        # Cmax = max(r[i,j],b[i,j],g[i,j])
        # Cmin = min(r[i,j],b[i,j],g[i,j])
        # Cmed = r[i,j] + g[i,j] + b[i,j] - Cmax - Cmin
        Cmax = int(src[i,j,Canalmax])
        Cmed = int(src[i,j,Canalmed])
        Cmin = int(src[i,j,Canalmin])
        # print(255 - max(max(Cmax - Cmin, 0),max(Cmed - Cmin, 0)))
        MCD[i,j] = 255 - max(max(Cmax - Cmin, 0),max(Cmed - Cmin, 0))
        # for  y in range(i-1,i+2,1):
        #     for x in range (j-1,j+2,1):
        for  y in range(i-patchCalc,i+ patchCalc + 1):
            for x in range (j-patchCalc,j+patchCalc+1):
                if x < 0 or y < 0 or x >= width or y >= height:
                    continue
                else:
                    Ibright[i,j] = max(max(b[y,x], g[y,x], r[y,x]),Ibright[i,j])
        # print(srcHsv[i,j,1])
        coefHsv = max(float(float(srcHsv[i,j,1])/255),coefHsv)
        # print(coefHsv)
        
print(coefHsv)
# coefHsv = 0.7
# cv2.waitKey(0)

MCD = np.array(MCD,dtype = np.uint8)
# cv2.imshow("Maximun Color Image", MCD)
#print (MCD)
#print(Ibright)
Imagen_dif_color = np.zeros((height,width,1))
Imagen_dif_color[:,:,0] = MCD
Ibright8 = np.array(Ibright,dtype = np.uint8)
# cv2.imshow("Imagen Canal Brillante", Ibright8)
# cv2.waitKey(0)

for i in range(0,height,1):
    for j in range(0,width,1):
        Ibright2[i,j] = int(coefHsv*float(Ibright[i,j]) + (1-coefHsv)*float(MCD[i,j]))
        if Ibright2[i,j] > 255:
            Ibright2[i,j] = 255 

Ibright28 = np.array(Ibright2,dtype = np.uint8)
# cv2.imshow("Imagen Canal Brillante 2", Ibright28)

plt.figure()
plt.suptitle('Imagen Original y Procesadas')
plt.subplot(2,2,1)
plt.imshow(src)
plt.title('Imagen Original')
plt.xlabel('Píxeles Horizontales')
plt.ylabel("Píxeles Verticales")
plt.subplot(2,2,2)
plt.imshow(Ibright8,cmap='gray', vmin=0, vmax=255)
plt.title('Imagen del Canal Brillante')
plt.xlabel('Píxeles Horizontales')
plt.ylabel("Píxeles Verticales")
plt.subplot(2,2,3)
plt.imshow(Ibright2,cmap='gray', vmin=0, vmax=255)
plt.title('Imagen del Canal Brillante Rectificada')
plt.xlabel('Píxeles Horizontales')
plt.ylabel("Píxeles Verticales")
plt.subplot(2,2,4)
plt.imshow(MCD,cmap='gray', vmin=0, vmax=255)
plt.title('Imagen de Máxima Diferencia de Color')
plt.xlabel('Píxeles Horizontales')
plt.ylabel("Píxeles Verticales")


kernel = [[-1,-1,-1],[1,0,-1],[1,1,1]]
kernel = np.asarray(kernel)
Image_variance = cv2.filter2D(src,-1,kernel)
# cv2.imshow('Color r',Image_variance[:,:,0])
# cv2.imshow('Color g',Image_variance[:,:,1])
# cv2.imshow('Color b',Image_variance[:,:,2])
# cv2.waitKey(0)
plt.figure()
plt.suptitle("Imágenes de Varianza")
plt.subplot(2,2,1)
plt.imshow(Image_variance[:,:,0],cmap='gray', vmin=0, vmax=255)
plt.title("Imagen de Varianza del Canal Rojo")
plt.xlabel('Píxeles Horizontales')
plt.ylabel("Píxeles Verticales")
plt.subplot(2,2,2)
plt.imshow(Image_variance[:,:,1],cmap='gray', vmin=0, vmax=255)
plt.title("Imagen de Varianza del Canal Verde")
plt.xlabel('Píxeles Horizontales')
plt.ylabel("Píxeles Verticales")
plt.subplot(2,2,3)
plt.imshow(Image_variance[:,:,2],cmap='gray', vmin=0, vmax=255)
plt.title("Imagen de Varianza del Canal Azul")
plt.xlabel('Píxeles Horizontales')
plt.ylabel("Píxeles Verticales")

# Imax = 0

# for i in range(0,height,1):
#     for j in range(0,width,1):
#         if Ibright2[i,j] > Imax:
#             Imax = Ibright2[i,j]

hist = cv2.calcHist([Ibright28[:,:]],[0],None,[256],[0,256])
# plt.figure()
# plt.plot(hist)
# plt.show()

NumPixelPer = height*width//100
PixAcc = 0
I1per = 0

for i in range(0,256,1):
    PixAcc += hist[i]
    if PixAcc >= NumPixelPer:
        I1per = i
        break

# I1per = Imax//100
PixVarMaxr = 256
PixVarMaxg = 256
PixVarMaxb = 256
AtmR = 0
AtmG = 0
AtmB = 0
print(I1per)

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

transR = np.zeros((height,width),dtype=np.int)
transG = np.zeros((height,width),dtype=np.int)
transB = np.zeros((height,width),dtype=np.int)
 
print(AtmR)
print(AtmG)
print(AtmB)

AtmR = np.array(AtmR,dtype = int) 
AtmG = np.array(AtmG,dtype = int) 
AtmB = np.array(AtmB,dtype = int) 
AtmP = (AtmR + AtmG + AtmB)//3

for i in range(0,height,1):
    for j in range(0,width,1):
        # print(Ibright2[i,j])
        # print(int((255*(float(Ibright2[i,j]) - float(AtmR)))/(255 - float(AtmR))))
        # print((255*(float(Ibright2[i,j]) - float(AtmR)))/(255 - float(AtmR)))
        transR[i,j] = int((255*(max(float(Ibright2[i,j]) - float(AtmR),0)))/(255 - float(AtmR)))
        transG[i,j] = int((255*(max(float(Ibright2[i,j]) - float(AtmG),0)))/(255 - float(AtmG)))
        transB[i,j] = int((255*(max(float(Ibright2[i,j]) - float(AtmB),0)))/(255 - float(AtmB)))
        # print(transB[i,j])
        if transR[i,j] > 255:
            transR[i,j] = 255
        if transG[i,j] > 255:
            transG[i,j] = 255
        if transB[i,j] > 255:
            transB[i,j] = 255
        if transB[i,j] < 0:
            print(transB[i,j])

# AtmR = np.array(AtmR,dtype = np.uint8) 
# AtmG = np.array(AtmG,dtype = np.uint8) 
# AtmB = np.array(AtmB,dtype = np.uint8) 

transR8 = np.array(transR,dtype = np.uint8) 
transG8 = np.array(transG,dtype = np.uint8) 
transB8 = np.array(transB,dtype = np.uint8) 
# guided = guidedFilter(transB8,13,70,0.01) 
# cv2.imshow('Guided Filter',guided)


TransP = np.zeros((height,width),dtype=int)
for i in range(0,height):
    for j in range(0,width):
        TransP[i,j]=int((transR[i,j]+transB[i,j]+transG[i,j])//3)

# guidedTrans = guided_filter(srcGray, TransP, 1, 0.01, 1)
# guidedTrans8 = np.array(guidedTrans,dtype=np.uint8)
transP8 = np.array(TransP,dtype=np.uint8)

# cv2.imwrite('ProyectoTransmitancia.png',TransP8)
# cv2.imwrite('ProyectoGray.png',srcGray)

plt.figure()
plt.suptitle('Imágenes de Transmitancias')
plt.subplot(2,2,1)
plt.imshow(transR8,cmap='gray', vmin=0, vmax=255)
plt.title('Imagen de Transmitancia Canal Rojo')
plt.xlabel('Píxeles Horizontales')
plt.ylabel("Píxeles Verticales")
plt.subplot(2,2,2)
plt.imshow(transG8,cmap='gray', vmin=0, vmax=255)
plt.title('Imagen de Transmitancia Canal Verde')
plt.xlabel('Píxeles Horizontales')
plt.ylabel("Píxeles Verticales")
plt.subplot(2,2,3)
plt.imshow(transB8,cmap='gray', vmin=0, vmax=255)
plt.title('Imagen de Transmitancia Canal Azul')
plt.xlabel('Píxeles Horizontales')
plt.ylabel("Píxeles Verticales")
plt.subplot(2,2,4)
plt.imshow(transP8,cmap='gray', vmin=0, vmax=255)
plt.title('Imagen de Transmitancia Promedio')
plt.xlabel('Píxeles Horizontales')
plt.ylabel("Píxeles Verticales")

# cv2.imshow("Transmitance Image R",transR8)
# cv2.imshow("Transmitance Image G",transG8)
# cv2.imshow("Transmitance Image B",transB8)
# cv2.imshow("Transmitance Image P",transP8)
# cv2.imshow("Transmitance Image Filter",guidedTrans8)
# cv2.waitKey(0)

# TransGuided8 = cv2.imread('../PruebaFiltrada.jpg')
TransGuided8 = cv2.imread('../TransmitanciaFiltrada.jpg')
TransGuided8 = cv2.cvtColor(TransGuided8,cv2.COLOR_BGR2GRAY)
TransGuided = np.array(TransGuided8,dtype=int)
# cv2.imshow('Transmitance Image Filtered', TransGuided8)

plt.figure()
plt.suptitle('Imagen de Transmitancia Promedio Original y Filtrada')
plt.subplot(1,2,1)
plt.imshow(transP8,cmap='gray', vmin=0, vmax=255)
plt.title('Imagen de Transmitancia Original')
plt.xlabel('Píxeles Horizontales')
plt.ylabel("Píxeles Verticales")
plt.subplot(1,2,2)
plt.imshow(TransGuided8,cmap='gray', vmin=0, vmax=255)
plt.title('Imagen de Transmitancia Filtrada')
plt.xlabel('Píxeles Horizontales')
plt.ylabel("Píxeles Verticales")

Image_dehaze = np.zeros((height,width,3),dtype=int)

for i in range(0,height,1):
    for j in range(0,width,1):
        Image_dehaze[i,j,0] = int(255*(max(float(r[i,j]) - float(AtmR),0)/float(TransGuided[i,j])) + float(AtmR))
        # print(str(r[i,j]) + ' y ' + str(TransP[i,j]))
        # print(Image_dehaze[i,j,0])
        # print(Image_dehaze[i,j,0])
        Image_dehaze[i,j,1] = int(255*(max(float(g[i,j]) - float(AtmG),0)/float(TransGuided[i,j])) + float(AtmG))
        Image_dehaze[i,j,2] = int(255*(max(float(b[i,j]) - float(AtmB),0)/float(TransGuided[i,j])) + float(AtmB))
        if Image_dehaze[i,j,0] > 255:
            Image_dehaze[i,j,0] = 255
        if Image_dehaze[i,j,1] > 255:
            Image_dehaze[i,j,1] = 2555
        if Image_dehaze[i,j,2] > 255:
            Image_dehaze[i,j,2] = 255
# print(Image_dehaze)
Image_dehaze = np.array(Image_dehaze,dtype = np.uint8) 
# Image_dehaze2 = Image.fromarray(Image_dehaze,"RGB")
# Image_dehaze2.show()
# cv2.imshow("Image dehaze",Image_dehaze)

plt.figure()
plt.suptitle('Imagen Original y Final')
plt.subplot(1,2,1)
plt.imshow(src)
plt.title('Imagen Original')
plt.xlabel('Píxeles Horizontales')
plt.ylabel("Píxeles Verticales")
plt.subplot(1,2,2)
plt.imshow(Image_dehaze)
plt.title('Imagen Original con Dehazing')
plt.xlabel('Píxeles Horizontales')
plt.ylabel("Píxeles Verticales")
plt.show()

Image_dehaze = cv2.cvtColor(Image_dehaze,cv2.COLOR_RGB2BGR)
# cv2.imwrite('FotoEjemplo2Filtrada.png',Image_dehaze)
cv2.imwrite('ImagenProyectoFiltrada.png',Image_dehaze)
# cv2.waitKey(0)








            
       

        


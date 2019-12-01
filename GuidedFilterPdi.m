% ImageToFilter = imread('ProyectoTransmitancia.png');
% GuideImage = imread('ProyectoGray.png');
ImageToFilter = imread('PruebaTransmitancia.png');
GuideImage = imread('PruebaGray.png');
subplot(2,2,1)
imshow(ImageToFilter);
subplot(2,2,2)
imshow(GuideImage);
ImageFiltered = imguidedfilter(ImageToFilter,GuideImage);
subplot(2,2,3)
imshow(ImageFiltered);
% imwrite(ImageFiltered,'TransmitanciaFiltrada.jpg');
imwrite(ImageFiltered,'PruebaFiltrada.jpg');
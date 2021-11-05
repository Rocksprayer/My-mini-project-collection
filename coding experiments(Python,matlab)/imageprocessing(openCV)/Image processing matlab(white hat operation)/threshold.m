img = imread("receipt_2.png");
gs = rgb2gray(img);
gsAdj = imadjust(gs);
imshow(gsAdj)
imhist(gsAdj)
% I = imread('printedtext.png');
% imshow(I)
gsAdj1=imsharpen(gsAdj,'Radius',1,'Amount',1)
BW=imbinarize(gsAdj1,'adaptive','ForegroundPolarity','dark','sensitivity',0.38);
imshow(BW)
S = sum(BW,2);
figure, plot(S)
% cal=imbinarize(I);
% cal2=imbinarize(I,'adaptive');
% cal3=imbinarize(I,'adaptive','ForegroundPolarity','dark','sensitivity',0.38);
% montage({I,cal,cal2,cal3})

imshow(gsAdj);
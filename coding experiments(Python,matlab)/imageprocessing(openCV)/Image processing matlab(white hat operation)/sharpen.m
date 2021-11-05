% I=imread('D:\a folder to store folders UwU(file hoc tap)\Gallery\142429971_271833941026953_2042559414955876759_n.jpg');
% bw=im2bw(I);
% gray=rgb2gray(I);
% imshow(gray,[50 100])

I=imread('receipt_2.png');
  imshow(I,'Border','tight')
  hold on
  Ish = imsharpen(I);
 figure, imshow(Ish,'Border','tight');
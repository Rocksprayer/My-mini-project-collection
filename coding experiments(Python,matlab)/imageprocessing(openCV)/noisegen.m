I=imread('sudoku.png');

J=imnoise(I,'gaussian');

imshow(J)
function [signal,Ibw,stripes] = processImage()
    % This function processes an image using the algorithm 
    % developed in previous chapters.
    img=imread('receipt_2.png');
    gs = rgb2gray(img);
    gs = imadjust(gs);
    
    H = fspecial("average",3);
    gssmooth = imfilter(gs,H,"replicate");
    
    SE = strel("disk",8);  
    Ibg = imclose(gssmooth, SE);
    Ibgsub =  Ibg - gssmooth;
    Ibw = ~imbinarize(Ibgsub);
    
    SE = strel("rectangle",[3 25]);
    stripes = imopen(Ibw, SE);
    
    signal = sum(stripes,2);
    montage({img,Ibw,stripes,Ibgsub});
    figure, plot(signal);
end
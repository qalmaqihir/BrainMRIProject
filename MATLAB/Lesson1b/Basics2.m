f = imread("bubbles75.jpg");
g = im2uint8(f);
h = im2int16(f);
i = mat2gray(f) % eliminates clipping
%imshow(i),figure, imshow(f)
k = im2bw(f);


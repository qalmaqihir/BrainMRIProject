I=imread('1.gif');
% Applying median filter
I = medfilt2(I, [3,3]);
% Convert to RGB
RGB = ind2rgb(I,jet(70));
imshow(RGB)
% Extract Red Channel
R=RGB(:,:,1);
imshow(R)
% Extract Green Channel
G=RGB(:,:,2);
imshow(G);
% Extract Blue Channel
B=RGB(:,:,3);
imshow(B);
% Extract Mean, Variance, and Skewness Features 


Mean1 = mean(mean(R));
Mean2=mean(mean(G));
Mean3=mean(mean(B));

Stad1=std2(std2(R));
Stad2=std2(std2(G));
Stad3=std2(std2(B));

Skew1 = skewness(skewness(R));
Skew2 = skewness(skewness(G));
Skew3 = skewness(skewness(B));

DataR=[Mean1, Stad1, Skew1];
DataG=[Mean2, Stad2, Skew2];
DataB=[Mean3, Stad3, Skew3];








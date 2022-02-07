
I1 = imread('../Images/1.gif');
I = medfilt2(I1, [3,3]);
imtool(I)

RGB = ind2rgb(I,jet(70));
J = im2double(RGB);
imshow(J);
R = J(:,:,1);
figure, imshow(R)

% Entorpy
p = hist(length(R));
R_entropy1 = -sum(p.*log2(p));


R_entropy = entropy(R);
fprintf("Entropy of Red CHannel: %6.2f\n",R_entropy);

%Kurtosis

R_kurtosis = kurtosis(kurtosis(R));
fprintf("Kurtosis of Red Channel: %6.2f\n",R_kurtosis);



%% 
%Contrast
% NOt this one
R_contrast = contrast(R,30);
fprintf("Contrast of Red Channel: ");
R_contrast;
figure,imshow(R_contrast);

%Contrast and homogeneity
glcm=graycomatrix(R,'Offset',[2 0;0 2]);
% stats = graycoprops(glcm,{'contrast','homogeneity'});
stats_C = graycoprops(glcm,{'contrast'});
fprintf("Red Channel stats: ")
R_contrast=sum(stats_C.Contrast)/2 % There are two values for contrasts, so average them
R_contrast

stats_H = graycoprops(glcm,{'homogeneity'});
sum(stats_H.Homogeneity)/2 % There are two values for Homogeneity, so average them



% A Oneline code 
%  R_contrast = sum(graycoprops(glcm,{'contrast'}))/2;
%  R_contrast

% ANother way

 R_stats = graycoprops(glcm,{'contrast','homogeneity'});
 R_stats

 R_contrast=sum(R_stats.Contrast)/2;
 R_homogeneity=sum(R_stats.Homogeneity)/2;

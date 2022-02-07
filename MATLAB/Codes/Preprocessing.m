
I=imread('../Images/1.gif');


%% 
% ######################################################## %
      % Basic Image Filtering  in the Spatial Domain %
% ######################################################## %

% Applying some pre-processing on the Image
% Median Filter => to remove salt and pepper noise

I = medfilt2(I,[3,3]);
figure,imshow(I);

% Some Basic Image Filtering in the Spatial Domain
% Other then ordfilt2, check the Image Processing toolbox

% Some filtering, e.g ordfilt2 is used to implement greyscale morphological
% opeprations, including greyscale dilation and erosion.

% Median filter 
B= ordfilt2(I,5,ones(3,3));
figure,montage({B,I},[])
title('Medain Filtered Image (Left) vs. Original(Right)');

%Minimum filter
C = ordfilt2(I,1,ones(3,3));
figure,montage({C,I},[])
title('Min Filtered Image (Left) vs. Original(Right)');

% Maximum filter
D = ordfilt2(I,9,ones(3,3));
figure,montage({D,I},[])
title('Max Filtered Image (Left) vs. Original(Right)');



%% 
% ######################################################## %
                % Edge Preserving Filtering %
% ######################################################## %
% Edge-Preserving filtering

%imdiffusefilt
% Performing Edge-Preserving Smoothing using Anisotropic Diffusion

% Smooth the image using anisotropic diffusion. For comparison, also smooth the image using Gaussian blurring. 
% Adjust the standard deviation sigma of the Gaussian smoothing kernel so
% that textured regions, such as the grass, are smoothed a similar amount for both methods./

Idiffusion = imdiffuseest(I);
montage({Idiffusion,I,[]})
title('Smoothing Using Anisotropic Diffusion (Left) vs. Original(Right)');

% Performing 3-D Edge-aware Noise Reduction (Using MRI volumes)

%Perform edge-aware noise reduction on the volume using anisotropic diffusion.
% To prevent over-smoothing the low-contrast features in the brain, decrease the number of iterations from the default number, 5. 
% The tradeoff is that less noise is removed.

diffusedImage = imdiffusefilt(I, 'NumberOfIterations',3);
imshowpair(I,diffusedImage, 'montage');
title("Noisy Image (left) Vs. Anisotropic-Diffusion-Filtered Image (right)");


% Denoise Grayscale Image Using Non-local Means filter

% The non-local means filter removes noise from the input image but preserves the sharpness of strong edges, such as the silhouette of the man and buildings. 
% This function also smooths textured regions, such as the grass in the foreground of the image, resulting in less detail when compared to the noisy image.

[filteredImage, estDoS]=imnlmfilt(I);

%Remove noise from the image through non-local means filtering. 
%The imnlmfilt function estimates the degree of smoothing based on the standard deviation of noise in the image.
montage({filteredImage,I})
title(['Estimated Degree of Smoothing, ',' estDoS = ',num2str(estDoS)])



%% 
% ######################################################## %
                    %Texture Filtering %
% ######################################################## %


% gabor
% Description
% A gabor object represents a linear Gabor filter that is sensitive to textures with a specified wavelength and orientation.
% You can use the gabor function to create a single Gabor filter or a Gabor filter bank. 
% A filter bank is a set of filters that represent combinations of multiple wavelengths, orientations, and other optional parameters. For example, if you specify two wavelengths and three orientations, then the Gabor filter bank consists of six filters for each combination of wavelength and orientation.

% To apply a Gabor filter or a Gabor filter bank to an image, use the imgaborfilt function.

A = checkerboard(20);

wavelenght=20;
orientation =[0,45,90,135];
g = gabor(wavelenght, orientation);

% Apply the filters to the checkerboard image
outMag = imgaborfilt(A,g);
% Display the results
outSize = size(outMag);
outMag = reshape(outMag,[outSize(1:2),1,outSize(3)]);
figure, montage(outMag,'DisplayRange',[]);
title('Montage of gabor magnitude output images.');



g = gabor([5 10],[0 90]);

figure;
subplot(2,2,1)
for p = 1:length(g)
    subplot(2,2,p);
    imshow(real(g(p).SpatialKernel),[]);
    lambda = g(p).Wavelength;
    theta  = g(p).Orientation;
    title(sprintf('Re[h(x,y)], \\lambda = %d, \\theta = %d',lambda,theta));
end



I = imread('../Images/1.gif');
I = im2gray(I);

tiledlayout(1,3)
nexttile
imshow(I)
title('Original Image')
nexttile
imshow(mag,[])
title('Gabor Magnitude')
nexttile
imshow(phase,[])
title('Gabor Phase')




gaborArray = gabor([4 8],[0 90]);
gaborMag = imgaborfilt(I,gaborArray);
figure
subplot(2,2,1);
for p = 1:4
    subplot(2,2,p)
    imshow(gaborMag(:,:,p),[]);
    theta = gaborArray(p).Orientation;
    lambda = gaborArray(p).Wavelength;
    title(sprintf('Orientation=%d, Wavelength=%d',theta,lambda));
end






%% 
% ######################################################## %
           % Filtering By Property Characteristics %
% ######################################################## %

% Two improtant function

% bwareafilt
% bwpropfilt 



%% 

% ######################################################## %
           % Integral Image Domain Filtering %
% ######################################################## %








%% 
% ######################################################## %
               % Frequency Domain Filtering  %
% ######################################################## %







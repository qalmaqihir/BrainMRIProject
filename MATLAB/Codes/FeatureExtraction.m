close all
clear ll
clc

src_file = dir('../Images/*.gif');
for c=1:length(src_file)
    file_name=strcat('../Images/',src_file (c).name)
    I1 = imread(file_name);
    I = medfilt2(I1, [3,3]);

    I = adapthisteq(I);
    I = adapthisteq(I, 'ClipLimit',0.02,'Distribution','rayleigh');

    % Time delay
    pause(2);
    title("Original");
    imshow(I);
    
    RGB = ind2rgb(I,jet(70));
    pause(2);

    % Skipping the Haar Wavelet Transform

    J = im2double(RGB);

    % Extracting the individual R,G,B channels

    R = J(:,:,1);
    G = J(:,:,2);
    B = J(:,:,3);

   
    % Computing the mean, std. , skewness, 

    R_mean = mean(mean(R));
    G_mean = mean(mean(G));
    B_mean = mean(mean(B));

    R_std = std2(std2(R));
    G_std = std2(std2(G));
    B_std = std2(std2(B));

    R_skewness = skewness(skewness(R));
    G_skewness = skewness(skewness(G));
    B_skewness = skewness(skewness(B));
    
%     Giving NaN
%     p = hist(length(R));
%     R_entropy = -sum(p.*log2(p))

    R_entropy = entropy(R);
    G_entropy = entropy(G);
    B_entropy = entropy(B);

    R_kurtosis = kurtosis(kurtosis(R));
    G_kurtosis = kurtosis(kurtosis(G));
    B_kurtosis = kurtosis(kurtosis(B));



    % Contrast and Homonegeity
    
    glcm_R=graycomatrix(R,'Offset',[2 0;0 2]);
    R_stats = graycoprops(glcm_R,{'contrast','homogeneity'});
    R_contrast=sum(R_stats.Contrast)/2;
    R_homogeneity=sum(R_stats.Homogeneity)/2;


    glcm_G=graycomatrix(G,'Offset',[2 0;0 2]);
    G_stats = graycoprops(glcm_G,{'contrast','homogeneity'});
    G_contrast=sum(G_stats.Contrast)/2;
    G_homogeneity=sum(G_stats.Homogeneity)/2;

    glcm_B=graycomatrix(B,'Offset',[2 0;0 2]);
    B_stats = graycoprops(glcm_B,{'contrast','homogeneity'});
    B_contrast=sum(B_stats.Contrast)/2;
    B_homogeneity=sum(B_stats.Homogeneity)/2;
    

     % Displaying each channel with the values

    pause(2)
    subplot(3,3,4)
    imshow(R,[]);
    fprintf("Red Image: Mean = %6.2f\t",R_mean);
    fprintf("Red Image: Standard deviation = %6.2f\t",R_std);
    fprintf("Red Image: Skewness = %6.2f\t",R_skewness);
    fprintf("Red Image: Entropy = %6.2f\t",R_entropy);
    fprintf("Red Image: Kurtosis = %6.2f\t",R_kurtosis);
    fprintf("Red Image: Contrast  = %6.2f\t",R_contrast);
    fprintf("Red Image: Homogeneity = %6.2f\t",R_homogeneity);

    subplot(3,3,5)
    imshow(G,[]);
    fprintf("Green Image: Mean = %6.2f\t",G_mean);
    fprintf("GreenImage: Standard deviation = %6.2f\t",G_std);
    fprintf("Green Image: Skewness = %6.2f\t",G_skewness);
    fprintf("Green Image: Entropy = %6.2f\t",G_entropy);
    fprintf("Green Image: Kurtosis = %6.2f\t",G_kurtosis);
    fprintf("Green Image: Contrast  = %6.2f\t",G_contrast);
    fprintf("Green Image: Homogeneity = %6.2f\t",G_homogeneity);

    subplot(3,3,6)
    imshow(B);
    fprintf("Blue Image: Mean = %6.2f\t",B_mean);
    fprintf("Blue Image: Standard deviation = %6.2f\t",B_std);
    fprintf("Blue Image: Skewness = %6.2f\t",B_skewness);
    fprintf("Blue Image: Entropy = %6.2f\t",B_entropy);
    fprintf("Blue Image: Kurtosis = %6.2f\t",B_kurtosis);
    fprintf("Blue Image: Contrast  = %6.2f\t",B_contrast);
    fprintf("Blue Image: Homogeneity = %6.2f\t",B_homogeneity);



    Array_data=[R_mean,R_std, R_skewness,R_entropy,R_kurtosis,R_homogeneity,R_contrast,...
                G_mean,G_std,G_skewness,G_entropy,R_kurtosis,R_homogeneity,R_contrast, ...
                B_mean,B_std, B_skewness,B_entropy,R_kurtosis,R_homogeneity,R_contrast];

    dlmwrite('C:\Users\UCA\Desktop\BrainMRIFiles/output.txt',Array_data,'-append','delimiter','\t','precision',8);
    fprintf('\n\n');


end


    
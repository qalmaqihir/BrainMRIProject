% This function loads images from the specified directories

% This file reads Gray MRI images,   converts them to RGB
% images,

% Extracts Three channels: Red, Green and Blue from all the RGB
% images

% Find Mean, Std Deviation and Skewness of each channel of each image

% The three moments represent 9 features of the each image

% The 9 features of all images are also displayeed in MATLAB

% These 9 features of all images are written into an array

% The elements of array are written to Data.txt file



    %Load all Images
    
     close all
     clear all
     clc
     
     srcFile1 = dir ('C:\Users\muhammad.fayaz\Desktop\Digital Image Processing\Zahid\Images\*.gif');
        for c=1:length(srcFile1)
            fileName1 = strcat ('C:\Users\muhammad.fayaz\Desktop\Digital Image Processing\Zahid\Images\',srcFile1 (c).name);
           I1 = imread(fileName1);
            I = medfilt2(I1, [3,3]);
            
            I = adapthisteq(I);
            I = adapthisteq(I,'cliplimit',0.02,'Distribution','rayleigh');
             % time delay command
            pause(2)
            title('orignal');
            imshow(I);
            % time delay command
            I = ind2rgb(I, jet(70))
            pause(2);
            imshow(I)
            % time delay command
            
            [C,s] = wavedec2(I,3,'haar') %
            ca2 = appcoef2(C,s,'haar',3)
            print(C)
            ca2=gray2ind(ca2,(70))
            J = ca2;
            J=im2double(J)
            % Extract the individual red, green, and blue color channels.
                        
                        redChannel = J(:, :, 1);
                        greenChannel = J(:, :, 2);
                    	blueChannel = J(:, :, 3);
                         % time delay command
%                           pause(2)
                       imshow(greenChannel);
                        % Red image:
                        
                        subplot(3, 3, 4);
                        imshow(redChannel, []);
                        % Compute mean, Standard deviation and skewness
                        redChannel_Mean = mean(redChannel(:));
                        redChannel_Variance = std2(redChannel);
                        redChannel_Skewness = skewness(redChannel(:));
                        fprintf('Red Image.  Mean = %6.2f\t', redChannel_Mean);
                        fprintf('Red Image. Standard Deviation = %6.2f\t', redChannel_Variance);
                        fprintf('Red Image. Skewness = %6.2f\n', redChannel_Skewness);
                        
                 %      %caption = sprintf('Red Image. Variance = %6.2f', redChannel_Variance);
                 %      %caption = sprintf('Red Image.  Mean = %6.2f', redChannel_Mean);
                 %      title(caption, 'FontSize', fontSize);
                 %      % Compute and display the histogram for the Red image.
                        %pixelCountRed = PlotHistogramOfOneColorChannel(redChannel, 7, 'Histogram of Red Image', 'r');
                        
                        
                        
                        %Green Image
                        
                        subplot(3, 3, 5);
                        imshow(greenChannel, []); 
                         % Compute mean, standard deviation and skewness
                        greenChannel_Mean = mean(greenChannel(:));
                        greenChannel_StandardDeviation = std2(greenChannel);
                        greenChannel_Skewness = skewness(greenChannel(:)); 
                        fprintf('Green Image. Mean = %6.2f\t', greenChannel_Mean);
                        fprintf('Green Image. Standard Deviation = %6.2f\t', greenChannel_StandardDeviation);
                        fprintf('Green Image. Skewness = %6.2f\n', greenChannel_Skewness);
                  %     caption = sprintf('Green Image.  Mean = %6.2f', greenChannel_Mean);
                  %     title(caption, 'FontSize', fontSize);
                        
                        
                        
                        %Blue Image
                        
                        subplot(3, 3, 6);
                        imshow(blueChannel, []); 
                         % Compute mean, Standard Deviation and Skewness
                        blueChannel_Mean = mean(blueChannel(:));
                        blueChannel_StandardDeviation = std2(blueChannel);
                        blueChannel_Skewness = skewness(blueChannel(:));
                        fprintf('Blue Image. Mean = %6.2f\t', blueChannel_Mean);
                        fprintf('Blue Image. Standard Deviation = %6.2f\t', blueChannel_StandardDeviation);
                        fprintf('Blue Image. Skewness = %6.2f\n\n\n\n\n', blueChannel_Skewness);
                   %     caption = sprintf('Blue Image.  Mean = %6.2f', blueChannel_Mean);
                   %    title(caption, 'FontSize', fontSize);
                                       % Make array of all moments and write them to Data
                    % txt file
                    
                       
                       Array = [redChannel_Mean,redChannel_Variance,redChannel_Skewness,greenChannel_Mean,greenChannel_StandardDeviation,greenChannel_Skewness,blueChannel_Mean,blueChannel_StandardDeviation,blueChannel_Skewness]; %#ok<AGROW>
                       dlmwrite('C:\Users\muhammad.fayaz\Desktop\Digital Image Processing\Zahid\New.txt', Array, '-append', 'delimiter', '\t', 'precision', 8);
                       fprintf('\n\n');
                                          
         end
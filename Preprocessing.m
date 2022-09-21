%%%%%
% Choose the data set to download and load file name
file_name = "f89";
open(strcat(file_name,"\I_raw.mat"));

for i = 1:size(I_raw,3)
    I_slice = I_raw(:,:,i,1);
    figure;
    imagesc(I_slice)
    title(strcat("I_{raw} data on slice = ",string(i)))
end


%% Process Section
% close all

% slice_focus = 2;
% figure
% im = imshow(imread(I_slice));
% area_selected = drawfreehand(im);
% 
% finalFig = area_selected.createMask();
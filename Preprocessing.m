%%%%%
% Choose the data set to download and load file name
file_name = "f43";
load(strcat("Brain_Dataset\",file_name,"\I_raw.mat"));

for i = 1:size(I_raw,3)
    I_slice = I_raw(:,:,i,1);
    figure;
    imagesc(I_slice)
    title(strcat("I_{raw} data on slice = ",string(i)))
end


%% Process Section
close all

slice_focus = 9;
figure
im = imagesc(I_raw(:,:,slice_focus,1));
area_selected = drawfreehand(im);

finalFig = area_selected.createMask();
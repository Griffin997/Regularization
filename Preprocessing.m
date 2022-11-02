%%%%%
% Choose the data set to download and load file name
directory_name = "MB_References";
file_name = "BLSA_1935_06_MCIAD_m79";
raw_name = "I4D_raw.mat";
load(strcat(directory_name,"\",file_name,'\', raw_name));
save_file = false;

for i = 1:size(I4D_raw,3)
    I_slice = I4D_raw(:,:,i,1);
    figure;
    imagesc(I_slice)
    title(strcat("I_{raw} data on slice = ",string(i)))
end


%% Process Section
close all

slice_focus = 5;
figure
im = imagesc(I4D_raw(:,:,slice_focus,1));
area_selected = drawfreehand;

finalFig_area = area_selected.createMask();

slice_oi = zeros(size(I4D_raw,1),size(I4D_raw,2),size(I4D_raw,4));
for i = 1:size(I4D_raw,4)
    slice_oi(:,:,i) = I4D_raw(:,:,slice_focus,i).*finalFig_area;
end

%% Save Section
if save_file

    save(strcat(directory_name,'/',file_name,'/','rS_slice',string(slice_focus),'.mat'), 'slice_oi')

end
%% Load Section Check
load(strcat(directory_name,'/',file_name,'/','rS_slice',string(slice_focus),'.mat'))

imagesc(slice_oi(:,:,32))
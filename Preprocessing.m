%%%%%
% Choose the data set to download and load file name
directory_name = "MB_References";
file_name = "BLSA_1742_04_MCIAD_m41";
raw_name = "I4D_NESMA.mat";
load(strcat(directory_name,"\",file_name,'\', raw_name));
save_file = true;

I4D_object = I4D_NESMA;

for i = 1:size(I4D_object,3)
    I_slice = I4D_object(:,:,i,1);
    figure;
    imagesc(I_slice)
    title(strcat("I_{NESMA} data on slice = ",string(i)))
end


%% Process Section
close all

slice_focus = 3;
figure
im = imagesc(I4D_object(:,:,slice_focus,1));
area_selected = drawfreehand;

finalFig_area = area_selected.createMask();

slice_oi = zeros(size(I4D_object,1),size(I4D_object,2),size(I4D_object,4));
for i = 1:size(I4D_object,4)
    slice_oi(:,:,i) = I4D_object(:,:,slice_focus,i).*finalFig_area;
end

%% Save Section
if save_file

    save(strcat(directory_name,'/',file_name,'/','NESMA_slice',string(slice_focus),'.mat'), 'slice_oi')

end
%% Load Section Check
load(strcat(directory_name,'/',file_name,'/','NESMA_slice',string(slice_focus),'.mat'))

imagesc(slice_oi(:,:,32))
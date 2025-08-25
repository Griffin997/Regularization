%%%%%
%This file is used to skull strip brain scans. It requires the
%participation of the user to work. The NESMA filtered file and raw file
%are both loaded in to the skull stripping code. The NESMA filtered slices
%are shown. Select the slice of interest (slice_oi) to skull strip. A
%window will pop up and you will free hand draw along the boundary of the
%brain between the brain and the skull. Everywhere outside this region will
%be set to 0 while the interior will be kept. The stripped image is
%returned. The stripped image is used to make a mask that is then applied
%to the raw data (this allows these masks to added at different times). The
%final step is to crop off all the white space.

%% Load in Data

% Choose the data set to download and load file name
directory_name = "MB_References";
file_name = "BLSA_1742_04_MCIAD_m41"; %"BLSA_1935_06_MCIAD_m79"  OR  "BLSA_1742_04_MCIAD_m41"
NESMA_name = "I4D_NESMA.mat";
raw_name = "I4D_raw.mat";
addpath(file_name);
load(strcat(directory_name,"\",file_name,'\', NESMA_name)); %loads as I4D_NESMA
load(strcat(directory_name,"\",file_name,'\', raw_name)); %loads as I4D_raw

save_file = true;
slice_focus = 5;

[n_vert, n_hori, n_slices, n_elem] = size(I4D_NESMA);

%% General Cropping
%This cropping is applied to all images consistently but is also applied to
%the original dataset

I4D_NESMA_cropped = I4D_NESMA(72:216, 48:232, :, :);
I4D_raw_cropped = I4D_raw(72:216, 48:232, :, :);

if save_file
    save(strcat(file_name,'\','I4D_raw_cropped.mat'), 'I4D_raw_cropped');
    save(strcat(file_name,'\','I4D_NESMA_cropped.mat'), 'I4D_NESMA_cropped');
end

%% Check Scans Pre Skull Stripping

for i = 1:n_slices
    I_slice = I4D_NESMA(:,:,i,1);
    figure;
    imagesc(I_slice)
    title(strcat("I_{raw} data on slice = ",string(i)))
end


%% Skull Stripping
close all

figure
im = imagesc(I4D_NESMA(:,:,slice_focus,1));
area_selected = drawfreehand;

finalFig_area = area_selected.createMask();

slice_oi = zeros(n_vert, n_hori, n_elem);
for i = 1:n_elem
    slice_oi(:,:,i) = I4D_NESMA(:,:,slice_focus,i).*finalFig_area;
end

% Save Skull Stripping Result
if save_file 
    save(strcat(directory_name,'/',file_name,'/','NESMA_slice',string(slice_focus),'.mat'), 'slice_oi')
end
%% Load Section Check
load(strcat(directory_name,'/',file_name,'/','NESMA_slice',string(slice_focus),'.mat'))

figure;
imagesc(slice_oi(:,:,1))
title('TE[0]')
figure;
imagesc(slice_oi(:,:,32))
title('TE[-1]')

% imagesc(slice_oi(n_vert/4:2*n_vert/4,n_hori/4:2*n_hori/4,1))

%% Remove Extra Whitespace
load(strcat(directory_name,'/',file_name,'/','NESMA_slice',string(slice_focus),'.mat'))
slice_oi = slice_oi(72:216, 48:232, :);
imagesc(slice_oi(:,:,1))

if save_file
    save(strcat(directory_name,'/',file_name,'/','NESMA_cropped_slice',string(slice_focus),'.mat'), 'slice_oi')
end

%% Apply Identical Skull Stripping Result to Raw Data
load(strcat(directory_name,'/',file_name,'/','NESMA_slice',string(slice_focus),'.mat'))

skull_strip_mask = slice_oi(:,:,1)>0;

%Check region of brain to strip
% imagesc(skull_strip_mask)

slice_oi = zeros(n_vert, n_hori, n_elem);
for i = 1:n_elem
    slice_oi(:,:,i) = I4D_raw(:,:,slice_focus,i).*skull_strip_mask;
end

imagesc(slice_oi(:,:,1))

if save_file
    save(strcat(directory_name,'/',file_name,'/','raw_slice',string(slice_focus),'.mat'), 'slice_oi')
end

%% Remove White Space to Raw Data
load(strcat(directory_name,'/',file_name,'/','raw_slice',string(slice_focus),'.mat'))
slice_oi = slice_oi(72:216, 48:232, :);
imagesc(slice_oi(:,:,1))

if save_file
    save(strcat(directory_name,'/',file_name,'/','raw_cropped_slice',string(slice_focus),'.mat'), 'slice_oi')
end
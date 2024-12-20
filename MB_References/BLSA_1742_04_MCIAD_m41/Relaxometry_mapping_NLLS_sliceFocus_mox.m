clear;clc;

%% Inputs

load NESMA_slice5;
[dim1,dim2,dim3]=size(slice_oi);
TE=11.32:11.32:11.32*32;

s_num = 5;

%% Initialization
T2_slice=single(zeros(dim1,dim2));
RSS_slice=single(zeros(dim1,dim2));

%% Mapping
options=optimset('Display','off');
for i=1:dim1
    rng(i)
    for j=1:dim2
        if slice_oi(i,j,1)>50
            y_NESMA(:,1)=slice_oi(i,j,:);
            P0=[y_NESMA(1,1) 20 1];
            Pi=lsqnonlin(@(P) fit_mo(P,y_NESMA,TE),P0,[0 0 0],[inf 300 inf],options);
            T2_slice(i,j)=Pi(2);
            RSS_slice(i,j)=sum(fit_mo(Pi,y_NESMA,TE).^2);
            
        end
    end
end

%%

figure;
imagesc(T2_slice(:,:),[0 300]);colormap jet; axis off;colorbar;title('T2s (ms)');
sgtitle(strcat("Processed Slice Data - m41 - slice ", string(s_num)))

%% 

m41_data = struct();
m41_data.slice.T2 = T2_slice;
m41_data.slice.RSS = RSS_slice;

save(strcat('m41_moX_dataStruct_slice',string(s_num),'.mat'), '-struct', 'm41_data');

%%

load_here = false;
if load_here
    load(strcat('m41_dataStruct_slice',string(s_num),'.mat'))
    T2_slice = slice.T2;
end

%% MB Results

load("m41_dataStruct.mat")
T2l_NESMA = NESMA.T2l;
T2s_NESMA = NESMA.T2s;
MWF_NESMA = NESMA.MWF;

T2l_raw = raw.T2l;
T2s_raw = raw.T2s;
MWF_raw = raw.MWF;

%% Comparison Figures

figure;
imagesc(T2_slice(:,:) - T2s_NESMA(:,:,s_num),[-60 60]);colormap jet; axis off;colorbar;title('T2s (ms)');
sgtitle(strcat("T2 NESMA Difference Data - m41 - slice ", string(s_num)))

figure;
imagesc(T2_slice(:,:) - T2s_raw(:,:,s_num),[-60 60]);colormap jet; axis off;colorbar;title('T2s (ms)');
sgtitle(strcat("T2 Raw Difference Data - m41 - slice ", string(s_num)))

%%

load I4D_NESMA;
load I4D_raw;

%%

figure;
imagesc(slice_oi(:,:,1) - I4D_NESMA(:,:,s_num,1));colormap jet; axis off;colorbar;
sgtitle(strcat("NESMA Difference Signals - m41 - slice ", string(s_num)))

figure;
imagesc(slice_oi(:,:,1) - I4D_raw(:,:,s_num,1));colormap jet; axis off;colorbar;
sgtitle(strcat("Raw Difference Signals - m41 - slice ", string(s_num)))

%%

figure;
imagesc(abs(slice_oi(:,:,1) - I4D_NESMA(:,:,s_num,1))>0);colormap jet; axis off;colorbar;
sgtitle(strcat("NESMA Difference Signals Location - m41 - slice ", string(s_num)))

figure;
imagesc(abs(slice_oi(:,:,1) - I4D_raw(:,:,s_num,1))>0);colormap jet; axis off;colorbar;
sgtitle(strcat("Raw Difference Signals Location - m41 - slice ", string(s_num)))

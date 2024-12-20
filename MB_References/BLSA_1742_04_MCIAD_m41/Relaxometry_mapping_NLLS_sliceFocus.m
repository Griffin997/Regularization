clear;clc;

%% Inputs

load NESMA_slice5;
[dim1,dim2,dim3]=size(slice_oi);
TE=11.32:11.32:11.32*32;

s_num = 5;

%% Initialization
MWF_slice=single(zeros(dim1,dim2));
T2s_slice=single(zeros(dim1,dim2));
T2l_slice=single(zeros(dim1,dim2));
RSS_slice=single(zeros(dim1,dim2));

%% Mapping
options=optimset('Display','off');
for i=1:dim1
    rng(i)
    for j=1:dim2
        if slice_oi(i,j,1)>50
            y_NESMA(:,1)=slice_oi(i,j,:);
            P0=[y_NESMA(1,1) 0.2 20 80 1];
            Pi=lsqnonlin(@(P) fit_bi(P,y_NESMA,TE),P0,[0 0 0 0 0],[inf 0.5 60 300 inf],options);
            MWF_slice(i,j)=Pi(2);
            T2s_slice(i,j)=Pi(3);
            T2l_slice(i,j)=Pi(4);
            RSS_slice(i,j)=sum(fit_bi(Pi,y_NESMA,TE).^2);
            
        end
    end
end

%%

figure;
subplot(131);imagesc(MWF_slice(:,:),[0 0.4]);colormap jet; axis off;colorbar;title('MWF (n.u.)');
subplot(132);imagesc(T2s_slice(:,:),[0 60]);colormap jet; axis off;colorbar;title('T2s (ms)');
subplot(133);imagesc(T2l_slice(:,:),[0 140]);colormap jet; axis off;colorbar;title('T2l (ms)');
sgtitle(strcat("Processed Slice Data - m41 - slice ", string(s_num)))

%% 

m41_data = struct();
m41_data.slice.T2l = T2l_slice;
m41_data.slice.T2s = T2s_slice;
m41_data.slice.MWF = MWF_slice;
m41_data.slice.RSS = RSS_slice;

save(strcat('m41_biX_dataStruct_slice',string(s_num),'.mat'), '-struct', 'm41_data');

%%

load_here = false;
if load_here
    load(strcat('m41_dataStruct_slice',string(s_num),'_iter2_300T22.mat'))
    T2l_slice = slice.T2l;
    T2s_slice = slice.T2s;
    MWF_slice = slice.MWF;
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
subplot(131);imagesc(MWF_slice(:,:) - MWF_NESMA(:,:,s_num),[-0.4 0.4]);colormap jet; axis off;colorbar;title('MWF (n.u.)');
subplot(132);imagesc(T2s_slice(:,:) - T2s_NESMA(:,:,s_num),[-60 60]);colormap jet; axis off;colorbar;title('T2s (ms)');
subplot(133);imagesc(T2l_slice(:,:) - T2l_NESMA(:,:,s_num),[-140 140]);colormap jet; axis off;colorbar;title('T2l (ms)');
sgtitle(strcat("NESMA Difference Data - m41 - slice ", string(s_num)))

figure;
subplot(131);imagesc(MWF_slice(:,:) - MWF_raw(:,:,s_num),[-0.4 0.4]);colormap jet; axis off;colorbar;title('MWF (n.u.)');
subplot(132);imagesc(T2s_slice(:,:) - T2s_raw(:,:,s_num),[-60 60]);colormap jet; axis off;colorbar;title('T2s (ms)');
subplot(133);imagesc(T2l_slice(:,:) - T2l_raw(:,:,s_num),[-140 140]);colormap jet; axis off;colorbar;title('T2l (ms)');
sgtitle(strcat("Raw Difference Data - m41 - slice ", string(s_num)))

%% Comparison of MWF

figure;
subplot(121);
imagesc(MWF_NESMA(:,:,s_num))
title("NESMA Brain MWF")
subplot(122);
imagesc(MWF_raw(:,:,s_num))
title("Raw Brain MWF")

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

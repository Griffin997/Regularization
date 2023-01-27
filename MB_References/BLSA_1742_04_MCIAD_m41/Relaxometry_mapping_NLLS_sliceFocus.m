clear;clc;

%% Inputs

load rS_slice5;
[dim1,dim2,dim3]=size(slice_oi);
TE=11.32:11.32:11.32*32;

%% Initialization
MWF_NESMA=single(zeros(dim1,dim2));
T2s_NESMA=single(zeros(dim1,dim2));
T2l_NESMA=single(zeros(dim1,dim2));

%% Mapping
options=optimset('Display','off');
for i=1:dim1
    for j=1:dim2
        if slice_oi(i,j,1)>50
            y_NESMA(:,1)=slice_oi(i,j,:);
            P0=[y_NESMA(1,1) 0.2 20 80 1];
            Pi=lsqnonlin(@(P) fit_bi(P,y_NESMA,TE),P0,[0 0 0 0 0],[inf 0.5 60 2000 inf],options);
            MWF_NESMA(i,j)=Pi(2);
            T2s_NESMA(i,j)=Pi(3);
            T2l_NESMA(i,j)=Pi(4);
            
        end
    end
end

%%

figure;
subplot(131);imagesc(MWF_slice(:,:),[0 0.4]);colormap jet; axis off;colorbar;title('MWF (n.u.)');
subplot(132);imagesc(T2s_slice(:,:),[0 60]);colormap jet; axis off;colorbar;title('T2s (ms)');
subplot(133);imagesc(T2l_slice(:,:),[0 140]);colormap jet; axis off;colorbar;title('T2l (ms)');
sgtitle(strcat("Processed Slice Data - m41 - slice 5"))

%% 

m41_data = struct();
m41_data.slice.T2l = T2l_slice;
m41_data.slice.T2s = T2s_slice;
m41_data.slice.MWF = MWF_slice;

save('m41_dataStruct_slice5.mat', '-struct', 'm41_data');

%%

load("m41_dataStruct_slice5.mat")
T2l_slice = slice.T2l;
T2s_slice = slice.T2s;
MWF_slice = slice.MWF;

%% MB Results

load("m41_dataStruct.mat")
T2l_NESMA = NESMA.T2l;
T2s_NESMA = NESMA.T2s;
MWF_NESMA = NESMA.MWF;

T2l_raw = raw.T2l;
T2s_raw = raw.T2s;
MWF_raw = raw.MWF;

%% Comparison Figures

s = 5;
figure;
subplot(131);imagesc(MWF_slice(:,:) - MWF_NESMA(:,:,s),[-0.4 0.4]);colormap jet; axis off;colorbar;title('MWF (n.u.)');
subplot(132);imagesc(T2s_slice(:,:) - T2s_NESMA(:,:,s),[-60 60]);colormap jet; axis off;colorbar;title('T2s (ms)');
subplot(133);imagesc(T2l_slice(:,:) - T2l_NESMA(:,:,s),[-140 140]);colormap jet; axis off;colorbar;title('T2l (ms)');
sgtitle(strcat("NESMA Difference Data - m41 - slice 5"))

figure;
subplot(131);imagesc(MWF_slice(:,:) - MWF_raw(:,:,s),[-0.4 0.4]);colormap jet; axis off;colorbar;title('MWF (n.u.)');
subplot(132);imagesc(T2s_slice(:,:) - T2s_raw(:,:,s),[-60 60]);colormap jet; axis off;colorbar;title('T2s (ms)');
subplot(133);imagesc(T2l_slice(:,:) - T2l_raw(:,:,s),[-140 140]);colormap jet; axis off;colorbar;title('T2l (ms)');
sgtitle(strcat("Raw Difference Data - m41 - slice 5"))

%%

load I4D_NESMA;
load I4D_raw;

%%

figure;
imagesc(slice_oi(:,:,1) - I4D_NESMA(:,:,s,1));colormap jet; axis off;colorbar;
sgtitle(strcat("NESMA Difference Signals - m41 - slice 5"))

figure;
imagesc(slice_oi(:,:,1) - I4D_raw(:,:,s,1));colormap jet; axis off;colorbar;
sgtitle(strcat("Raw Difference Signals - m41 - slice 5"))

%%

figure;
imagesc(abs(slice_oi(:,:,1) - I4D_NESMA(:,:,s,1))>0);colormap jet; axis off;colorbar;
sgtitle(strcat("NESMA Difference Signals Location - m41 - slice 5"))

figure;
imagesc(abs(slice_oi(:,:,1) - I4D_raw(:,:,s,1))>0);colormap jet; axis off;colorbar;
sgtitle(strcat("Raw Difference Signals Location - m41 - slice 5"))
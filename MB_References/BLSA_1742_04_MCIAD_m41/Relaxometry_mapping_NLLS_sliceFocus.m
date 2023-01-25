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
subplot(131);imagesc(MWF_NESMA(:,:),[0 0.4]);colormap jet; axis off;colorbar;title('MWF (n.u.)');
subplot(132);imagesc(T2s_NESMA(:,:),[0 60]);colormap jet; axis off;colorbar;title('T2s (ms)');
subplot(133);imagesc(T2l_NESMA(:,:),[0 140]);colormap jet; axis off;colorbar;title('T2l (ms)');
sgtitle(strcat("NESMA Data - m41 - slice 5"))

%% 

m41_data = struct();
m41_data.NESMA.T2l = T2l_NESMA;
m41_data.NESMA.T2s = T2s_NESMA;
m41_data.NESMA.MWF = MWF_NESMA;

save('m41_dataStruct_slice5.mat', '-struct', 'm41_data');

%%

load("m41_dataStruct_slice5.mat")
T2l_NESMA = NESMA.T2l;
T2s_NESMA = NESMA.T2s;
MWF_NESMA = NESMA.MWF;

% T2l_raw = raw.T2l;
% T2s_raw = raw.T2s;
% MWF_raw = raw.MWF;


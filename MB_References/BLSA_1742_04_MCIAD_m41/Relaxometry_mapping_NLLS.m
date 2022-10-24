clear;clc;

%% Inputs
load I4D_NESMA;
load I4D_raw;
[dim1,dim2,dim3,dim4]=size(I4D_NESMA);
TE=11.32:11.32:11.32*32;

%% Initialization
MWF_NESMA=single(zeros(dim1,dim2,dim3));
T2s_NESMA=single(zeros(dim1,dim2,dim3));
T2l_NESMA=single(zeros(dim1,dim2,dim3));

MWF_raw=single(zeros(dim1,dim2,dim3));
T2s_raw=single(zeros(dim1,dim2,dim3));
T2l_raw=single(zeros(dim1,dim2,dim3));

%% Mapping
options=optimset('Display','off');
for k=1:dim3
    disp(['NLLS ... Slice # ' num2str(k) ' of '  num2str(dim3)]);
    for i=1:dim1
        for j=1:dim2
            if I4D_NESMA(i,j,k,1)>50
                y_NESMA(:,1)=I4D_NESMA(i,j,k,:);
                P0=[y_NESMA(1,1) 0.2 20 80 1];
                Pi=lsqnonlin(@(P) fit_bi(P,y_NESMA,TE),P0,[0 0 0 0 0],[inf 0.5 60 2000 inf],options);
                MWF_NESMA(i,j,k)=Pi(2);
                T2s_NESMA(i,j,k)=Pi(3);
                T2l_NESMA(i,j,k)=Pi(4);
                
                y_raw(:,1)=I4D_raw(i,j,k,:);
                Pi=lsqnonlin(@(P) fit_bi(P,y_raw,TE),P0,[0 0 0 0 0],[inf 0.5 60 2000 inf],options);
                MWF_raw(i,j,k)=Pi(2);
                T2s_raw(i,j,k)=Pi(3);
                T2l_raw(i,j,k)=Pi(4);
            end
        end
    end
end

%%
s=5;
figure;
subplot(221);imagesc(MWF_NESMA(:,:,s),[0 0.4]);colormap jet;impixelinfo;axis off;colorbar;title('MWF (n.u.)');
subplot(222);imagesc(T2s_NESMA(:,:,s),[0 60]);colormap jet; impixelinfo;axis off;colorbar;title('T2s (ms)');
subplot(223);imagesc(T2l_NESMA(:,:,s),[0 140]);colormap jet; impixelinfo;axis off;colorbar;title('T2l (ms)');
sgtitle("NESMA Data - m41 - slice 5 (1-10)")

figure;
subplot(221);imagesc(MWF_raw(:,:,s),[0 0.4]);colormap jet;impixelinfo;axis off;colorbar;title('MWF (n.u.)');
subplot(222);imagesc(T2s_raw(:,:,s),[0 60]);colormap jet; impixelinfo;axis off;colorbar;title('T2s (ms)');
subplot(223);imagesc(T2l_raw(:,:,s),[0 140]);colormap jet; impixelinfo;axis off;colorbar;title('T2l (ms)');
sgtitle("Raw Data - m41 - slice 5 (1-10)")

%% 

m41_data = struct();
m41_data.raw.T2l = T2l_raw;
m41_data.raw.T2s = T2s_raw;
m41_data.raw.MWF = MWF_raw;

m41_data.NESMA.T21 = T2l_NESMA;
m41_data.NESMA.T2s = T2s_NESMA;
m41_data.NESMA.MWF = MWF_NESMA;

save('m41_dataStruct.mat', '-struct', 'm41_data');

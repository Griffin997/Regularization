clear;clc;

%% Inputs
load I_raw;
I_raw = double(I_raw);
[dim1,dim2,dim3,dim4]=size(I_raw);
TE=11.32:11.32:11.32*32;

%% Initialization
MWF_raw=single(zeros(dim1,dim2,dim3));
T2s_raw=single(zeros(dim1,dim2,dim3));
T2l_raw=single(zeros(dim1,dim2,dim3));

%% Mapping
options=optimset('Display','off');
for k=1:dim3
    disp(['NLLS ... Slice # ' num2str(k) ' of '  num2str(dim3)]);
    for i=1:dim1
        for j=1:dim2
            if I_raw(i,j,k,1)>50               
                y_raw(:,1)=I_raw(i,j,k,:);
                P0=[y_raw(1,1) 0.2 20 80 1];
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
subplot(221);imagesc(MWF_raw(:,:,s),[0 0.4]);colormap jet; axis off;colorbar;title('MWF (n.u.)');
subplot(222);imagesc(T2s_raw(:,:,s),[0 60]);colormap jet; axis off;colorbar;title('T2s (ms)');
subplot(223);imagesc(T2l_raw(:,:,s),[0 140]);colormap jet; axis off;colorbar;title('T2l (ms)');
sgtitle("raw Data - f49 - slice 5 (1-10)")

%% 

f49_data = struct();
f49_data.raw.T2l = T2l_raw;
f49_data.raw.T2s = T2s_raw;
f49_data.raw.MWF = MWF_raw;

save('f49_dataStruct.mat', '-struct', 'f49_data');

%%

load("f49_dataStruct.mat")

T2l_raw = raw.T2l;
T2s_raw = raw.T2s;
MWF_raw = raw.MWF;

clear;clc;

%% Inputs

cropped_opt = true;
file_name = 'BLSA_1935_06_MCIAD_m79';
pat_num = file_name(end-2:end);

if cropped_opt
    load(strcat(file_name,'\I4D_NESMA_cropped.mat'));
    load(strcat(file_name,'\I4D_raw_cropped.mat'));
    data_NESMA = I4D_NESMA_cropped;
    data_raw = I4D_raw_cropped;
else
    load(strcat(file_name,'\I4D_NESMA.mat'));
    load(strcat(file_name,'\I4D_raw.mat'));
    data_NESMA = I4D_NESMA;
    data_raw = I4D_raw;
end

% load I4D_NESMA_cropped;         %I4D_NESMA OR I4D_NESMA_cropped
% load I4D_raw_cropped;           %I4D_raw OR I4D_raw_cropped
[dim1,dim2,dim3,dim4]=size(data_NESMA);
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
    disp(['NLLS ... Slice # ' num2str(k) ' of '  num2str(dim3) ' for ' pat_num]);
    for i=1:dim1
        rng(i)
        for j=1:dim2
            if data_NESMA(i,j,k,1)>50
                y_NESMA(:,1)=data_NESMA(i,j,k,:);
                P0=[y_NESMA(1,1) 0.2 20 80 1];
                Pi=lsqnonlin(@(P) fit_bi(P,y_NESMA,TE),P0,[0 0 0 0 0],[inf 0.5 60 2000 inf],options);
                MWF_NESMA(i,j,k)=Pi(2);
                T2s_NESMA(i,j,k)=Pi(3);
                T2l_NESMA(i,j,k)=Pi(4);
                
                y_raw(:,1)=data_raw(i,j,k,:);
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
f = figure;
f.Position = [400,300,1300,400];
subplot(131);imagesc(MWF_NESMA(:,:,s),[0 0.4]);colormap hot; axis off;colorbar;title('MWF (n.u.)');
subplot(132);imagesc(T2s_NESMA(:,:,s),[0 60]);colormap hot; axis off;colorbar;title('T2s (ms)');
subplot(133);imagesc(T2l_NESMA(:,:,s),[0 140]);colormap hot; axis off;colorbar;title('T2l (ms)');
sgtitle(strcat("NESMA Data - ", pat_num," - slice ", string(s), " (1-10)"))

f = figure;
f.Position = [400,300,1300,400];
subplot(131);imagesc(MWF_raw(:,:,s),[0 0.4]);colormap hot; axis off;colorbar;title('MWF (n.u.)');
subplot(132);imagesc(T2s_raw(:,:,s),[0 60]);colormap hot; axis off;colorbar;title('T2s (ms)');
subplot(133);imagesc(T2l_raw(:,:,s),[0 140]);colormap hot; axis off;colorbar;title('T2l (ms)');
sgtitle(strcat("Raw Data - ", pat_num," - slice ", string(s), " (1-10)"))

%% 

m79_data = struct();
m79_data.raw.T2l = T2l_raw;
m79_data.raw.T2s = T2s_raw;
m79_data.raw.MWF = MWF_raw;

m79_data.NESMA.T2l = T2l_NESMA;
m79_data.NESMA.T2s = T2s_NESMA;
m79_data.NESMA.MWF = MWF_NESMA;

%%

if cropped_opt
    save(strcat(file_name,'\',pat_num, '_cropped_dataStruct.mat'), '-struct', strcat(pat_num, '_data'));
else
    save(strcat(file_name,'\',pat_num, '_dataStruct.mat'), '-struct', strcat(pat_num, '_data'));
end

%%

load(strcat(pat_num, '_cropped_dataStruct.mat'))
T2l_NESMA = NESMA.T2l;
T2s_NESMA = NESMA.T2s;
MWF_NESMA = NESMA.MWF;

T2l_raw = raw.T2l;
T2s_raw = raw.T2s;
MWF_raw = raw.MWF;

%%
slice = 5;
imagesc(NESMA.MWF(:,:,slice), [0 0.4]); axis off; axis equal;
clear;
% warning off
load urban_162band

data=urban_detection;
mask=groundtruth;
%
img=data(:,:,[37,18,8]);
% img=data(:,:,[25,45,70]);
%  img=data(:,:,[80,60,35]);

for i=1:3
    max_f=max(max(img(:,:,i)));
    min_f=min(min(img(:,:,i)));
    img(:,:,i)=(img(:,:,i)-min_f)/(max_f-min_f);
end
figure,imshow(img);
% imwrite(img,'im.jpg');
figure,imshow(mask,[]);
% imwrite(mask,'gt.jpg');
DataTest=data;
[H,W,Dim]=size(DataTest);
num=H*W;
for i=1:Dim
    DataTest(:,:,i) = (DataTest(:,:,i)-min(min(DataTest(:,:,i)))) / (max(max(DataTest(:,:,i))-min(min(DataTest(:,:,i)))));
end
%
mask_reshape = reshape(mask, 1, num);
anomaly_map = logical(double(mask_reshape)>0);
normal_map = logical(double(mask_reshape)==0);
Y=reshape(DataTest, num, Dim)';
%% LEBSR
alpha =0.1;
lambda=10;
beta=1;
p=0.1;
m=6;
mu=1e-6;
Dict=ConstructionDict(Y,5,10);

tic
[A,S,J,L,W0,loss]=LEBSR(Y,alpha,beta,m,mu,p,lambda,Dict,1);
t_proposed=toc;

res=Y-A*S;
u_s=mean(res);
res_det=res-u_s;
r=sum(res_det.^2,1);

[AUC_LEBSR,AUCPF_LEBSR,AUCPD_LEBSR]=my_cal_AUC(r,normal_map,anomaly_map);
f_show=reshape(r,[H,W]);
f_show=(f_show-min(f_show(:)))/(max(f_show(:))-min(f_show(:)));
figure('name','LEBSR'), imshow(f_show);
% imshow(f_show);colormap(jet);colorbar;
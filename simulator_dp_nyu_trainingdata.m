%{
...
@InProceedings{Pan_2021_CVPR,
author = {Pan, Liyuan and Chowdhury, Shah and Hartley, Richard and Liu, Miaomiao and Zhang, Hongguang and Li, Hongdong},
title = {Dual Pixel Exploration: Simultaneous Depth Estimation and Image Restoration},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2021},
pages = {4340-4349} }
...
...
%}
%%
path_to_depth = '../data/NYU/nyu_depth_v2_labeled.mat';
% read mat file
data = load(path_to_depth);

datapath = '../data/simudata/';
dataset = 'NYU/training/';

name = [datapath dataset];
if ~exist([name '/dp/'],'dir'), mkdir([name '/dp/']); end
if ~exist([name '/disp/'],'dir'), mkdir([name '/disp/']); end
if ~exist([name '/gt/'],'dir'), mkdir([name '/gt/']); end
%%
newsize = [480,640];


%% load data
for idx = 0:4
    if idx == 0
        f=6/1e3;
        f_pix=3000;
        aperture_size=floor(f_pix/6);
        Fd = 10;
        scaled = 4;
        crop = 15;
    end
    if idx == 1
        f=4/1e3;
        f_pix=2000;
        aperture_size=floor(f_pix/2);
        Fd = 12;
        scaled = 4;
        crop = 15;
    end
    if idx == 2
        f=8/1e3;
        f_pix=3000;
        aperture_size=floor(f_pix/2);
        Fd = 6;
        scaled = 3;
        crop = 20;
    end
    if idx == 3
        f=35/1e3;
        f_pix=3000;
        aperture_size=floor(f_pix/5);
        Fd = 6;
        scaled = 0.5;
        crop = 20;
    end
    if idx == 4
        f=35/1e3;
        f_pix=3000;
        aperture_size=floor(f_pix/2);
        Fd = 7;%7
        scaled = 1.0;
        crop = 20;
    end
    
    for img_number = 1:1000
        
        img = data.images(:,:,:,img_number);
        RGB_img = im2double(img);
        
        depth_in = data.depths(:,:,img_number)/scaled;
        
        %% depth -> disp
        disp = scaledepth(depth_in,f_pix,f,Fd,aperture_size);
        
        % manually control focusing area,1 for foreground 0 for background
        % disp = scaledepth_m(depth_in,f_pix,f,Fd,aperture_size,0);
        
        %% simulator
        [img_left,img_right] = generatedpimage(RGB_img,disp);
        
        %% resize and save
        img_left = imresize(img_left(crop:end-crop,crop:end-crop,:), newsize);
        img_right = imresize(img_right(crop:end-crop,crop:end-crop,:), newsize);
        RGB_img = imresize(RGB_img(crop:end-crop,crop:end-crop,:), newsize);
        disp = imresize(disp(crop:end-crop,crop:end-crop), newsize);
        %
        idxi = idx*1000;
        filename = [datapath dataset sprintf('disp/%04d_dm.pfm',img_number+idxi)];
        pfmwrite(disp, filename);
        filename = [datapath dataset sprintf('dp/%04d_l.png',img_number+idxi)];
        imwrite(img_left, filename);
        filename = [datapath dataset sprintf('dp/%04d_r.png',img_number+idxi)];
        imwrite(img_right, filename);
        filename = [datapath dataset sprintf('gt/%04d_m.png',img_number+idxi)];
        imwrite(RGB_img, filename);
    end
end


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
newsize = [480,640];

%% load data
for idx = 1:2
    if idx ==1
        f=30/1e3;
        f_pix = 3000;
        aperture_size=floor(f_pix/2);
        Fd = 7;
        scaled = 0.2;
        crop = 20;
        ids = 1001;
        ide = 1350;
        dataset = 'NYU/testing/';
        name = [datapath dataset];
        if ~exist([name '/dp/'],'dir'), mkdir([name '/dp/']); end
        if ~exist([name '/disp/'],'dir'), mkdir([name '/disp/']); end
        if ~exist([name '/gt/'],'dir'), mkdir([name '/gt/']); end
    else
        
        f=30/1e3;  %4mm
        f_pix = 3000;   % focal length in number of pixels raise raise
        pixelsize=f/f_pix;
        aperture_size=floor(f_pix/2);   % 5 aperture size in grid of pixels
        Fd = 11;%7
        scaled = 0.8; %0.4;1
        crop = 20;
        ids = 1001;
        ide = 1150;
        dataset = 'NYU/testing1/';
        name = [datapath dataset];
        if ~exist([name '/dp/'],'dir'), mkdir([name '/dp/']); end
        if ~exist([name '/disp/'],'dir'), mkdir([name '/disp/']); end
        if ~exist([name '/gt/'],'dir'), mkdir([name '/gt/']); end
    end
    
    for img_number = ids:ide %1449
        
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
        idxi = 5000;
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


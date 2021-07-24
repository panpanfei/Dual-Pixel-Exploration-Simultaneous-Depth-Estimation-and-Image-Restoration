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
%% camera para
% mm
f=35/1e3; 
% focal length in number of pixels 
f_pix=3000;   
% aperture size in grid of pixels
aperture_size=floor(f_pix/2);   
%focald
Fd = 5;
% scale for depth range
scaled = 0.2; 
% crop image boundary
crop = 20; 
%resize
newsize = [480,640];

%% load data
img_name = imread('./input.png');
RGB_img = im2double(img_name);

depth_name = load('./depths.mat');
depth_in = depth_name.depths/scaled;

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
% 
imwrite(img_left, './oult_l.png');
imwrite(img_right, './oult_r.png');
imwrite(RGB_img, './oult_gt.png');


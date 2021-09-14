# For academic use only

CVPR 2021

[Dual Pixel Exploration: Simultaneous Depth Estimation and Image Restoration](https://openaccess.thecvf.com/content/CVPR2021/papers/Pan_Dual_Pixel_Exploration_Simultaneous_Depth_Estimation_and_Image_Restoration_CVPR_2021_paper.pdf)

<pre>
@InProceedings{Pan_2021_CVPR,   
author = {Pan, Liyuan and Chowdhury, Shah and Hartley, Richard and Liu, Miaomiao and Zhang, Hongguang and Li, Hongdong},  
title = {Dual Pixel Exploration: Simultaneous Depth Estimation and Image Restoration},  
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},   
month = {June},   
year = {2021},  
pages = {4340-4349} 
}
</pre>

How to use - DATASET
----------------
The codes are tested in MATLAB 2020a (64bit) under ubuntu 20.04 LTS 64bit version.

1. Download the [NYU dataset](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html).
2. Run "simulator_dp_nyu_trainingdata.m" and "simulator_dp_nyu_testingdata.m" to get the training and testing set.  


How to use - DDDNet
----------------
1. Make sure the '--input_test_file' is right.
2. Run 'dual_pixel_test.py' and 'dual_pixel_test_MB.py' to get the result of our-sys and DPD-blur dataset.
3. For the DPD-blur dataset, the result for the original resolution images is 25.4136 / 0.8394 in PSNR/SSIM.


Notes 
----------------
Should you have any questions regarding this code and the corresponding results, please contact Liyuan.Pan@anu.edu.au




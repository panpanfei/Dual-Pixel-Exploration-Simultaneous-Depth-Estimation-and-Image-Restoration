import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
import math
import argparse
import random
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

import model_test 
import utils.readpfm as rp
import utils.savepfm as sp
import scipy.misc
from PIL import Image
import PIL.Image as pil
import skimage


parser = argparse.ArgumentParser(description="DDD Network")
parser.add_argument("-g","--gpu",type=int, default=1)
parser.add_argument('-im', '--img_list', type=str, default ="./data/dpd_test.txt")
parser.add_argument('-t', '--input_test_file', type=str, default ="../../../data/dd_dp_dataset_png/")
parser.add_argument('-o', '--output_file', type=str, default ="test_results/DPD/")
args = parser.parse_args()

#Hyper Parameters
METHOD = "model_dpd"
OUT_DIR = args.output_file
GPU = range(args.gpu)
TEST_DIR = args.input_test_file

if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
 
def torchPSNR(tar_img, prd_img):

    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps
 
 
def main():
    print("init data folders - DPD-blur dataset")

    Estd_stereo = model_test.YRStereonet_3D()
    Esti_stereod = model_test.Mydeblur()
    Estd_stereo = torch.nn.DataParallel(Estd_stereo, device_ids=GPU)
    Esti_stereod = torch.nn.DataParallel(Esti_stereod, device_ids=GPU)
    Estd_stereo.cuda() #Unet
    Esti_stereod.cuda()
   
    if os.path.exists(str('./checkpoints/' + METHOD + "/Estd" + ".pkl")):
        Estd_stereo.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "//Estd" + ".pkl")), strict=True)
        print("load Estd " + " success")    
    if os.path.exists(str('./checkpoints/' + METHOD + "/Esti" + ".pkl")):
        Esti_stereod.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "//Esti" + ".pkl")), strict=True)
        print("load Esti " + " success")

        
    print("Testing...")
    psnr = []
    ipsnr = []
    # ipsnr = 0.0  
    img_file = open(args.img_list, 'r')
    img_list = img_file.readlines()
    scalep=0.0;

    for image_name in img_list:

        with torch.no_grad():
            image_idl = image_name.split(' ')[0]
            image_idr = image_name.split(' ')[1]
            image_idgt = image_name.split(' ')[2]

            image_lefto = transforms.ToTensor()((Image.open(TEST_DIR + image_idl).convert('RGB'))) # modify the path
            image_righto = transforms.ToTensor()((Image.open(TEST_DIR + image_idr).convert('RGB')))
 
            gt_image = transforms.ToTensor()((Image.open(TEST_DIR + image_idgt).convert('RGB')))

            image_id = image_idgt[-8:-4]
            image_suffix = '.png'

            image_left = Variable(image_lefto - scalep).cuda().unsqueeze(0) 
            image_right = Variable(image_righto - scalep).cuda().unsqueeze(0)
          
            est_blurdisp= Estd_stereo(image_left, image_right)
            deblur_image, est_mdisp= Esti_stereod(image_left, image_right, est_blurdisp)

            torchvision.utils.save_image(deblur_image.data + scalep, OUT_DIR + '/' + image_id + '_our.' + image_suffix)
            psnr.append(torchPSNR(deblur_image.squeeze(0)+ scalep, gt_image.cuda()))
            ipsnr.append(torchPSNR(image_left.squeeze(0)+ scalep, gt_image.cuda()))

    psnr  = torch.stack(psnr).mean().item()
    ipsnr  = torch.stack(ipsnr).mean().item()
    print('test psnr:  ',  psnr, 'input psnr:  ',  ipsnr)
 

if __name__ == '__main__':
    main()

        

        


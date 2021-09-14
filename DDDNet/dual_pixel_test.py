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
from dual_pixel_test_datasets import *
import model_test 
import scipy.misc
from PIL import Image
import PIL.Image as pil
import skimage
import PIL.Image as pil
from tqdm import tqdm


parser = argparse.ArgumentParser(description="DDD")
parser.add_argument('--start_epoch',type = int, default = 1)
parser.add_argument('--batchsize',type = int, default = 8)
parser.add_argument('--gpu',type=int, default=4)
parser.add_argument('--input_test_file', type=str, default ="../../../data/simudata/NYU/")
parser.add_argument('--img_list_t', type=str, default ="./data/nyu_test.txt")
parser.add_argument('--output_file', type=str, default ="test_results/DDDsys/")
parser.add_argument('--modelname', type=str, default = "model_nyu", help="model_nyu")
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
args = parser.parse_args()


#Hyper Parameters
METHOD = args.modelname
OUT_DIR = args.output_file
GPU = range(args.gpu)
TEST_DIR = args.input_test_file

 
if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

print("init data folders - NYU dataset")

test_dataset = GoProDataset_test(
    img_list = args.img_list_t,
    root_dir = args.input_test_file,
    transform = transforms.ToTensor()
    )
test_dataloader = DataLoader(test_dataset, batch_size = args.batchsize, shuffle=False, num_workers=args.workers)

mse = nn.MSELoss().cuda()

Estd_stereo = model_test.YRStereonet_3D()
Esti_stereod = model_test.Mydeblur()
Estd_stereo = torch.nn.DataParallel(Estd_stereo, device_ids=GPU)
Esti_stereod = torch.nn.DataParallel(Esti_stereod, device_ids=GPU)
Estd_stereo.cuda()
Esti_stereod.cuda()


Estd_stereo.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/Estd" + ".pkl")), strict=False)
print("ini load Estd " + " success")
Esti_stereod.load_state_dict(torch.load(str('./checkpoints/' + METHOD + "/Esti" + ".pkl")), strict=False)
print("ini load Esti " + " success")
  

def test():
    pscale = 0.0

    print("Testing...")
    Estd_stereo.eval()
    Esti_stereod.eval()

    with torch.no_grad():
    
        psnr = []
        errors_m = []
        for i, inputs in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            gt_image = Variable(inputs['gt'], requires_grad=False).cuda()
            gt_dispm = Variable(inputs['dispm'], requires_grad=False).cuda()
            image_left = Variable(inputs['left'] - pscale, requires_grad=False).cuda()
            image_right = Variable(inputs['right'] - pscale, requires_grad=False).cuda()

            image_id = inputs['img_id']
            image_suffix = 'png'

            
            est_blurdispt = Estd_stereo(image_left,image_right)
            deblur_imaget, est_mdispt = Esti_stereod(image_left, image_right, est_blurdispt) 

            psnr.append(torchPSNR(deblur_imaget, gt_image))


            for i in range(deblur_imaget.size(0)):
                torchvision.utils.save_image(deblur_imaget[i].data + pscale, OUT_DIR + '/' + image_id[i] + '_o.' + image_suffix)

            errors_m.append(compute_errors(gt_dispm.cpu().detach().numpy(), est_mdispt.cpu().detach().numpy()))

        mean_errors_m = np.array(errors_m).mean(0)
                
        psnr  = torch.stack(psnr).mean().item()

        print('test m:-----------')
        print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        print(("&{: 8.3f}  " * 7).format(*mean_errors_m.tolist()) + "\\\\")

        print('test psnr: ',  psnr)
 
def torchPSNR(tar_img, prd_img):

    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps

if __name__ == '__main__':
    test()

        

        


import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
from skimage import io, transform
import os
import numpy as np
import colorsys, random
import utils.readpfm as rp
import scipy.misc
import re
from struct import unpack
import torchvision.transforms.functional as TF
from torchvision.transforms import ColorJitter

def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)# / gt 0825 gang jia

    sq_rel = np.mean(((gt - pred) ** 2) / gt)# / gt

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
def readPFM(file): 
    with open(file, "rb") as f:
            # Line 1: PF=>RGB (3 channels), Pf=>Greyscale (1 channel)
        type = f.readline().decode('latin-1')
        if "PF" in type:
            channels = 3
        elif "Pf" in type:
            channels = 1
        else:
            sys.exit(1)
        # Line 2: width height
        line = f.readline().decode('latin-1')
        width, height = re.findall('\d+', line)
        width = int(width)
        height = int(height)

            # Line 3: +ve number means big endian, negative means little endian
        line = f.readline().decode('latin-1')
        BigEndian = True
        if "-" in line:
            BigEndian = False
        # Slurp all binary data
        samples = width * height * channels;
        buffer = f.read(samples * 4)
        # Unpack floats with appropriate endianness
        if BigEndian:
            fmt = ">"
        else:
            fmt = "<"
        fmt = fmt + str(samples) + "f"
        img = unpack(fmt, buffer)
        img = np.reshape(img, (height, width))
        img = np.flipud(img)
#        quit()
    return img, height, width

class GoProDataset_test(Dataset):
    def __init__(self, img_list, root_dir, crop=False, crop_size=256, rotate=False, transform=None, dploader=readPFM):
        """
        Args:
             split_file: Path to the split file
             root_dir: Directory with all the images
             transform: Optional transform to be appeared on a sample
        """
        img_file = open(img_list, 'r')
        self.img_list = img_file.readlines()
        self.root_dir = root_dir
        self.transform = transform        
        self.crop = crop
        self.crop_size = crop_size
        self.rotate = rotate   
        self.dploader = dploader  

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):

        img_id = self.img_list[idx][0:-1].split(' ')[0]
        imagel = self.img_list[idx][0:-1].split(' ')[1]
        imager = self.img_list[idx][0:-1].split(' ')[2]
        imagegt = self.img_list[idx][0:-1].split(' ')[3]
        dispm = self.img_list[idx][0:-1].split(' ')[5]
        
        depth,_,_   =self.dploader(os.path.join(self.root_dir + dispm))
        middle   = np.ascontiguousarray(depth,   dtype=np.float32)
 
        
        gt = (Image.open(os.path.join(self.root_dir + imagegt)).convert('RGB'))
        left = (Image.open(os.path.join(self.root_dir + imagel)).convert('RGB'))
        right = (Image.open(os.path.join(self.root_dir + imager)).convert('RGB'))
        
        middle = torch.from_numpy(middle)

        if self.transform:
            gt = self.transform(gt)
            left = self.transform(left)
            right = self.transform(right)
            
        return {'left': left, 'right': right, 'gt': gt,'dispm': middle, 'img_id': img_id} 

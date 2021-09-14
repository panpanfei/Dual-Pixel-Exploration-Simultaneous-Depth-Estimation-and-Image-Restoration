from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import pdb
import sys

import torch.nn.init as init
import skimage.io
from torch.nn import Parameter, Softmax

#=======================================================
def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, 0.5*math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())

class Mydeblur(nn.Module):
    def __init__(self):
        super(Mydeblur, self).__init__()

        self.feat = 128

        self.encoder1 = Encoder(in_channel = 7, out_channel = self.feat).apply(weight_init)
        self.encoder2 = Encoder(in_channel = 7, out_channel = self.feat).apply(weight_init) 
        self.encoder3 = Encoder(in_channel = 7, out_channel = self.feat).apply(weight_init) 

        self.decoder1 = Decoder(in_channel = self.feat, out_channel = 3).apply(weight_init) 
        self.decoder2 = Decoder(in_channel = self.feat, out_channel = 7).apply(weight_init)
        self.decoder3 = Decoder(in_channel = self.feat, out_channel = 7).apply(weight_init)

        self.decoderd = Decoder(in_channel = self.feat, out_channel = 1).apply(weight_init) 

 
        self.cam_attention = CAM_Module(self.feat)
 
        self.down = ConvBlock(4, self.feat, 8, 4, 2, activation='sigmoid', norm=None)
        self.conv = ConvBlock(self.feat,1,  3, 1, 1, activation='sigmoid', norm=None)
        # self.softmax = nn.Softmin(dim=1)

    def forward(self, image_left, image_right, est_blurdisp):

        H = image_left.size(2)
        W = image_left.size(3)
        images = {}
        features = {}
        residual = {}
        for i in ['lv1', 'lv2', 'lv3']:
                images[i] = {}
                features[i] = {}
                residual[i] = {}
        images['lv1'] = torch.cat((image_left, image_right, est_blurdisp.unsqueeze(1)),1)
        images['lv2'] = {}
        images['lv2'][0] = images['lv1'][:,:,0:int(H/2),:]
        images['lv2'][1] = images['lv1'][:,:,int(H/2):H,:]
        images['lv3'] = {}
        images['lv3'][0] = images['lv2'][0][:,:,:,0:int(W/2)]
        images['lv3'][1] = images['lv2'][0][:,:,:,int(W/2):W]
        images['lv3'][2] = images['lv2'][1][:,:,:,0:int(W/2)]
        images['lv3'][3] = images['lv2'][1][:,:,:,int(W/2):W]            

        features['lv3'][0] = self.encoder3(images['lv3'][0])
        #print(features['lv3'][0].size())
        features['lv3'][1] = self.encoder3(images['lv3'][1])
        features['lv3'][2] = self.encoder3(images['lv3'][2])
        features['lv3'][3] = self.encoder3(images['lv3'][3])
        features['lv3']['top'] = torch.cat((features['lv3'][0], features['lv3'][1]), 3)
        features['lv3']['bot'] = torch.cat((features['lv3'][2], features['lv3'][3]), 3)
        features['lv3']['merge'] = torch.cat((features['lv3']['top'], features['lv3']['bot']), 2)
        #print(features['lv3']['merge'].size())
        residual['lv3']['top'] = self.decoder3(features['lv3']['top'])
        residual['lv3']['bot'] = self.decoder3(features['lv3']['bot'])
        residual['lv3']['merge'] = torch.cat((residual['lv3']['top'], residual['lv3']['bot']),2)

        features['lv2'][0] = self.encoder2(images['lv2'][0] + residual['lv3']['top'])
        features['lv2'][1] = self.encoder2(images['lv2'][1] + residual['lv3']['bot'])
        features['lv2']['merge'] = torch.cat((features['lv2'][0], features['lv2'][1]), 2) + features['lv3']['merge']
        residual['lv2']['merge'] = self.decoder2(features['lv2']['merge'])

        features['lv1']['merge'] = self.encoder1(images['lv1'] + residual['lv2']['merge']) + features['lv2']['merge']
        
        featuresf = self.down(torch.cat(((image_left-image_right), est_blurdisp.unsqueeze(1)),1))

        feat = self.cam_attention(featuresf)
        deblur_image = self.decoder1(features['lv1']['merge']+feat)
        est_mdisp = self.decoderd(features['lv1']['merge']+feat)
        return deblur_image, est_mdisp.squeeze(1)  



class YRStereonet_3D(nn.Module):
    def __init__(self, maxdisp=12, channel = 128, mchannel = 32, patchsize = 192):
        super(YRStereonet_3D, self).__init__()
        self.maxdisp = maxdisp
        self.channel = channel
        self.mchannel = mchannel
        self.patchsize = patchsize
        self.feature = Feature()
        self.features = Feature()
        self.matching = Matching()
        self.disp = Disp(self.maxdisp)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, xl, yr):  
        x = self.feature(xl) 
        y = self.feature(yr)
  
        with torch.cuda.device_of(x):
            cost = x.new().resize_(x.size()[0], x.size()[1]*2, int(self.maxdisp/3),  x.size()[2],  x.size()[3])    
        for i in range(int(self.maxdisp/3)):
            if i > 0 : 
                cost[:,:x.size()[1], i,:,i:] = x[:,:,:,i:]
                cost[:,x.size()[1]:, i,:,i:] = y[:,:,:,:-i]
            else:
                cost[:,:x.size()[1],i,:,i:] = x
                cost[:,x.size()[1]:,i,:,i:] = y
        
        cost = self.matching(cost)   

        displ = self.disp(cost)
        return displ

	

class Encoder(nn.Module):
    def __init__(self, in_channel = 3, out_channel = 128):
        super(Encoder, self).__init__()
        #Conv1
        self.layer1 = nn.Conv2d(in_channel, 32, kernel_size=3, padding=1)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
            )
        #Conv2
        self.layer5 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.layer6 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
            )
        self.layer7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
            )
        #Conv3
        self.layer9 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.layer10 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
            )
        self.layer11 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, out_channel, kernel_size=3, padding=1)
            )
        
    def forward(self, x):
        #Conv1
        x = self.layer1(x)
        x = self.layer2(x) + x
        x = self.layer3(x) + x
        #Conv2
        x = self.layer5(x)
        x = self.layer6(x) + x
        x = self.layer7(x) + x
        #Conv3
        x = self.layer9(x)    
        x = self.layer10(x) + x
        x = self.layer11(x) + x 
        return x

class Decoder(nn.Module):
    def __init__(self, in_channel = 128, out_channel = 3):
        super(Decoder, self).__init__()        
        # Deconv3
        self.layer13 = nn.Sequential(
            nn.Conv2d(in_channel, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
            )
        self.layer14 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
            )
        self.layer16 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        #Deconv2
        self.layer17 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
            )
        self.layer18 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
            )
        self.layer20 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        #Deconv1
        self.layer21 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
            )
        self.layer22 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
            )
        self.layer24 = nn.Conv2d(32, out_channel, kernel_size=3, padding=1)
        
    def forward(self,x):        
        #Deconv3
        x = self.layer13(x) + x
        x = self.layer14(x) + x
        x = self.layer16(x)                
        #Deconv2
        x = self.layer17(x) + x
        x = self.layer18(x) + x
        x = self.layer20(x)
        #Deconv1
        x = self.layer21(x) + x
        x = self.layer22(x) + x
        x = self.layer24(x)
        return x


class CAM_Module(nn.Module):
    """ Channel attention module"""
    # paper: Dual Attention Network for Scene Segmentation
    def __init__(self,in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature ( B X C X H X W)
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        # print(attention.size())
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out
class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

#-----------------
class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        self.start = nn.Sequential(
            BasicConv(3, 32, kernel_size=3, padding=1),
            BasicConv(32, 64, kernel_size=3, stride=1, padding=1))
        self.layer1 = nn.Sequential(
            BasicConv(64, 128, kernel_size=3, stride=1, padding=4, dilation=4),
            BasicConv(128, 128, kernel_size=3, stride=1, padding=8,dilation=8))
        # self.start = nn.Sequential(
        #     BasicConv(3, 32, kernel_size=3, padding=1),
        #     BasicConv(32, 64, kernel_size=3, stride=3, padding=1))
        # self.layer1 = nn.Sequential(
        #     BasicConv(64, 128, kernel_size=3, stride=1, padding=4, dilation=4),
        #     BasicConv(128, 128, kernel_size=3, stride=1, padding=8,dilation=8))

        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64,64)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))


        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16,16)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.end = nn.Sequential(
            BasicConv(192, 96, kernel_size=3, padding=1),
            BasicConv(96, 32, kernel_size=1, bn=False, relu=False, padding=0))

    def forward(self, x):
        # print(x.size())
        x = self.start(x)
        # print('1',x.size())

        x = self.layer1(x)
        # print('2',x.size())


        output_branch1 = self.branch1(x)
        output_branch1 = F.interpolate(output_branch1, (x.size()[2],x.size()[3]),mode='bilinear',align_corners=True)
        output_branch3 = self.branch3(x)
        output_branch3 = F.interpolate(output_branch3, (x.size()[2],x.size()[3]),mode='bilinear',align_corners=True)
              
        output_feature = torch.cat((output_branch1, output_branch3,  x), 1)
        output_feature = self.end(output_feature)
        # print('2',output_feature.size())

        return output_feature

class Matching(nn.Module):
    def __init__(self):
        super(Matching, self).__init__()
        self.start =  nn.Sequential(
            BasicConv(64, 32, is_3d=True, kernel_size=3, padding=1),
            BasicConv(32, 48, is_3d=True, kernel_size=3, stride=2, padding=1),
            BasicConv(48, 48, is_3d=True, kernel_size=3, padding=1))
        self.conv1a = nn.Sequential(
            BasicConv(48, 64, is_3d=True, kernel_size=3, stride=2, padding=1),
            BasicConv(64, 64, is_3d=True, kernel_size=3, padding=1))
        self.conv2a = nn.Sequential(
            BasicConv(64, 96, is_3d=True, kernel_size=3, stride=2, padding=1),
            BasicConv(96, 96, is_3d=True, kernel_size=3, padding=1))
        self.conv3a = nn.Sequential(
            BasicConv(96, 128, is_3d=True, kernel_size=3, stride=2, padding=1),
            BasicConv(128, 128, is_3d=True, kernel_size=3, padding=1))
        self.deconv3a = Conv2x(128, 96, is_3d=True, deconv=True)
        self.deconv2a = Conv2x(96, 64, is_3d=True, deconv=True)
        self.deconv1a = Conv2x(64, 48, is_3d=True, deconv=False) # True -> False
        self.end = nn.Sequential(
            BasicConv(48, 24, is_3d=True, kernel_size=4, padding=1, stride=2, deconv=True),
            BasicConv(24, 1, is_3d=True, kernel_size=3, padding=1, stride=1, bn=False, relu=False))
        
    def forward(self, x):
        x = self.start(x)
        rem0 = x
        x = self.conv1a(x)
        #rem1 = x
        #x = self.conv2a(x)
        #rem2 = x
        #x = self.conv3a(x)
        #pdb.set_trace()
        #x = self.deconv3a(x, rem2)
        # = self.deconv2a(x, rem1)
        # print(x.size(), rem0.size())
        x = self.deconv1a(x, rem0) #(4 64 4 48 48,  4 48 2 96 96)
        x = self.end(x)
        return x

class Refine(nn.Module):
    def __init__(self):
        super(Refine, self).__init__()
        self.start_d = nn.Conv2d(1, 16, 3, padding=1)
        self.refine = nn.Sequential(
            nn.Conv2d(17, 32, 3, padding=1),
            ResBlock(32, 32, dilation=1),
            ResBlock(32, 32, dilation=2),
            ResBlock(32, 32, dilation=4),
            ResBlock(32, 32, dilation=8),
            ResBlock(32, 32, dilation=1),
            ResBlock(32, 32, dilation=1),
            nn.Conv2d(32, 1, 3, padding=1))
        
    def forward(self,d,dl):
        x = self.start_d(d)
        x = torch.cat((x, dl), 1)
        x = self.refine(x) + d
        x = F.relu(x, inplace=True)
        return x.squeeze(1)
def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation, dilation = dilation, bias=False), nn.BatchNorm2d(out_planes))

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dilation=1, stride=1, downsample=None):
        super(ResBlock, self).__init__()

        # To keep the shape of input and output same when dilation conv, we should compute the padding:
        # Reference:
        #   https://discuss.pytorch.org/t/how-to-keep-the-shape-of-input-and-output-same-when-dilation-conv/14338
        # padding = [(o-1)*s+k+(k-1)*(d-1)-i]/2, here the i is input size, and o is output size.
        # set o = i, then padding = [i*(s-1)+k+(k-1)*(d-1)]/2 = [k+(k-1)*(d-1)]/2      , stride always equals 1
        # if dilation != 1:
        #     padding = (3+(3-1)*(dilation-1))/2
        padding = dilation

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride,padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride,padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.downsample = downsample
        self.stride = stride
        self.in_ch = in_channel
        self.out_ch = out_channel
        self.p = padding
        self.d = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual

        out = self.relu2(out)

        return out

class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()
#        print(in_channels, out_channels, deconv, is_3d, bn, relu, kwargs)
        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

class Disp(nn.Module):
    def __init__(self, maxdisp=12):
        super(Disp, self).__init__()
        self.maxdisp = maxdisp
        self.softmax = nn.Softmin(dim=1)
        self.disparity = DisparityRegression(maxdisp=self.maxdisp)

    def forward(self, x):
        x = F.interpolate(x, [self.maxdisp, x.size()[3], x.size()[4]], mode='trilinear', align_corners=False)
        # print('2.1', x.size())
        x = torch.squeeze(x, 1)
        # print('2.2', x.size()) (4, 12, 192, 192)
        x = self.softmax(x)
        # print('2.3', x.size()) (4, 12, 192, 192)
        x = self.disparity(x)
        # print('2.4', x.size()) (4, 192, 192)
        return x

class DisparityRegression(nn.Module):
    def __init__(self, maxdisp):
        super(DisparityRegression, self).__init__()
        self.maxdisp = maxdisp

    def forward(self, x):
        assert(x.is_contiguous() == True)
        with torch.cuda.device_of(x):
            disp = torch.reshape(torch.arange(0, self.maxdisp, device=torch.cuda.current_device(), dtype=torch.float32),[1,self.maxdisp,1,1])
            disp = disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
            out = torch.sum(x * disp, 1)
        return out

class Conv2x(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, bn=True, relu=True):
        super(Conv2x, self).__init__()
        self.concat = concat
        
        if deconv and is_3d: 
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3
        self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, stride=1, padding=1) # stride 2->1

        if self.concat: 
            self.conv2 = BasicConv(out_channels*2, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
    def forward(self, x, rem):
        # (4, 64, 1, 48, 48)
        x = self.up2(x)
        # x (4, 64, 2, 96, 96) rem (4, 48, 2, 96, 96)
        x = self.conv1(x) # should be (4, 48, 2, 96, 96)
        # print(x.size())

        assert(x.size() == rem.size())
        if self.concat:
            x = torch.cat((x, rem), 1)
        else: 
            x = x + rem
        x = self.conv2(x)
        return x
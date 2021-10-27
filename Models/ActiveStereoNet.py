import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import pdb
from .blocks import *

import pdb

class SiameseTower(nn.Module):
    def __init__(self, scale_factor):
        super(SiameseTower, self).__init__()

        self.conv1 = conv_block(nc_in=1, nc_out=32, k=3, s=1, norm=None, act=None)
        res_blocks = [ResBlock(32, 32, 3, 1, 1)] * 3
        self.res_blocks = nn.Sequential(*res_blocks)    
        convblocks = [conv_block(32, 32, k=3, s=2, norm='bn', act='lrelu')] * int(math.log2(scale_factor))
        self.conv_blocks = nn.Sequential(*convblocks)
        self.conv2 = conv_block(nc_in=32, nc_out=32, k=3, s=1, norm=None, act=None)
    
    def forward(self, x):

        #pdb.set_trace()
        out = self.conv1(x)
        out = self.res_blocks(out)
        out = self.conv_blocks(out)
        out = self.conv2(out)

        return out

class CoarseNet(nn.Module):
    def __init__(self, maxdisp, scale_factor, img_shape):
        super(CoarseNet, self).__init__()
        self.maxdisp = maxdisp
        self.scale_factor = scale_factor
        self.img_shape = img_shape

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.conv3d_1 = conv3d_block(64, 32, 3, 1, norm='bn', act='lrelu')
        self.conv3d_2 = conv3d_block(32, 32, 3, 1, norm='bn', act='lrelu')
        self.conv3d_3 = conv3d_block(32, 32, 3, 1, norm='bn', act='lrelu')
        self.conv3d_4 = conv3d_block(32, 32, 3, 1, norm='bn', act='lrelu')

        self.conv3d_5 = conv3d_block(32, 1, 3, 1, norm=None, act=None)
        self.disp_reg = DisparityRegression(self.maxdisp)

    def costVolume(self, refimg_fea, targetimg_fea, views):
        #Cost Volume
        cost = torch.zeros(refimg_fea.size()[0], refimg_fea.size()[1]*2, self.maxdisp//self.scale_factor, refimg_fea.size()[2], refimg_fea.size()[3]).cuda()
        views = views.lower()
        if views == 'left':
            for i in range(self.maxdisp//self.scale_factor):
                if i > 0:
                    cost[:, :refimg_fea.size()[1], i, :, i:] = refimg_fea[:,:,:,i:]
                    cost[:, refimg_fea.size()[1]:, i, :, i:] = targetimg_fea[:,:,:,:-i]
                else:
                    cost[:, :refimg_fea.size()[1], i, :,:] = refimg_fea
                    cost[:, refimg_fea.size()[1]:, i, :,:] = targetimg_fea
        elif views == 'right':
            for i in range(self.maxdisp // self.scale_factor):
                if i > 0:
                    cost[:, :refimg_fea.size()[1], i, :, :-i] = refimg_fea[:,:,:,i:]
                    cost[:, refimg_fea.size()[1]:, i, :, :-i] = targetimg_fea[:,:,:,:-i]
                else:
                    cost[:, :refimg_fea.size()[1], i, :,:] = refimg_fea
                    cost[:, refimg_fea.size()[1]:, i, :,:] = targetimg_fea
        return cost

    def Coarsepred(self, cost):
        #pdb.set_trace()
        cost = self.conv3d_1(cost)
        cost = self.conv3d_2(cost) + cost
        cost = self.conv3d_3(cost) + cost
        cost = self.conv3d_4(cost) + cost
        
        cost = self.conv3d_5(cost)
        #pdb.set_trace()
        cost = F.interpolate(cost, size=[self.maxdisp, self.img_shape[1], self.img_shape[0]], mode='trilinear', align_corners=False)
        #pdb.set_trace()
        pred = cost.softmax(dim=2).squeeze(dim=1)
        pred = self.disp_reg(pred)

        return pred
    
    def forward(self, refimg_fea, targetimg_fea):
        '''
        Args:
            refimg_fea: output of SiameseTower for a left image
            targetimg_fea: output of SiameseTower for the right image

        '''
        cost_left = self.costVolume(refimg_fea, targetimg_fea, 'left')
        #cost_right = self.costVolume(refimg_fea, targetimg_fea, 'right')

        pred_left = self.Coarsepred(cost_left)
        #pred_right = self.Coarsepred(cost_right)

        return pred_left#, pred_right
        


        
class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # stream_1, left_img
        self.conv1_s1 = conv_block(1, 16, 3, 1)
        self.resblock1_s1 = ResBlock(16, 16, 3, 1, 1)
        self.resblock2_s1 = ResBlock(16, 16, 3, 1, 2)

        # stream_2, upsampled low_resolution disp
        self.conv1_s2 = conv_block(1, 16, 1, 1)
        self.resblock1_s2 = ResBlock(16, 16, 3, 1, 1)
        self.resblock2_s2 = ResBlock(16, 16, 3, 1, 2)

        # cat
        self.resblock3 = ResBlock(32, 32, 3, 1, 4)
        self.resblock4 = ResBlock(32, 32, 3, 1, 8)
        self.resblock5 = ResBlock(32, 32, 3, 1, 1)
        self.resblock6 = ResBlock(32, 32, 3, 1, 1)
        self.conv2 = conv_block(32, 1, 3, 1)

    def forward(self, left_img, up_disp):
        
        stream1 = self.conv1_s1(left_img)
        stream1 = self.resblock1_s1(stream1)
        stream1 = self.resblock2_s1(stream1)

        stream2 = self.conv1_s2(up_disp)
        stream2 = self.resblock1_s2(stream2)
        stream2 = self.resblock2_s2(stream2)

        #print(stream1.shape, stream2.shape)
        out = torch.cat((stream1, stream2), 1)
        out = self.resblock3(out)
        out = self.resblock4(out)
        out = self.resblock5(out)
        out = self.resblock6(out)
        out = self.conv2(out)

        return out

        
class InvalidationNet(nn.Module):
    def __init__(self):
        super(InvalidationNet, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        resblocks1 = [ResBlock(64, 64, 3, 1, 1)] * 5
        self.resblocks1 = nn.Sequential(*resblocks1)
        self.conv1 = conv_block(64, 1, 3, 1, norm=None, act=None)

        self.conv2 = conv_block(5, 32, 3, 1)
        resblocks2 = [ResBlock(32, 32, 3, 1, 1)] * 4
        self.resblocks2 = nn.Sequential(*resblocks2)
        self.conv3 = conv_block(32, 1, 3, 1, norm=None, act=None)

    def forward(self, left_tower, right_tower, left_img, freso_disp):

        features = torch.cat((left_tower, right_tower), 1)
        out1 = self.resblocks1(features)
        out1 = self.conv1(out1)

        input = torch.cat((left_img, out1, freso_disp), 1)
        
        out2 = self.conv2(input)
        out2 = self.resblocks2(out2)
        out2 = self.conv3(out2)

        return out2
        


class ActiveStereoNet(nn.Module):
    def __init__(self, maxdisp, scale_factor, img_shape):
        super(ActiveStereoNet, self).__init__()
        self.maxdisp = maxdisp
        self.scale_factor = scale_factor
        self.SiameseTower = SiameseTower(scale_factor)
        self.CoarseNet = CoarseNet(maxdisp, scale_factor, img_shape)
        self.RefineNet = RefineNet()
        #self.InvalidationNet = InvalidationNet()
        self.img_shpae = img_shape


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
        
        
    
    def forward(self, left, right):
        
        #pdb.set_trace()
        left_tower = self.SiameseTower(left)
        right_tower = self.SiameseTower(right)
        #pdb.set_trace()
        coarseup_pred = self.CoarseNet(left_tower, right_tower)
        res_disp = self.RefineNet(left, coarseup_pred)

        ref_pred = coarseup_pred + res_disp
        
        

        return nn.ReLU(False)(ref_pred)


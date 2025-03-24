from argparse import Namespace

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import sys
from model import config as cfg 
import matplotlib.pyplot as plt
from torch.nn import Module, Conv2d, Parameter, Softmax
import cv2
import os

from huggingface_hub import PyTorchModelHubMixin




def patch_split(input, bin_size):
    """
    b c (bh rh) (bw rw) -> b (bh bw) rh rw c
    """
    B, C, H, W = input.size()
    bin_num_h = bin_size[0]
    bin_num_w = bin_size[1]
    rH = H // bin_num_h
    rW = W // bin_num_w
    out = input.view(B, C, bin_num_h, rH, bin_num_w, rW)
    out = out.permute(0,2,4,3,5,1).contiguous() # [B, bin_num_h, bin_num_w, rH, rW, C]
    out = out.view(B,-1,rH,rW,C) # [B, bin_num_h * bin_num_w, rH, rW, C]
    return out

def patch_recover(input, bin_size):
    """
    b (bh bw) rh rw c -> b c (bh rh) (bw rw)
    """
    B, N, rH, rW, C = input.size()
    bin_num_h = bin_size[0]
    bin_num_w = bin_size[1]
    H = rH * bin_num_h
    W = rW * bin_num_w
    out = input.view(B, bin_num_h, bin_num_w, rH, rW, C)
    out = out.permute(0,5,1,3,2,4).contiguous() # [B, C, bin_num_h, rH, bin_num_w, rW]
    out = out.view(B, C, H, W) # [B, C, H, W]
    return out

class GCN(nn.Module):
    def __init__(self, num_node, num_channel):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv2d(num_node, num_node, kernel_size=1, bias=False)
        self.relu = nn.PReLU(num_node)
        self.conv2 = nn.Linear(num_channel, num_channel, bias=False)
    def forward(self, x):
        # x: [B, bin_num_h * bin_num_w, K, C]
        out = self.conv1(x)
        out = self.relu(out + x)
        out = self.conv2(out)
        return out
class CAAM(nn.Module):
    """
    Class Activation Attention Module
    """
    def __init__(self, feat_in, num_classes, bin_size, norm_layer):
        super(CAAM, self).__init__()
        feat_inner = feat_in // 2
        self.norm_layer = norm_layer
        self.bin_size = bin_size
        self.dropout = nn.Dropout2d(0.1)
        self.conv_cam = nn.Conv2d(feat_in, num_classes, kernel_size=1)
        self.pool_cam = nn.AdaptiveAvgPool2d(bin_size)
        self.sigmoid = nn.Sigmoid()

        bin_num = bin_size[0] * bin_size[1]
        self.gcn = GCN(bin_num, feat_in)
        self.fuse = nn.Conv2d(bin_num, 1, kernel_size=1)
        self.proj_query = nn.Linear(feat_in, feat_inner)
        self.proj_key = nn.Linear(feat_in, feat_inner)
        self.proj_value = nn.Linear(feat_in, feat_inner)
              
        self.conv_out = nn.Sequential(
            nn.Conv2d(feat_inner, feat_in, kernel_size=1, bias=False),
            norm_layer(feat_in),
            nn.PReLU(feat_in)
        )
        self.scale = feat_inner ** -0.5
        self.relu = nn.PReLU(1)

    def forward(self, x):
        cam = self.conv_cam(x) # [B, K, H, W]
        cls_score = self.sigmoid(self.pool_cam(cam)) # [B, K, bin_num_h, bin_num_w]

        residual = x # [B, C, H, W]
        cam = patch_split(cam, self.bin_size) # [B, bin_num_h * bin_num_w, rH, rW, K]
        x = patch_split(x, self.bin_size) # [B, bin_num_h * bin_num_w, rH, rW, C]

        B = cam.shape[0]
        rH = cam.shape[2]
        rW = cam.shape[3]
        K = cam.shape[-1]
        C = x.shape[-1]
        cam = cam.view(B, -1, rH*rW, K) # [B, bin_num_h * bin_num_w, rH * rW, K]
        x = x.view(B, -1, rH*rW, C) # [B, bin_num_h * bin_num_w, rH * rW, C]

        bin_confidence = cls_score.view(B,K,-1).transpose(1,2).unsqueeze(3) # [B, bin_num_h * bin_num_w, K, 1]
        pixel_confidence = F.softmax(cam, dim=2)

        local_feats = torch.matmul(pixel_confidence.transpose(2, 3), x) * bin_confidence # [B, bin_num_h * bin_num_w, K, C]
        local_feats = self.gcn(local_feats) # [B, bin_num_h * bin_num_w, K, C]
        global_feats = self.fuse(local_feats) # [B, 1, K, C]
        global_feats = self.relu(global_feats).repeat(1, x.shape[1], 1, 1) # [B, bin_num_h * bin_num_w, K, C]
        
        query = self.proj_query(x) # [B, bin_num_h * bin_num_w, rH * rW, C//2]
        key = self.proj_key(local_feats) # [B, bin_num_h * bin_num_w, K, C//2]
        value = self.proj_value(global_feats) # [B, bin_num_h * bin_num_w, K, C//2]
        
        aff_map = torch.matmul(query, key.transpose(2, 3)) # [B, bin_num_h * bin_num_w, rH * rW, K]
        aff_map = F.softmax(aff_map, dim=-1)
        out = torch.matmul(aff_map, value) # [B, bin_num_h * bin_num_w, rH * rW, C]
        
        out = out.view(B, -1, rH, rW, value.shape[-1]) # [B, bin_num_h * bin_num_w, rH, rW, C]
        out = patch_recover(out, self.bin_size) # [B, C, H, W]
        
        out_conv = self.conv_out(out)
        out = residual + out_conv
        
        return out



class ConvBatchnormRelu(nn.Module):
    '''
    This class defines the convolution layer with batch normalization and PReLU activation
    '''

    def __init__(self, nIn, nOut, kSize=3, stride=1, groups=1,dropout_rate=0.0):
        '''

        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(nOut)
        self.act = nn.PReLU(nOut)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        # output = self.conv1(output)
        output = self.bn(output)
        output = self.act(output)
        if self.dropout:
            output = self.dropout(output)
        return output


# class C(nn.Module):
#     '''
#     This class is for a convolutional layer.
#     '''

#     def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
#         '''

#         :param nIn: number of input channels
#         :param nOut: number of output channels
#         :param kSize: kernel size
#         :param stride: optional stride rate for down-sampling
#         '''
#         super().__init__()
#         padding = int((kSize - 1) / 2)
#         self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False,
#                               groups=groups)

#     def forward(self, input):
#         '''
#         :param input: input feature map
#         :return: transformed feature map
#         '''
#         output = self.conv(input)
#         return output

class DilatedConv(nn.Module):
    '''
    This class defines the dilated convolution.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut,kSize, stride=stride, padding=padding, bias=False,
                              dilation=d, groups=groups)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output

class BatchnormRelu(nn.Module):
    '''
        This class groups the batch normalization and PReLU activation
    '''
    def __init__(self, nOut):
        '''
        :param nOut: output feature maps
        '''
        super().__init__()
        self.nOut=nOut
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: normalized and thresholded feature map
        '''
        output = self.bn(input)
        output = self.act(output)
        return output


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, stride=1, dilation=1):
        super(DepthwiseSeparableConv, self).__init__()
        padding = int((kernel_size - 1) / 2) * dilation
        self.depthwise = nn.Conv2d(nin, nin, kernel_size, stride, padding, dilation, groups=nin, bias=False)
        self.pointwise = nn.Conv2d(nin, nout, 1, 1, 0, 1, 1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class StrideESP(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        n = int(nOut/5)
        n1 = nOut - 4*n
        self.c1 = DilatedConv(nIn, n, 3, 2)
        self.d1 = DilatedConv(n, n1, 3, 1, 1)
        self.d2 = DilatedConv(n, n, 3, 1, 2)
        self.d4 = DilatedConv(n, n, 3, 1, 4)
        self.d8 = DilatedConv(n, n, 3, 1, 8)
        self.d16 = DilatedConv(n, n, 3, 1, 16)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-3)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        combine = torch.cat([d1, add1, add2, add3, add4],1)
        output = self.bn(combine)
        output = self.act(output)
        return output




class DepthwiseESP(nn.Module):
    '''
    This class defines the ESP block, which is based on the following principle
        Reduce ---> Split ---> Transform --> Merge
    '''
    def __init__(self, nIn, nOut, add=True):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param add: if true, add a residual connection through identity operation. You can use projection too as
                in ResNet paper, but we avoid to use it if the dimensions are not the same because we do not want to
                increase the module complexity
        '''
        super().__init__()
        n = max(int(nOut/5),1)
        n1 = max(nOut - 4*n,1)
        self.c1 = DepthwiseSeparableConv(nIn, n, 1, 1)
        self.d1 = DepthwiseSeparableConv(n, n1, 3, 1, 1) # dilation rate of 2^0
        self.d2 = DepthwiseSeparableConv(n, n, 3, 1, 2) # dilation rate of 2^1
        self.d4 = DepthwiseSeparableConv(n, n, 3, 1, 4) # dilation rate of 2^2
        self.d8 = DepthwiseSeparableConv(n, n, 3, 1, 8) # dilation rate of 2^3
        self.d16 = DepthwiseSeparableConv(n, n, 3, 1, 16) # dilation rate of 2^4
        self.bn = BatchnormRelu(nOut)
        self.add = add

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        # reduce
        output1 = self.c1(input)
        # split and transform
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)
        # heirarchical fusion for de-gridding
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16
        #merge
        combine = torch.cat([d1, add1, add2, add3, add4], 1)
        if self.add:
            combine = input + combine
        output = self.bn(combine)
        return output

class AvgDownsampler(nn.Module):
    '''
    This class projects the input image to the same spatial dimensions as the feature map.
    For example, if the input image is 512 x512 x3 and spatial dimensions of feature map size are 56x56xF, then
    this class will generate an output of 56x56x3
    '''
    def __init__(self, samplingTimes):
        '''
        :param samplingTimes: The rate at which you want to down-sample the image
        '''
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, samplingTimes):
            #pyramid-based approach for down-sampling
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        '''
        :param input: Input RGB Image
        :return: down-sampled image (pyramid-based approach)
        '''
        for pool in self.pool:
            input = pool(input)
        return input


class Encoder(nn.Module):
    '''
    This class defines the ESPNet-C network in the paper
    '''
    def __init__(self, config):
        super().__init__()
        chanel_img = cfg.chanel_img
        model_cfg = cfg.sc_ch_dict[config] 
        self.level1 = ConvBatchnormRelu(chanel_img, model_cfg['chanels'][0], stride = 2)
        self.sample1 = AvgDownsampler(1)
        self.sample2 = AvgDownsampler(2)

        self.b1 = ConvBatchnormRelu(model_cfg['chanels'][0] + chanel_img,model_cfg['chanels'][1])
        self.level2_0 = StrideESP(model_cfg['chanels'][1], model_cfg['chanels'][2])

        self.level2 = nn.ModuleList()
        for i in range(0, model_cfg['p']):
            self.level2.append(DepthwiseESP(model_cfg['chanels'][2] , model_cfg['chanels'][2]))
        self.b2 = ConvBatchnormRelu(model_cfg['chanels'][3] + chanel_img,model_cfg['chanels'][3] + chanel_img)

        self.level3_0 = StrideESP(model_cfg['chanels'][3] + chanel_img, model_cfg['chanels'][3])
        self.level3 = nn.ModuleList()
        for i in range(0, model_cfg['q']):
            self.level3.append(DepthwiseESP(model_cfg['chanels'][3] , model_cfg['chanels'][3]))
        self.b3 = ConvBatchnormRelu(model_cfg['chanels'][4],model_cfg['chanels'][2])
        
    def forward(self, input):
        '''
        :param input: Receives the input RGB image
        :return: the transformed feature map with spatial dimensions 1/8th of the input image
        '''
        output0 = self.level1(input)
        inp1 = self.sample1(input)
        inp2 = self.sample2(input)
        output0_cat = self.b1(torch.cat([output0, inp1], 1))
        output1_0 = self.level2_0(output0_cat) # down-sampled
        
        for i, layer in enumerate(self.level2):
            if i==0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.b2(torch.cat([output1,  output1_0, inp2], 1))
        output2_0 = self.level3_0(output1_cat)
        for i, layer in enumerate(self.level3):
            if i==0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)
        output2_cat=torch.cat([output2_0, output2], 1)
        out_encoder = self.b3(output2_cat)
        
        return out_encoder,inp1,inp2

class UpSimpleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2, padding=0, output_padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-03)
        self.act = nn.PReLU(out_channels)

    def forward(self, input):
        output = self.deconv(input)
        output = self.bn(output)
        output = self.act(output)
        return output

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, sub_dim=3, last=False,kernel_size = 3):
        super(UpConvBlock, self).__init__()
        self.last=last
        self.up_conv = UpSimpleBlock(in_channels, out_channels)
        if not last:
            self.conv1 = ConvBatchnormRelu(out_channels+sub_dim,out_channels,kernel_size)
        self.conv2 = ConvBatchnormRelu(out_channels,out_channels,kernel_size)

    def forward(self, x, ori_img=None):
        x = self.up_conv(x)
        if not self.last:
            x = torch.cat([x, ori_img], dim=1)
            x = self.conv1(x)
        x = self.conv2(x)
        return x

class TwinLiteNetPlus(nn.Module,
                      PyTorchModelHubMixin,
                      repo_url="https://github.com/chequanghuy/TwinLiteNetPlus.git",
                      pipeline_tag="image-segmentation",
                      license="mit",
                      coders={
                            Namespace : (
                                lambda x: vars(x),
                                lambda data: Namespace(**data),
                            )
    }):
    '''
    This class defines the ESPNet network
    '''

    def __init__(self, args: Namespace = None):

        super().__init__()
        chanel_img = cfg.chanel_img
        model_cfg = cfg.sc_ch_dict[args.config] 
        self.encoder = Encoder(args.config)
        self.sigle_ll = False
        self.sigle_da = False

        self.caam = CAAM(feat_in=cfg.sc_ch_dict[args.config]['chanels'][2], num_classes=cfg.sc_ch_dict[args.config]['chanels'][2],bin_size =(2,4), norm_layer=nn.BatchNorm2d)
        self.conv_caam = ConvBatchnormRelu(cfg.sc_ch_dict[args.config]['chanels'][2],cfg.sc_ch_dict[args.config]['chanels'][1])

        self.up_1_da = UpConvBlock(cfg.sc_ch_dict[args.config]['chanels'][1],cfg.sc_ch_dict[args.config]['chanels'][0]) # out: Hx4, Wx4
        self.up_2_da = UpConvBlock(cfg.sc_ch_dict[args.config]['chanels'][0],8) #out: Hx2, Wx2
        self.out_da = UpConvBlock(8,2,last=True)  

        self.up_1_ll = UpConvBlock(cfg.sc_ch_dict[args.config]['chanels'][1],cfg.sc_ch_dict[args.config]['chanels'][0]) # out: Hx4, Wx4
        self.up_2_ll = UpConvBlock(cfg.sc_ch_dict[args.config]['chanels'][0],8) #out: Hx2, Wx2
        self.out_ll = UpConvBlock(8,2,last=True)


    def forward(self, input):
        '''
        :param input: RGB image
        :return: transformed feature map
        '''
        out_encoder,inp1,inp2=self.encoder(input)
        #visualize_feature_map_subset(out_encoder, "outencoder", 128)

        out_caam=self.caam(out_encoder)
        out_caam=self.conv_caam(out_caam)

        out_da=self.up_1_da(out_caam,inp2)
        out_da=self.up_2_da(out_da,inp1)
        out_da=self.out_da(out_da)

        out_ll=self.up_1_ll(out_caam,inp2)
        out_ll=self.up_2_ll(out_ll,inp1)
        out_ll=self.out_ll(out_ll)


        return out_da,out_ll

def netParams(model):
    return np.sum([np.prod(parameter.size()) for parameter in model.parameters()])
import time

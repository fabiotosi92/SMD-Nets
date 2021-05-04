from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from .submodule import *
from .utils import unet


class HSMNet(nn.Module):
    def __init__(self, opt, level=1, clean=1, phase='train'):
        super(HSMNet, self).__init__()

        self.opt = opt
        self.clean = clean
        self.level = level
        self.training = (phase == 'train')
        self.maxdisp = self.opt.maxdisp
        self.feature_extraction = unet()
        self.init_dim = 352 + self.maxdisp // 16

        # block 4
        self.decoder6 = decoderBlock(6, 32, 32, up=True, pool=True)
        if self.level > 2:
            self.decoder5 = decoderBlock(6, 32, 32, up=False, pool=True)
        else:
            self.decoder5 = decoderBlock(6, 32, 32, up=True, pool=True)
            if self.level > 1:
                self.decoder4 = decoderBlock(6, 32, 32, up=False)
            else:
                self.decoder4 = decoderBlock(6, 32, 32, up=True)
                self.decoder3 = decoderBlock(5, 16, 32, stride=(2, 1, 1), up=False, nstride=1)
        # reg
        self.disp_reg8 = disparityregression(self.maxdisp, 16)
        self.disp_reg16 = disparityregression(self.maxdisp, 16)
        self.disp_reg32 = disparityregression(self.maxdisp, 32)
        self.disp_reg64 = disparityregression(self.maxdisp, 64)

    def feature_vol(self, refimg_fea, targetimg_fea, maxdisp, leftview=True):
        '''
        diff feature volume
        '''
        width = refimg_fea.shape[-1]
        cost = Variable(
            torch.cuda.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1], maxdisp, refimg_fea.size()[2],
                                   refimg_fea.size()[3]).fill_(0.))
        for i in range(maxdisp):
            feata = refimg_fea[:, :, :, i:width]
            featb = targetimg_fea[:, :, :, :width - i]
            # concat
            if leftview:
                cost[:, :refimg_fea.size()[1], i, :, i:] = torch.abs(feata - featb)
            else:
                cost[:, :refimg_fea.size()[1], i, :, :width - i] = torch.abs(featb - feata)
        cost = cost.contiguous()
        return cost

    def forward(self, left, right):
        nsample = left.shape[0]
        conv4, conv3, conv2, conv1, enc0, enc1, enc2, enc3 = self.feature_extraction(torch.cat([left, right], 0))
        conv40, conv30, conv20, conv10, enc00, enc10, enc20, enc30 = conv4[:nsample], conv3[:nsample], conv2[:nsample], conv1[:nsample], enc0[:nsample], enc1[:nsample], enc2[:nsample], enc3[:nsample]
        conv41, conv31, conv21, conv11, enc01, enc11, enc21, enc31 = conv4[nsample:], conv3[nsample:], conv2[nsample:], conv1[nsample:], enc0[nsample:], enc1[nsample:], enc2[nsample:], enc3[nsample:]

        # simplified version of the original HSM
        feat3 = self.feature_vol(conv10, conv11, self.maxdisp // 8)
        feat3_2x, cost3 = self.decoder3(feat3)

        return [cost3, enc00, enc10, enc20, enc30]


    def _expand(self, pred):
        if len(pred.size()) == 3:
            pred = pred.unsqueeze(1)
        return pred

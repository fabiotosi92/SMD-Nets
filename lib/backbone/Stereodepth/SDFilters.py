from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from .SDLayers import *


def gaussian_blur(image, sigma):
    """
    Apply gaussian blur to an image.

    Inputs:
        image           a [Cx]HxW torch.Tensor
        sigma           the size of the blur

    Outputs:
        blurred         the blurred version of image
    """
    radius = 2 * sigma

    def get_blur_kernels(sigma, device):
        steps = torch.arange(2 * radius + 1) - radius
        weights = (-steps.float().pow_(2.0) / 2 / sigma).exp()
        weights = weights / weights.sum()
        weights = weights.to(device)
        return weights.view(1, 1, -1, 1).contiguous(), weights.view(1, 1, 1, -1).contiguous()

    kernel_H, kernel_W = get_blur_kernels(sigma, image.device)

    reshaped_image = image.view(-1, 1, *image.shape[-2:])

    blurred_H = torch.nn.functional.conv2d(reshaped_image, kernel_H, padding=(radius, 0))
    blurred = torch.nn.functional.conv2d(blurred_H, kernel_W, padding=(0, radius))
    return blurred.view(*image.shape)

def local_contrast_normalization(image, radius=25):
    """
    Normalizes an image, by -- for each pixel -- subtracting the mean and
    dividing by the variance of its neighbourhood
    """
    d = 2*radius + 1
    def get_boxfilter_weights(d, device):
        return torch.ones((1,1,d,d), requires_grad=False).to(device) / d**2

    boxfilter_weights = get_boxfilter_weights(d, image.device)
    reshaped_image = image.view(-1, 1, *image.shape[-2:])
    avgs = torch.nn.functional.conv2d(reshaped_image, boxfilter_weights, padding=radius)
    avg2s = torch.nn.functional.conv2d(reshaped_image.pow(2), boxfilter_weights, padding=radius)
    stds = (avg2s - avgs**2).sqrt_() * d**2 / (d**2 - 1)
    lcned = torch.where(
        stds == 0,
        torch.tensor(0.).to(image.device),
        (reshaped_image - avgs) / stds
    )

    return lcned.view(image.shape), avgs.view(image.shape), stds.view(image.shape)

class SDFilter(nn.Module):
    def __init__(self, opt):
        super(SDFilter, self).__init__()

        self.opt = opt

        self.maxdisp = self.opt.maxdisp
        self.num_input_images = 2 #2 if self.opt.num_views else 1
        self.channels = 3 if self.opt.mode == "passive" else 1
        self.pretrained = self.opt.imagenet_pt
        self.init_dim = 240

        print("=> number of input images: {}".format(self.num_input_images))
        print("=> pretrained on ImageNet: {}".format(self.pretrained))

        self.encoder = ResnetEncoder(num_layers=18, pretrained=self.pretrained, num_input_images=self.num_input_images, channels=self.channels)
        num_ch_enc = self.encoder.num_ch_enc
        self.decoder = DepthDecoder(num_ch_enc, scales=range(4))

    def forward(self, left, right):
        _, _, h, w = left.shape
        features = self.encoder(left, right)
        return self.decoder(features, (h, w))


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features, size):
        self.outputs = {}
        self.disparities = {}
        h, w = size

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)

            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.convs[("dispconv", i)](x)

            self.outputs[("out", i)] = x

        out3 = self.outputs[("out", 3)]
        out2 = self.outputs[("out", 2)]
        out1 = self.outputs[("out", 1)]
        out0 = self.outputs[("out", 0)]

        return [out3, out2, out1, out0]


class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def __init__(self, block, layers, num_classes=1000, num_input_images=1, channels=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0]) #64
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) #128
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2) #256
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2) #512

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1, channels=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images, channels=channels)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """

    def __init__(self, num_layers, pretrained, num_input_images=1, channels=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.channels = channels

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        #if num_input_images > 1:
        self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images, channels)
        #else:
        #    self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, left, right):
        self.features = []
        if self.channels == 2:
            IR_cleaned = local_contrast_normalization(gaussian_blur(left, sigma=1), radius=25)[0]
            x = torch.cat((IR_cleaned, right), 1)
        else:
            x = torch.cat((left/255., right/255.), 1)
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features

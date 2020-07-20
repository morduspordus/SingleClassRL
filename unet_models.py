from __future__ import print_function, division

import torch
import torch.nn as nn
from torchvision import models
import segmentation_models_pytorch as smp
import torch.nn.functional as F


class MobileNetV2_Layers(nn.Module):
    """
        Returns intermediate and final layers as features. Concatenates features at the same spatial resolution.
    """

    def __init__(self):
        super(MobileNetV2_Layers, self).__init__()
        features = list(models.mobilenet_v2(pretrained=True).features)
        self.features = nn.ModuleList(features).eval()

    def forward(self, x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            results.append(x)

        combined_results = []
        out = torch.cat((results[0], results[1]), dim=1)
        combined_results.append(out)

        out = torch.cat((results[2], results[3]), dim=1)
        combined_results.append(out)

        out = torch.cat((results[4], results[5], results[6]), dim=1)
        combined_results.append(out)

        out = torch.cat((results[7], results[8], results[9], results[10], results[11], results[12], results[13]), dim=1)
        combined_results.append(out)

        out = torch.cat((results[14], results[15], results[16], results[17]), dim=1)
        combined_results.append(out)

        return(combined_results)


class pix2pix_upsample(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=2, stride=2, padding=0):
        super(pix2pix_upsample, self).__init__()
        self.upsample = nn.Sequential(
                             nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding),
                             nn.BatchNorm2d(out_ch),
                             nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.upsample(x)
        return x


class Unet_Main(nn.Module):

    def __init__(self, output_channels, final_activation, num_final_features, upsample_param):

        super(Unet_Main, self).__init__()

        self.num_classes = output_channels
        self.final_activation = final_activation
        self.num_final_features = num_final_features
        self.upsample_param = upsample_param

        self.upstack1 = pix2pix_upsample(self.upsample_param[0][0], self.upsample_param[0][1])
        self.upstack2 = pix2pix_upsample(self.upsample_param[1][0], self.upsample_param[1][1])
        self.upstack3 = pix2pix_upsample(self.upsample_param[2][0], self.upsample_param[2][1])
        self.upstack4 = pix2pix_upsample(self.upsample_param[3][0], self.upsample_param[3][1])

        self.up_stack = [self.upstack1, self.upstack2, self.upstack3, self.upstack4]

        upsample_layer = nn.Sequential(
            nn.ConvTranspose2d(self.upsample_param[4], self.num_final_features, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(self.num_final_features),
            nn.ReLU(inplace=True)
        )

        final_layer = nn.Sequential(
            nn.Conv2d(self.num_final_features, output_channels, kernel_size=3, stride=1, padding=1),
        )

        feature_list = [upsample_layer, final_layer]
        self.features = nn.Sequential(*feature_list)

    def freeze_weights(self):
        for param in self.down_stack.parameters():
            param.requires_grad = False

    def forward(self, x):
      skips = self.down_stack(x)

      x = skips[-1]
      skips = reversed(skips[:-1])

      for up, skip in zip(self.up_stack, skips):
          x = up(x)
          x = torch.cat((x, skip), dim=1)

      x = self.features(x)

      if self.final_activation == 'softmax':
          x = torch.softmax(x, dim=1)

      return x


class Unet_MobileNetV2(Unet_Main):
    """
        Unet based on MobileNetV2 features
    """
    def __init__(self, output_channels, use_fixed_features, final_activation='softmax', num_final_features=64):
        upsample_param = [(800, 512), (1056, 256), (352, 128), (176, 64), 112]
        super(Unet_MobileNetV2, self).__init__(output_channels, final_activation, num_final_features, upsample_param)
        self.down_stack = MobileNetV2_Layers()

        if use_fixed_features:
            self.freeze_weights()


class Unet_se_resnext50_32x4d(Unet_Main):
    """
        Unet based on se_resnext50_32x4d features
    """

    def __init__(self, output_channels, use_fixed_features, final_activation='softmax', num_final_features=64):
        upsample_param = [(2048, 512), (1536, 256), (768, 128), (384, 64), 128]
        super(Unet_se_resnext50_32x4d, self).__init__(output_channels, final_activation, num_final_features, upsample_param)

        self.main_model = smp.Unet('se_resnext50_32x4d', encoder_weights='imagenet')
        self.down_stack = self.main_model.encoder

        if use_fixed_features:
            self.freeze_weights()


import torch
import torch.nn as nn
import torch.nn.functional as F


class CoarseSaliencyModel(nn.Module):
    def __init__(self, input_shape, pretrained, branch=''):
        super(CoarseSaliencyModel, self).__init__()
        self.c, self.fr, self.h, self.w = input_shape
        assert self.h % 8 == 0 and self.w % 8 == 0, 'Input shape should be divisible by 8.'

        # self.conv1 = nn.Conv3d(64, 3, 3, 3, padding=1)
        # self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2), padding=0)
        # self.conv2 = nn.Conv3d(128, 3, 3, 3, padding=1)
        # self.pool2 = nn.MaxPool3d((2, 2, 2), (2, 2, 2), padding=0)
        # self.conv3a = nn.Conv3d(256, 3, 3, 3, padding=1)
        # self.conv3b = nn.Conv3d(256, 3, 3, 3, padding=1)
        # self.pool3 = nn.MaxPool3d((2, 2, 2), (2, 2, 2), padding=0)
        # self.conv4a = nn.Conv3d(512, 3, 3, 3, padding=1)
        # self.conv4b = nn.Conv3d(512, 3, 3, 3, padding=1)
        # self.pool4 = nn.MaxPool3d((4, 1, 1), (4, 1, 1), padding=0)
        self.conv1 = nn.Conv3d(in_channels=self.c, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                               padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv2 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                               padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv3a = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv4a = nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(in_channels=512, out_channels=512, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(4, 1, 1), stride=(4, 1, 1))
        self.bilinear = nn.Upsample(scale_factor=8, mode='bilinear')

        if pretrained:
            raise NotImplementedError('Pretrained weights not supported yet.')

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3a(x))
        x = F.relu(self.conv3b(x))
        x = self.pool3(x)
        x = F.relu(self.conv4a(x))
        x = F.relu(self.conv4b(x))
        x = self.pool4(x)

        x = x.view(x.shape[0], 512, self.h // 8, self.w // 8)
        x = self.bilinear(x)
        return x  # F.interpolate(x, size=(self.h, self.w), mode='bilinear')


# class SaliencyBranch(nn.Module):
#     def __init__(self, input_shape, c3d_pretrained, branch=''):
#         super(SaliencyBranch, self).__init__()
#         c, fr, h, w = input_shape
#
#         self.coarse_predictor = CoarseSaliencyModel(input_shape=(c, fr, h // 4, w // 4), pretrained=c3d_pretrained,
#                                                     branch=branch)
#         self.ff_conv1 = nn.Conv2d(c, 32, kernel_size=(3, 3), padding=(1, 1))
#         self.ff_conv2 = nn.Conv2d(32, 16, kernel_size=(3, 3), padding=(1, 1))
#         self.ff_conv3 = nn.Conv2d(16, 8, kernel_size=(3, 3), padding=(1, 1))
#         self.ff_conv4 = nn.Conv2d(8, 1, kernel_size=(3, 3), padding=(1, 1))
#         self.crop_conv = nn.Conv2d(1, 1, kernel_size=(3, 3), padding=(1, 1))
#
#     def forward(self, ff_in, small_in, crop_in):
#         ff_last_frame = ff_in.view(ff_in.shape[0], -1, ff_in.shape[2], ff_in.shape[3])
#
#         coarse_h = self.coarse_predictor(small_in)
#         print(coarse_h.shape)
#         coarse_h = F.relu(self.crop_conv(coarse_h))
#         coarse_h = coarse_h.repeat(4, 4, 1, 1)
#         fine_h = torch.cat((coarse_h, ff_last_frame), dim=1)
#         fine_h = F.relu(self.ff_conv1(fine_h))
#         fine_h = F.relu(self.ff_conv2(fine_h))
#         fine_h = F.relu(self.ff_conv3(fine_h))
#         fine_h = self.ff_conv4(fine_h)
#         fine_out = F.relu(fine_h)
#         crop_h = self.coarse_predictor(crop_in)
#         crop_out = F.relu(self.crop_conv(crop_h))
#
#         return fine_out, crop_out


class SaliencyBranch(nn.Module):
    def __init__(self, input_shape, c3d_pretrained, branch):
        super(SaliencyBranch, self).__init__()

        self.c, self.fr, self.h, self.w = input_shape
        self.coarse_predictor = CoarseSaliencyModel(input_shape=(self.c, self.fr, self.h // 4, self.w // 4),
                                                    pretrained=c3d_pretrained, branch=branch)
        self.conv0 = nn.Conv2d(512, 1, (3, 3), padding=1)
        self.conv1 = nn.Conv2d(4, 1, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(1, 32, (3, 3), padding=1)
        self.conv3 = nn.Conv2d(32, 16, (3, 3), padding=1)
        self.conv4 = nn.Conv2d(16, 8, (3, 3), padding=1)
        self.upsample = nn.Upsample(scale_factor=4)
        self.conv5 = nn.Conv2d(512, 1, (3, 3), padding=1)

    def forward(self, ff_in, small_in, crop_in):
        # c, fr, h, w = ff_in.shape
        print('ff_in', ff_in.shape)
        ff_last_frame = ff_in.view(1, self.c, self.h, self.w)  # remove singleton dimension
        print('ff_last', ff_last_frame.shape)
        coarse_h = self.coarse_predictor(small_in)
        print('coarse_h', coarse_h.shape)
        coarse_h = F.relu(self.conv0(coarse_h))
        coarse_h = self.upsample(coarse_h)

        fine_h = torch.cat((coarse_h, ff_last_frame), dim=1)
        print(fine_h.shape)
        fine_h = F.leaky_relu(self.conv1(fine_h), negative_slope=0.001)
        fine_h = F.leaky_relu(self.conv2(fine_h), negative_slope=0.001)
        fine_h = F.leaky_relu(self.conv3(fine_h), negative_slope=0.001)
        fine_h = self.conv4(fine_h)
        fine_out = F.relu(fine_h)

        crop_h = self.coarse_predictor(crop_in)
        print('crop_h', crop_h.shape)
        crop_out = F.relu(self.conv5(crop_h))

        return fine_out, crop_out

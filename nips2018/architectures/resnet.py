import numpy as np
import torch
from attorch.layers import Elu1, BiasBatchNorm2d, Conv2dPad, SpatialXFeatureLinear, WidthXHeightXFeatureLinear
from torch import nn as nn
from torch.autograd import Variable
from torch.nn.init import xavier_normal


class ResNet(nn.Module):
    def __init__(self, neurons, in_shape, components=1, nonlinearity=None, ker0=13, kerh=3, normalize=False,
                 positive=True, channels=32):
        super().__init__()
        if nonlinearity == 'relu':
            self.f = nn.ReLU()
        else:
            self.f = Elu1()

        self.conv0 = nn.Conv2d(1, channels, ker0)
        self.bn0 = BiasBatchNorm2d(channels, momentum=.99)

        self.conv1 = Conv2dPad(channels, channels, kerh, pad=kerh // 2)
        self.bn1 = BiasBatchNorm2d(channels, momentum=.99)

        self.conv2 = Conv2dPad(channels, channels, kerh, pad=kerh // 2)
        self.bn2 = BiasBatchNorm2d(channels, momentum=.99)

        tmp = Variable(torch.from_numpy(np.random.randn(1, *in_shape[1:]).astype(np.float32)))

        tmp = self.conv2(self.conv1(self.conv0(tmp)))
        nout = tmp.size()[1:]

        if components == 0:
            self.ro0 = SpatialXFeatureLinear(nout, neurons, normalize=normalize, positive=positive)
            self.ro1 = SpatialXFeatureLinear(nout, neurons, normalize=normalize, positive=positive,
                                             spatial=self.ro0.spatial)
            self.ro2 = SpatialXFeatureLinear(nout, neurons, normalize=normalize, positive=positive,
                                             spatial=self.ro0.spatial)
        else:
            self.ro0 = WidthXHeightXFeatureLinear(nout, neurons, normalize=normalize, positive=positive,
                                                  components=components)
            self.ro1 = WidthXHeightXFeatureLinear(nout, neurons, normalize=normalize, positive=positive,
                                                  width=self.ro0.width, height=self.ro0.height, components=components)
            self.ro2 = WidthXHeightXFeatureLinear(nout, neurons, normalize=normalize, positive=positive,
                                                  width=self.ro0.width, height=self.ro0.height, components=components)

    @staticmethod
    def init_conv(m):
        if isinstance(m, (nn.Conv2d, Conv2dPad)):
            xavier_normal(m.weight.data)
            m.bias.data.fill_(0)

    def initialize_readout(self, y_train):
        self.apply(self.init_conv)
        self.ro0.initialize_core(init_noise=1e-3)
        self.ro0.bias.data = y_train.mean(0).squeeze() - 1
        self.ro1.bias.data.fill_(0)
        self.ro2.bias.data.fill_(0)

    def forward(self, x):
        y0 = self.bn0(self.conv0(x))
        z0 = self.ro0(y0)

        y1 = self.bn1(self.conv1(self.f(y0)))
        z1 = self.ro1(y1)

        y2 = self.bn2(self.conv2(self.f(y1)))
        z2 = self.ro1(y2)

        return self.f(z0 + z1 + z2)
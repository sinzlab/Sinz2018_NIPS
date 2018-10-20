import numpy as np
import torch
from attorch.layers import WidthXHeightXFeatureLinear, SpatialXFeatureLinear, Elu1
from attorch.module import ModuleDict
from torch import nn as nn
from torch.autograd import Variable
from torchvision import models


class VGGCaMP(nn.Module):
    def __init__(self, neurons, in_shape, chop_at=8, components=1, normalize=True, positive=True):
        super().__init__()
        vgg = models.vgg19(pretrained=True).eval()
        self.core = nn.Sequential(*[layer for i, layer in enumerate(vgg.features) if i <= chop_at])
        tmp = Variable(torch.from_numpy(np.random.randn(1, 3, *in_shape[2:]).astype(np.float32)))
        nout = self.core(tmp).size()[1:]
        if components > 0:
            self.readout = ModuleDict({
                k: WidthXHeightXFeatureLinear(nout, neur, components=components,
                                              normalize=normalize, positive=positive)
                for k, neur in neurons.items()
            })
        else:
            self.readout = ModuleDict({
                k: SpatialXFeatureLinear(nout, neur, normalize=normalize, positive=positive)
                for k, neur in neurons.items()})
        self.nonlinearity = Elu1()

    def core_requires_grad(self, state):
        for p in self.core.parameters():
            p.core_requires_grad = state

    def initialize_readout(self, mu_dict):
        for k, mu in mu_dict.items():
            self.readout[k].initialize(init_noise=1e-3)
            self.readout[k].bias.data = mu.squeeze() - 1

    def forward(self, x, readout_key):
        N, _, w, h = x.size()
        x = x.expand(N, 3, w, h)
        x = self.core(x)
        if isinstance(readout_key, str):
            x = self.readout[readout_key](x)
            ret = self.nonlinearity(x)
        else:
            ret = tuple(self.nonlinearity(self.readout[key](x)) for key in readout_key)
        return ret
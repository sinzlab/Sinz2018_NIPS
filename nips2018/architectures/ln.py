import numpy as np
from attorch.layers import Elu1, SpatialXFeatureLinear
from attorch.module import ModuleDict
from torch import nn as nn
from torch.autograd import Variable
import torch
from torch.nn.init import xavier_normal


class LinearNonlinear(nn.Module):
    def __init__(self, neurons, in_shape):
        super().__init__()
        self.n_neurons = neurons
        self.in_shape = in_shape[1:]
        self.readout = ModuleDict({k: nn.Linear(int(np.prod(in_shape[1:])), neur) for k, neur in neurons.items()})
        self.nonlinearity = Elu1()

    def forward(self, x, readout_key):
        x = x.view(x.size(0), -1)
        if isinstance(readout_key, str):
            x = self.readout[readout_key](x)
            ret = self.nonlinearity(x)
        else:
            ret = tuple(self.nonlinearity(self.readout[key](x)) for key in readout_key)
        return ret

    def initialize_readout(self, mu_dict, noise_std):
        print('Initializing readout', flush=True)
        for k, mu in mu_dict.items():
            self.readout[k].bias.data = mu.squeeze() - 1
            self.readout[k].weight.data.normal_(std=noise_std)
            # self.readout[k].weight.data.fill_(0.)

    def weight(self, readout_key):
        return self.readout[readout_key].weight.view(self.n_neurons[readout_key], 1, *self.in_shape[-2:])

class FactorizedLN(nn.Module):
    def __init__(self, neurons, in_shape):
        super().__init__()
        self.n_neurons = neurons
        self.in_shape = in_shape[1:]
        # self.pool = nn.Conv2d(in_shape[1], in_shape[1], 2, 2, groups=in_shape[1]) # kernel size and stride = 2 for downsampling
        self.pool = nn.AvgPool2d(2,2)
        self.bn = nn.BatchNorm2d(in_shape[1], momentum=0.99)
        tmp = Variable(torch.from_numpy(np.random.randn(1, *in_shape[1:]).astype(np.float32)))
        nout = self.pool(tmp).size()[1:]
        self.readout = ModuleDict({k: SpatialXFeatureLinear(nout, neur, normalize=True, positive=False) for k, neur in neurons.items()})
        self.nonlinearity = Elu1()

    def forward(self, x, readout_key):
        x = self.bn(self.pool(x))

        if isinstance(readout_key, str):
            x = self.readout[readout_key](x)
            ret = self.nonlinearity(x)
        else:
            ret = tuple(self.nonlinearity(self.readout[key](x)) for key in readout_key)
        return ret

    def initialize(self, mu_dict, noise_std):
        print('Initializing readout', flush=True)
        # xavier_normal(self.pool.weight.data)
        # self.pool.bias.data.fill_(0)
        for k, mu in mu_dict.items():
            self.readout[k].initialize(init_noise=noise_std)
            self.readout[k].bias.data = mu.squeeze() - 1

    def weight(self, readout_key):
        return self.readout[readout_key].normalized_spatial
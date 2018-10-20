from collections import OrderedDict

from torch import nn
from torch.nn.init import xavier_normal
import torch
from torch.nn import functional as F
import torch.nn.init as init
from attorch.regularizers import LaplaceL23d, LaplaceL2
from torch.autograd import Variable
from attorch.module import ModuleDict
from ..utils.logging import Messager


class Shifter(Messager):
    def __repr__(self):
        s = super().__repr__()
        s += ' [{} regularizers: '.format(self.__class__.__name__)
        ret = []
        for attr in filter(lambda x: 'gamma' in x, dir(self)):
            ret.append('{} = {}'.format(attr, getattr(self, attr)))
        return s + '|'.join(ret) + ']\n'


class GRU(nn.Module, Messager):
    def __init__(self, input_features=2, hidden_channels=2, bias=True, **kwargs):
        super().__init__()
        self.msg('Ignoring input', kwargs, 'when creating', self.__class__.__name__, depth=1)
        self.gru = nn.GRUCell(input_features, hidden_channels, bias=bias)
        self.hidden_states = hidden_channels

    def regularizer(self):
        return self.gru.bias_hh.abs().mean()

    def initialize(self, stdv=1e-2):
        # self.msg('Initializing GRU shifter with 0')
        # for weight in self.parameters():
        #     weight.data.fill_(0)

        self.msg('Initializing GRU shifter with stdv', stdv)
        for weight in self.parameters():
            # xavier_normal(weight.data, gain=gain)
            weight.data.uniform_(-stdv, stdv)

    def initialize_state(self, batch_size, hidden_size, cuda):
        state_size = [batch_size, hidden_size]
        state = Variable(torch.zeros(*state_size))
        if cuda:
            state = state.cuda()
        return state

    def forward(self, input):
        N, T, f = input.size()
        states = []

        hidden = self.initialize_state(N, self.hidden_states, input.is_cuda)

        x = input.transpose(0, 1)

        for t in range(T):
            hidden = self.gru(x[t, ...], hidden)
            states.append(hidden)
        states = torch.stack(states).transpose(0, 1)
        return states


class MLP(nn.Module, Messager):
    def __init__(self, input_features=2, hidden_channels=10, shift_layers=1, **kwargs):
        super().__init__()
        self.msg('Ignoring input', kwargs, 'when creating', self.__class__.__name__, depth=1)

        feat = []
        if shift_layers > 1:
            feat = [nn.Linear(input_features, hidden_channels), nn.Tanh()]
        else:
            hidden_channels = input_features

        for _ in range(shift_layers - 2):
            feat.extend([nn.Linear(hidden_channels, hidden_channels), nn.Tanh()])

        feat.extend([nn.Linear(hidden_channels, 2), nn.Tanh()])
        self.mlp = nn.Sequential(*feat)

    def regularizer(self):
        return 0

    def initialize(self):
        for linear_layer in [p for p in self.parameters() if isinstance(p, nn.Linear)]:
            xavier_normal(linear_layer.weight)

    def forward(self, input):
        return self.mlp(input)


class MLPShifter(Shifter, ModuleDict, Messager):
    def __init__(self, data_keys, input_channels=2, hidden_channels_shifter=2,
                 shift_layers=1, gamma_shifter=0, **kwargs):
        self.msg('Ignoring input', kwargs, 'when creating', self.__class__.__name__, )
        super().__init__()
        self.gamma_shifter = gamma_shifter
        for k in data_keys:
            self.add_module(k, MLP(input_channels, hidden_channels_shifter, shift_layers))

    def initialize(self):
        for linear_layer in [p for p in self.parameters() if isinstance(p, nn.Linear)]:
            xavier_normal(linear_layer.weight)

    def regularizer(self, data_key):
        return self[data_key].regularizer() * self.gamma_shifter


class GRUShifter(Shifter, ModuleDict, Messager):
    def __init__(self, data_keys, input_channels=2, hidden_channels=2,
                 bias=True, gamma_shifter=0, init_noise=1e-3, **kwargs):
        self.msg('Ignoring input', kwargs, 'when creating', self.__class__.__name__, )
        super().__init__()
        self.gamma_shifter = gamma_shifter
        self.init_noise = init_noise
        for k in data_keys:
            self.add_module(k, GRU(input_channels, hidden_channels, bias))

    def initialize(self):
        for k in self:
            self[k].initialize(stdv=self.init_noise)

    def regularizer(self, data_key):
        return self[data_key].regularizer() * self.gamma_shifter


class StaticAffineShifter(Shifter, ModuleDict, Messager):
    def __init__(self, data_keys, input_channels, bias=True, gamma_shifter=0, **kwargs):
        self.msg('Ignoring input', kwargs, 'when creating', self.__class__.__name__)
        super().__init__()
        self.gamma_shifter = gamma_shifter
        for k in data_keys:
            self.add_module(k, StaticAffine(input_channels, 2, bias=bias))

    def initialize(self, bias=None):
        self.msg('Initializing affine weights', depth=0)
        for k in self:
            if bias is not None:
                self[k].initialize(bias=bias[k])
            else:
                self[k].initialize()

    def regularizer(self, data_key):
        return self[data_key].weight.pow(2).mean() * self.gamma_shifter


class StaticAffine(nn.Linear, Messager):
    def __init__(self, input_channels, output_channels, bias=True, **kwargs):
        self.msg('Ignoring input', kwargs, 'when creating', self.__class__.__name__)
        super().__init__(input_channels, output_channels, bias=bias)

    def forward(self, input):
        N, T, f = input.size()
        x = input.view(N * T, f)
        x = super().forward(x)
        return F.tanh(x.view(N, T, -1))

    def initialize(self, bias=None):
        self.weight.data.normal_(0, 1e-6)
        if self.bias is not None:
            if bias is not None:
                self.msg('Setting bias to predefined value', bias.numpy())
                self.bias.data = bias
            else:
                self.bias.data.normal_(0, 1e-6)


class StaticAffine2dShifter(Shifter, ModuleDict, Messager):
    def __init__(self, data_keys, input_channels, bias=True, gamma_shifter=0, **kwargs):
        self.msg('Ignoring input', kwargs, 'when creating', self.__class__.__name__)
        super().__init__()
        self.gamma_shifter = gamma_shifter
        for k in data_keys:
            self.add_module(k, StaticAffine2d(input_channels, 2, bias=bias))

    def initialize(self):
        self.msg('Initializing affine weights', depth=0)
        for k in self:
            self[k].initialize()

    def regularizer(self, data_key):
        return self[data_key].weight.pow(2).mean() * self.gamma_shifter


class StaticAffine2d(nn.Linear, Messager):
    def __init__(self, input_channels, output_channels, bias=True, **kwargs):
        self.msg('Ignoring input', kwargs, 'when creating', self.__class__.__name__)
        super().__init__(input_channels, output_channels, bias=bias)

    def forward(self, x):
        x = super().forward(x)
        return F.tanh(x)

    def initialize(self):
        self.weight.data.normal_(0, 1e-6)
        if self.bias is not None:
            self.bias.data.normal_(0, 1e-6)



class SharedGRUShifter(Shifter, nn.Module, Messager):
    def __init__(self, data_keys, input_channels=2, hidden_channels=2,
                 bias=True, gamma_shifter=0, init_noise=1e-3, **kwargs):
        self.msg('Ignoring input', kwargs, 'when creating', self.__class__.__name__, )
        super().__init__()
        self.gamma_shifter = gamma_shifter
        self.data_keys = data_keys
        self.init_noise = init_noise
        self.shifter = GRU(input_channels, hidden_channels, bias)

    def __getitem__(self, item):
        if item in self.data_keys:
            return self.shifter
        else:
            return super().__getitem__(item)

    def initialize(self):
        self.shifter.initialize(stdv=self.init_noise)

    def regularizer(self, data_key):
        return self.shifter.regularizer() * self.gamma_shifter


def NoShifter(*args, **kwargs):
    """
    Dummy function to create an object that returns None
    Args:
        *args:   will be ignored
        *kwargs: will be ignored

    Returns:
        None
    """
    return None

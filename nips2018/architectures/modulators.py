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


class GateGRU(nn.Module, Messager):
    def __init__(self, neurons, input_channels=3, hidden_channels=5, bias=True, offset=0, **kwargs):
        super().__init__()
        self.msg('Ignoring input', kwargs, 'when creating', self.__class__.__name__, depth=1)
        self.gru = nn.GRUCell(input_channels, hidden_channels, bias=bias)
        self.linear = nn.Linear(hidden_channels, neurons)
        self.hidden_states = hidden_channels
        self.offset = offset

    def regularizer(self, subs_idx=None):
        subs_idx = subs_idx if subs_idx is not None else slice(None)
        return self.linear.weight[subs_idx, :].abs().mean()

    def initialize_state(self, batch_size, hidden_size, cuda):
        state_size = [batch_size, hidden_size]
        state = Variable(torch.zeros(*state_size))
        if cuda:
            state = state.cuda()
        return state

    def forward(self, input, readoutput=None, subs_idx=None):
        N, T, f = input.size()
        states = []

        hidden = self.initialize_state(N, self.hidden_states, input.is_cuda)
        x = input.transpose(0, 1)
        for t in range(T):
            hidden = self.gru(x[t, ...], hidden)
            states.append(self.linear(hidden))
        if readoutput is None:
            self.msg('Nothing to modulate. Returning modulation only')
            return torch.exp(torch.stack(states, 1))
        else:
            lag = T - readoutput.size(1)
        states = torch.exp(torch.stack(states, 1)[:, lag:, :])
        if subs_idx is not None:
            states = states[..., subs_idx]
        return readoutput * (states + self.offset)


class MLP(nn.Module, Messager):
    def __init__(self, neurons, input_channels=3, hidden_channels=10, layers=2, **kwargs):
        super().__init__()
        self.msg('Ignoring input', kwargs, 'when creating', self.__class__.__name__, depth=1)
        feat = [nn.Linear(input_channels, hidden_channels), nn.ReLU()]
        for _ in range(layers - 1):
            feat.extend([nn.Linear(hidden_channels, hidden_channels), nn.ReLU()])
        self.mlp = nn.Sequential(*feat)
        self.linear = nn.Linear(hidden_channels, neurons)

    def regularizer(self):
        return self.linear.weight.abs().mean()

    def initialize(self):
        for linear_layer in [p for p in self.parameters() if isinstance(p, nn.Linear)]:
            xavier_normal(linear_layer.weight)

    def forward(self, input, readoutput=None):
        mod = torch.exp(self.linear(self.mlp(input)))

        if readoutput is None:
            self.msg('Nothing to modulate. Returning modulation only')
            return mod
        else:
            return readoutput * mod


class GRUModulator(ModuleDict, Messager):
    _base_modulator = None

    def __init__(self, n_neurons, input_channels=3, hidden_channels=5,
                 bias=True, gamma_modulator=0, offset=0, **kwargs):
        self.msg('Ignoring input', kwargs, 'when creating', self.__class__.__name__)
        super().__init__()
        self.gamma_modulator = gamma_modulator
        for k, n in n_neurons.items():
            self.add_module(k, self._base_modulator(n, input_channels, hidden_channels, bias, offset))

    def initialize(self):
        self.msg('Initializing', self.__class__.__name__, flush=True)
        for k, mu in self.items():
            self[k].gru.reset_parameters()

    def regularizer(self, data_key, subs_idx=None):
        return self[data_key].regularizer(subs_idx=subs_idx) * self.gamma_modulator


class StaticModulator(ModuleDict, Messager):
    _base_modulator = None

    def __init__(self, n_neurons, input_channels=3, hidden_channels=5,
                 layers=2, gamma_modulator=0, **kwargs):
        self.msg('Ignoring input', kwargs, 'when creating', self.__class__.__name__)
        super().__init__()
        self.gamma_modulator = gamma_modulator
        for k, n in n_neurons.items():
            if isinstance(input_channels, OrderedDict):
                ic = input_channels[k]
            else:
                ic = input_channels
            self.add_module(k, self._base_modulator(n, ic, hidden_channels, layers=layers))

    def initialize(self):
        self.msg('Initializing', self.__class__.__name__, flush=True)
        for k, mu in self.items():
            self[k].initialize()

    def regularizer(self, data_key):
        return self[data_key].regularizer() * self.gamma_modulator


class GateGRUModulator(GRUModulator):
    _base_modulator = GateGRU


class MLPModulator(StaticModulator):
    _base_modulator = MLP


class FeedbackMLPModulator(StaticModulator):
    _base_modulator = MLP


def NoModulator(*args, **kwargs):
    """
    Dummy function to create an object that returns None
    Args:
        *args:   will be ignored
        *kwargs: will be ignored

    Returns:
        None
    """
    return None

from collections import OrderedDict, defaultdict
from itertools import count
from warnings import warn

from torch import nn
from torch.nn.init import xavier_normal
import torch
from torch.nn import functional as F, Parameter
import torch.nn.init as init
from attorch.regularizers import LaplaceL23d, LaplaceL2
from torch.autograd import Variable
from attorch.layers import ExtendedConv2d
from ..utils.logging import Messager

try:
    from netgard.flat_cajal import CajalUnit
except:
    warn("Could not import CajalUnit. You won't be able to use the NetGardCore")
from attorch.layers import BiasBatchNorm2d, Elu1
from attorch.module import ModuleDict


class Core(Messager):
    def initialize(self):
        self.msg('Not initializing anything')

    def __repr__(self):
        s = super().__repr__()
        s += ' [{} regularizers: '.format(self.__class__.__name__)
        ret = []
        for attr in filter(lambda x: 'gamma' in x or 'skip' in x, dir(self)):
            ret.append('{} = {}'.format(attr, getattr(self, attr)))
        return s + '|'.join(ret) + ']\n'


class Core2d(Core):
    def initialize(self, cuda=False):
        self.apply(self.init_conv)
        if cuda:
            self = self.cuda()

    @staticmethod
    def init_conv(m):
        if isinstance(m, nn.Conv2d):
            xavier_normal(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0)


class Core3d(Core):
    def initialize(self, cuda=False):
        self.apply(self.init_conv)
        if cuda:
            self = self.cuda()

    @staticmethod
    def init_conv(m):
        if isinstance(m, nn.Conv3d):
            xavier_normal(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0)


# ---------------------- Identity Core ---------------------------
class IdentityCore(nn.Module, Core2d):
    def __init__(self, **kwargs):
        self.msg('Ignoring input', kwargs, 'when creating', self.__class__.__name__)
        super().__init__()

    def forward(self, x):
        return x

    def regularizer(self):
        return 0.0


# ---------------------- NetGard Cores ----------------------------
class NetGardCore(nn.Module, Messager):
    multiple_outputs = True  # a flag to indicate that there are multiple readout points

    def __init__(self, input_channels, in_shape, timesteps=10, ro_method='last',
                 name_map=None, gamma_input=0.0, gamma_hidden=0.0, momentum=0.99, eps=1e-8, unit_config=None):
        super().__init__()
        self.in_shape = in_shape
        self.ro_method = ro_method
        self.timesteps = timesteps
        self.momentum = momentum

        self.gamma_input = gamma_input
        self.gamma_hidden = gamma_hidden

        self.laplace_reg = LaplaceL2()
        self.eps = eps

        if unit_config is None:
            unit_config = {}
        self.unit = CajalUnit(in_shape, **unit_config)

        self.area_lut = {
            'V1': 0,
            'LM': 1
        }

        self.cell_lut = {
            'L2/3': 'pyramidal',
            'L4': 'stellate'
        }

        if name_map is None:
            name_map = {}
        self.name_map = name_map
        self.batch_norm = None

    def initialize(self):
        self.unit.initialize()

    def regularizer(self):
        val = {'sum': 0.0}

        def collect(m):
            if m is not self.unit.stellate0_input and (isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d)):
                val['sum'] = val['sum'] + (
                        m.weight.pow(2).sum(3, keepdim=True).sum(2, keepdim=True) + self.eps).sqrt().mean()

        self.unit.apply(collect)

        return val['sum'] * self.gamma_hidden + self.laplace_reg(self.unit.stellate0_input.weight) * self.gamma_input

    def configure_outputs(self, img_shape, readout_keys):
        name_map = {}
        for k in readout_keys:
            *_, area, cell = k.split('-')
            target_name = '{}{}'.format(self.cell_lut[cell], self.area_lut[area])
            name_map[k] = target_name
        self.name_map = name_map

        state_shapes = self.get_state_shapes(img_shape)

        out_shapes = {k: state_shapes[name_map[k]] for k in readout_keys}

        bn_dict = {}
        for k, out_shape in out_shapes.items():
            bn_dict[k] = BiasBatchNorm2d(out_shape[0], momentum=self.momentum)

        self.batch_norm = ModuleDict(bn_dict)

        return out_shapes

    def get_state_shapes(self, img_shape):
        tmp = Variable(torch.zeros(1, *img_shape[-3:]), volatile=True)
        self.unit.eval()
        _, state = self.unit(tmp)
        self.unit.train()
        shapes = {k: v.size()[1:] for k, v in state.items()}
        return shapes

    def forward(self, input, readout_key=None, timesteps=None, get_state=False):
        if timesteps is None:
            timesteps = self.timesteps

        if readout_key is None:
            readout_key = list(self.name_map.keys())

        method = self.ro_method

        is_single = isinstance(readout_key, str)
        if is_single:
            readout_key = [readout_key]

        state = None
        output = (0.0,) * len(readout_key)

        for i in range(timesteps):
            _, state = self.unit(input, state)

            if method == 'mean':
                output = tuple(o + self.batch_norm[k](state[self.name_map[k]]) / timesteps
                               for k, o in zip(readout_key, output))

        if method == 'last':
            output = tuple(self.batch_norm[k](state[self.name_map[k]])
                           for k in readout_key)

        if is_single:
            output = output[0]

        if get_state:
            return state
        else:
            return output


# ---------------------- Conv2d Cores -----------------------------

class Stacked2dCore(Core2d, nn.Module):
    def __init__(self, input_channels, hidden_channels, input_kern, hidden_kern, layers=3,
                 gamma_hidden=0, gamma_input=0., skip=0, final_nonlinearity=True, bias=False,
                 momentum=0.1, pad_input=True, batch_norm=True, **kwargs):
        self.msg('Ignoring input', kwargs, 'when creating', self.__class__.__name__)
        super().__init__()
        self._input_weights_regularizer = LaplaceL2()

        self.layers = layers
        self.gamma_input = gamma_input
        self.gamma_hidden = gamma_hidden
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.skip = skip

        self.features = nn.Sequential()
        # --- first layer
        layer = OrderedDict()
        layer['conv'] = \
            nn.Conv2d(input_channels, hidden_channels, input_kern,
                      padding=input_kern // 2 if pad_input else 0, bias=bias)
        if batch_norm:
            layer['norm'] = nn.BatchNorm2d(hidden_channels, momentum=momentum)
        if layers > 1 or final_nonlinearity:
            layer['nonlin'] = nn.ELU(inplace=True)
        self.features.add_module('layer0', nn.Sequential(layer))

        # --- other layers
        for l in range(1, self.layers):
            layer = OrderedDict()
            layer['conv'] = \
                nn.Conv2d(hidden_channels if not skip > 1 else min(skip, l) * hidden_channels,
                          hidden_channels, hidden_kern,
                          padding=hidden_kern // 2, bias=bias)
            if batch_norm:
                layer['norm'] = nn.BatchNorm2d(hidden_channels, momentum=momentum)
            if final_nonlinearity or l < self.layers - 1:
                layer['nonlin'] = nn.ELU(inplace=True)
            self.features.add_module('layer{}'.format(l), nn.Sequential(layer))

        self.apply(self.init_conv)

    def forward(self, input_):
        ret = []
        for l, feat in enumerate(self.features):
            do_skip = l >= 1 and self.skip > 1
            input_ = feat(input_ if not do_skip else torch.cat(ret[-min(self.skip, l):], dim=1))
            ret.append(input_)
        return torch.cat(ret, dim=1)

    def laplace(self):
        return self._input_weights_regularizer(self.features[0].conv.weight)

    def group_sparsity(self):
        ret = 0
        for l in range(1, self.layers):
            ret = ret + self.features[l].conv.weight.pow(2).sum(3, keepdim=True).sum(2, keepdim=True).sqrt().mean()
        return ret / ((self.layers - 1) if self.layers > 1 else 1)

    def regularizer(self):
        return self.group_sparsity() * self.gamma_hidden + self.gamma_input * self.laplace()

    @property
    def outchannels(self):
        return len(self.features) * self.hidden_channels


class LinearCore(Stacked2dCore):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, layers=1, skip=0, final_nonlinearity=False, bias=False,
                         hidden_kern=0, gamma_hidden=0)


# ---------------------- Conv3d Core -----------------------------
class Conv3dLinearCore(nn.Sequential, Core3d):
    def __init__(self, input_channels=1, input_kern=5, hidden_channels=32, momentum=0.99,
                 gamma_input=0, **kwargs):
        self.msg('Ignoring input', kwargs, 'when creating', self.__class__.__name__)
        super().__init__()
        self._input_weight_regularizer = LaplaceL23d()
        assert input_kern % 2 == 1, 'kernel sizes must be odd'

        self.gamma_input = gamma_input

        self.conv = nn.Conv3d(input_channels, hidden_channels, input_kern, bias=False)
        self.norm = nn.BatchNorm3d(hidden_channels, momentum=momentum)

    def laplace_l2(self):
        return self._input_weight_regularizer.cuda()(self.conv.weight)

    def regularizer(self):
        return self.laplace_l2() * self.gamma_input

    def forward(self, input):
        return self.norm(self.conv(input))


class Conv3dCore(nn.Sequential, Core3d):
    def __init__(self, input_channels=1, input_kern=5, hidden_kern=3, channels=32, dilation=1,
                 layers=3, momentum=0.99, progress=0, gamma_input=0, gamma_hidden=0., **kwargs):
        self.msg('Ignoring input', kwargs, 'when creating', self.__class__.__name__)
        super().__init__()
        self.laplace_reg = LaplaceL23d()
        assert input_kern % 2 == hidden_kern % 2 == 1, 'kernel sizes must be odd'
        self.layers = layers
        self.gamma_hidden = gamma_hidden
        self.gamma_input = gamma_input

        hidden_padding = hidden_kern // 2 if not isinstance(hidden_kern, tuple) else tuple(k // 2 for k in hidden_kern)
        input_padding = input_kern // 2 if not isinstance(input_kern, tuple) else tuple(k // 2 for k in input_kern)

        self.add_module('layer0',
                        nn.Sequential(
                            nn.Conv3d(input_channels, channels, input_kern, bias=False, padding=input_padding),
                            nn.BatchNorm3d(channels, momentum=momentum),
                            nn.ELU(inplace=True)
                        )
                        )

        for l in range(0, layers - 1):
            self.add_module('layer{}'.format(l + 1),
                            nn.Sequential(
                                nn.Conv3d(channels, channels, hidden_kern, bias=False,
                                          padding=hidden_padding, dilation=dilation, ),
                                nn.BatchNorm3d(channels, momentum=momentum),
                                nn.ELU(inplace=True)
                            )
                            )
            dilation += progress

    def laplace_l2(self):
        return self.laplace_reg.cuda()(self.mod0[0].weight)

    def group_sparsity(self):
        ret = 0
        for l in range(1, self.layers):
            ret = ret + self[l][0].weight.pow(2).sum(3, keepdim=True).sum(2, keepdim=True).sqrt().mean()
        return ret

    def regularizer(self):
        return self.group_sparsity() * self.gamma_hidden + self.laplace_l2() * self.gamma_input

    def forward(self, input):
        ret = []
        for l in range(self.layers):
            input = self[l](input)
            ret.append(input)
        return torch.cat(ret, dim=1)


class Stacked3dCore(Core3d, nn.Module):
    def __init__(self, input_channels, hidden_channels, input_kern, hidden_kern, layers=3,
                 gamma_hidden=0, gamma_input=0., skip=0, final_nonlinearity=True, bias=False,
                 momentum=0.1, pad_input=False, dilation=1, normalize=True, **kwargs):
        self.msg('Ignoring input', kwargs, 'when creating', self.__class__.__name__)
        super().__init__()
        self._input_weights_regularizer = LaplaceL23d()

        self.layers = layers
        self.gamma_input = gamma_input
        self.gamma_hidden = gamma_hidden
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.skip = skip

        self.features = nn.Sequential()

        def padding(k, d):
            if not isinstance(k, tuple) and not isinstance(d, tuple):
                return ((k - 1) * d  + 1) // 2
            else:
                if not isinstance(k, tuple):
                    k = len(d) * (k,)
                if not isinstance(d, tuple):
                    d = len(k) * (d,)
                return tuple(padding(kk, dd) for kk, dd in zip(k, d))


        # --- first layer
        layer = OrderedDict()
        layer['conv'] = \
            nn.Conv3d(input_channels, hidden_channels, input_kern,
                      padding=padding(input_kern, 1) if pad_input else 0, bias=bias, dilation=dilation)
        if normalize:
            # layer['norm'] = nn.BatchNorm3d(hidden_channels, momentum=momentum)
            layer['norm'] = nn.InstanceNorm3d(hidden_channels, momentum=momentum, eps=.1, affine=True)
        if layers > 1 or final_nonlinearity:
            layer['nonlin'] = nn.ELU(inplace=True)
        self.features.add_module('layer0', nn.Sequential(layer))

        # --- other layers

        for l in range(1, self.layers):
            layer = OrderedDict()
            dch = hidden_channels if not skip > 1 else min(skip, l) * hidden_channels
            layer['conv'] = \
                nn.Conv3d(dch,
                          hidden_channels, hidden_kern,
                          padding=padding(hidden_kern, 1), bias=bias)
            if normalize:
                # layer['norm'] = nn.BatchNorm3d(hidden_channels, momentum=momentum)
                layer['norm'] = nn.InstanceNorm3d(hidden_channels, momentum=momentum, eps=1., affine=True)

            if final_nonlinearity or l < self.layers - 1:
                layer['nonlin'] = nn.ELU(inplace=True)
            self.features.add_module('layer{}'.format(l), nn.Sequential(layer))


        self.apply(self.init_conv)

    def forward(self, input_):
        ret = []
        for l, feat in enumerate(self.features):
            do_skip = l >= 1 and self.skip > 1
            input_ = feat(input_ if not do_skip else torch.cat(ret[-min(self.skip, l):], dim=1))
            ret.append(input_)
        return torch.cat(ret, dim=1)

    def laplace(self):
        return self._input_weights_regularizer(self.features[0].conv.weight)

    def group_sparsity(self):
        ret = 0
        for l in range(1, self.layers):
            ret = ret + self.features[l].conv.weight.pow(2).sum(4, keepdim=True).sum(3, keepdim=True).sum(2,
                                                                                                          keepdim=True).sqrt().mean()
        return ret / ((self.layers - 1) if self.layers > 1 else 1)

    def regularizer(self):
        return self.group_sparsity() * self.gamma_hidden + self.gamma_input * self.laplace()

    @property
    def outchannels(self):
        return len(self.features) * self.hidden_channels


# ---- general RNN core cell
class RNNCore:
    @staticmethod
    def init_conv(m):
        if isinstance(m, nn.Conv2d):
            xavier_normal(m.weight.data)
            if m.bias is not None:
                init.constant(m.bias.data, 0.)

    def __repr__(self):
        s = super().__repr__()
        s += ' [{} regularizers: '.format(self.__class__.__name__)
        ret = []
        for attr in filter(lambda x: not x.startswith('_') and 'gamma' in x, dir(self)):
            ret.append('{} = {}'.format(attr, getattr(self, attr)))
        return s + '|'.join(ret) + ']\n'


# ---------------- GRU core --------------------------------

class ConvGRUCell(RNNCore, nn.Module, Messager):
    def __init__(self, input_channels, rec_channels, input_kern, rec_kern, gamma_rec=0, pad_input=True,
                 **kwargs):
        super().__init__()
        self.msg('Ignoring input', kwargs, 'when creating', self.__class__.__name__)

        self._laplace_reg = LaplaceL2()

        rec_padding = rec_kern // 2
        input_padding = input_kern // 2 if pad_input else 0
        self.rec_channels = rec_channels
        self._shrinkage = 0 if pad_input else input_kern - 1

        self.gamma_rec = gamma_rec
        self.reset_gate_input = nn.Conv2d(input_channels, rec_channels, input_kern, padding=input_padding)
        self.reset_gate_hidden = nn.Conv2d(rec_channels, rec_channels, rec_kern, padding=rec_padding)

        self.update_gate_input = nn.Conv2d(input_channels, rec_channels, input_kern, padding=input_padding)
        self.update_gate_hidden = nn.Conv2d(rec_channels, rec_channels, rec_kern, padding=rec_padding)

        self.out_gate_input = nn.Conv2d(input_channels, rec_channels, input_kern, padding=input_padding)
        self.out_gate_hidden = nn.Conv2d(rec_channels, rec_channels, rec_kern, padding=rec_padding)

        self.apply(self.init_conv)
        self.register_parameter('_prev_state', None)

    def init_state(self, input_):
        if self._prev_state is None:
            self.msg('Initializing first hidden state', depth=1)
            batch_size, _, *spatial_size = input_.data.size()
            state_size = [batch_size, self.rec_channels] + [s - self._shrinkage for s in spatial_size]
            prev_state = torch.zeros(*state_size)
            if input_.is_cuda:
                prev_state = prev_state.cuda()
            self._prev_state = Parameter(prev_state)
        return self._prev_state

    def forward(self, input_, prev_state):
        # get batch and spatial sizes

        # generate empty prev_state, if None is provided
        if prev_state is None:
            prev_state = self.init_state(input_)

        update = self.update_gate_input(input_) + self.update_gate_hidden(prev_state)
        update = F.sigmoid(update)

        reset = self.reset_gate_input(input_) + self.reset_gate_hidden(prev_state)
        reset = F.sigmoid(reset)

        out = self.out_gate_input(input_) + self.out_gate_hidden(prev_state * reset)
        h_t = F.tanh(out)
        new_state = prev_state * (1 - update) + h_t * update

        return new_state

    def regularizer(self):
        return self.gamma_rec * self.bias_l1()

    def bias_l1(self):
        return self.reset_gate_hidden.bias.abs().mean() / 3 + \
               self.update_gate_hidden.weight.abs().mean() / 3 + \
               self.out_gate_hidden.bias.abs().mean() / 3


class FeatureGRUCell(RNNCore, nn.Module):
    _BaseFeatures = None

    def __init__(self, input_channels, hidden_channels, rec_channels,
                 input_kern, hidden_kern, rec_kern, layers,
                 gamma_input, gamma_hidden, gamma_rec, momentum, **kwargs):
        super().__init__()
        self.features = self._BaseFeatures(input_channels, hidden_channels, input_kern, hidden_kern,
                                           layers=layers, momentum=momentum,
                                           gamma_input=gamma_input, gamma_hidden=gamma_hidden, **kwargs)
        self.gru = ConvGRUCell(self.features.outchannels,
                               rec_channels=rec_channels,
                               input_kern=rec_kern,
                               rec_kern=rec_kern,
                               gamma_rec=gamma_rec, **kwargs)

    def forward(self, x, prev_state=None):
        return self.gru(self.features(x), prev_state)

    def regularizer(self):
        return self.features.regularizer() + self.gru.regularizer()


class FeatureGRUCore(Core3d, nn.Module):
    _cell = None

    def __init__(self, input_channels, hidden_channels, rec_channels,
                 input_kern, hidden_kern, rec_kern, layers=2,
                 gamma_hidden=0, gamma_input=0, gamma_rec=0, momentum=.1, bias=True, **kwargs):
        super().__init__()
        self.cell = self._cell(input_channels, hidden_channels, rec_channels,
                               input_kern, hidden_kern, rec_kern, layers=layers,
                               gamma_input=gamma_input, gamma_hidden=gamma_hidden, gamma_rec=gamma_rec,
                               momentum=momentum, bias=bias, **kwargs)

    def regularizer(self):
        return self.cell.regularizer()

    def forward(self, input):
        N, _, d, w, h = input.size()
        states = []
        hidden = None

        x = input.transpose(1, 2).transpose(0, 1)

        for t in range(x.size(0)):
            hidden = self.cell(x[t, ...], hidden)
            states.append(hidden)
        return torch.stack(states, 2)


class StackedGRUCell(FeatureGRUCell):
    _BaseFeatures = Stacked2dCore


class StackedFeatureGRUCore(FeatureGRUCore):
    _cell = StackedGRUCell


# ------------- Static Cores -----------------------------
class StaticCore(Core3d, nn.Module):
    _BaseFeatures = None

    def __init__(self, input_channels, hidden_channels, input_kern, hidden_kern, layers,
                 gamma_input, gamma_hidden, momentum, **kwargs):
        super().__init__()
        self.features = self._BaseFeatures(input_channels, hidden_channels, input_kern, hidden_kern,
                                           layers=layers, momentum=momentum,
                                           gamma_input=gamma_input, gamma_hidden=gamma_hidden, **kwargs)

    def forward(self, input):
        N, _, d, w, h = input.size()
        states = []

        x = input.transpose(1, 2).transpose(0, 1)

        for t in range(x.size(0)):
            output = self.features(x[t, ...])
            states.append(output)
        return torch.stack(states, 2)

    def regularizer(self):
        return self.features.regularizer()


class StackedFeatureStaticCore(StaticCore):
    _BaseFeatures = Stacked2dCore

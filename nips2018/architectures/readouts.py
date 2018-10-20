import torch
from attorch.layers import (SpatialXFeatureLinear, SpatialXFeatureLinear3d,
                            elu1, FullLinear,
                            SpatialTransformerPooled3d, SpatialTransformerPooled2d,
                            SpatialTransformerPyramid3d, SpatialTransformerPyramid2d,
                            FactorizedSpatialTransformerPyramid2d)
from attorch.module import ModuleDict
import numpy as np
from ..utils.measures import corr
from sklearn.linear_model import LinearRegression
from torch import nn
from torch.nn.init import xavier_normal
from tqdm import tqdm
import warnings

from ..utils.logging import Messager


class Readout(Messager):
    def initialize(self, *args, **kwargs):
        raise NotImplementedError('initialize is not implemented for ', self.__class__.__name__)

    def __repr__(self):
        s = super().__repr__()
        s += ' [{} regularizers: '.format(self.__class__.__name__)
        ret = []
        for attr in filter(lambda x: not x.startswith('_') and
                                     ('gamma' in x or 'pool' in x or 'positive' in x), dir(self)):
            ret.append('{} = {}'.format(attr, getattr(self, attr)))
        return s + '|'.join(ret) + ']\n'


class FullyConnectedReadout(Readout, ModuleDict):
    def __init__(self, in_shape, neurons, gamma_readout, **kwargs):
        self.msg('Ignoring input', kwargs, 'when creating', self.__class__.__name__)
        super().__init__()

        self.in_shape = in_shape
        self.neurons = neurons
        self.gamma_readout = gamma_readout

        # use no nonlinearity - e.g. identity
        self.nonlinearity = lambda x: x

        for k, neur in neurons.items():
            if isinstance(self.in_shape, dict):
                in_shape = self.in_shape[k]
            self.add_module(k, FullLinear(in_shape, neur))

    def initialize(self, mu_dict):
        self.msg('Initializing', self.__class__.__name__, flush=True)
        for k, mu in mu_dict.items():
            self[k].initialize(init_noise=1e-6)
            self[k].bias.data = mu.squeeze() - 1

    def regularizer(self, readout_key):
        return self[readout_key].l1() * self.gamma_readout


class SpatialXFeaturesReadout(Readout, ModuleDict):
    def __init__(self, in_shape, neurons, gamma_readout, positive=True, normalize=True, **kwargs):
        self.msg('Ignoring input', kwargs, 'when creating', self.__class__.__name__)
        super().__init__()

        self.in_shape = in_shape
        self.neurons = neurons
        self.positive = positive
        self.normalize = normalize
        self.gamma_readout = gamma_readout

        for k, neur in neurons.items():
            if isinstance(self.in_shape, dict):
                in_shape = self.in_shape[k]
            self.add_module(k, SpatialXFeatureLinear(in_shape, neur, normalize=normalize, positive=positive))

    def initialize(self, mu_dict):
        self.msg('Initializing', self.__class__.__name__, flush=True)
        for k, mu in mu_dict.items():
            self[k].initialize(init_noise=1e-6)
            self[k].bias.data = mu.squeeze() - 1

    def regularizer(self, readout_key):
        return self[readout_key].l1() * self.gamma_readout


class SpatialXFeatures3dReadout(Readout, ModuleDict):
    def __init__(self, in_shape, neurons, positive=True, normalize=False, gamma_readout=0, **kwargs):
        self.msg('Ignoring input', kwargs, 'when creating', self.__class__.__name__)
        super().__init__()

        self.in_shape = in_shape
        self.neurons = neurons
        self.positive = positive
        self.normalize = normalize
        self.gamma_readout = gamma_readout

        for k, neur in neurons.items():
            if isinstance(self.in_shape, dict):
                in_shape = self.in_shape[k]
            self.add_module(k, SpatialXFeatureLinear3d(in_shape, neur, normalize=normalize, positive=positive))

    def initialize(self, mu_dict):
        self.msg('Initializing', self.__class__.__name__, flush=True)
        for k, mu in mu_dict.items():
            self[k].initialize(init_noise=1e-6)
            self[k].bias.data = mu.squeeze() - 1

    def regularizer(self, readout_key):
        return self[readout_key].l1() * self.gamma_readout


class SpatialTransformerPooled2dReadout(Readout, ModuleDict):
    _BaseReadout = None

    def __init__(self, in_shape, neurons, positive=False, gamma_features=0, pool_steps=0, **kwargs):
        self.msg('Ignoring input', kwargs, 'when creating', self.__class__.__name__)
        super().__init__()

        self.in_shape = in_shape
        self.neurons = neurons
        self._positive = positive
        self.gamma_features = gamma_features
        self._pool_steps = pool_steps
        for k, neur in neurons.items():
            if isinstance(self.in_shape, dict):
                in_shape = self.in_shape[k]
            self.add_module(k, SpatialTransformerPooled2d(in_shape, neur, positive=positive, pool_steps=pool_steps))

    @property
    def positive(self):
        return self._positive

    @positive.setter
    def positive(self, value):
        self._positive = value
        for k in self:
            self[k].positive = value

    def initialize(self, mu_dict):
        self.msg('Initializing with mu_dict:', *['{}: {}'.format(k, len(m)) for k, m in mu_dict.items()],
                 flush=True)

        for k, mu in mu_dict.items():
            self[k].initialize()
            self[k].bias.data = mu.squeeze() - 1

    def regularizer(self, readout_key):
        return self[readout_key].feature_l1() * self.gamma_features

    @property
    def pool_steps(self):
        return self._pool_steps

    @pool_steps.setter
    def pool_steps(self, value):
        self._pool_steps = value
        for k in self:
            self[k].poolsteps = value


class PooledReadout(Readout):
    @property
    def positive(self):
        return self._positive

    @positive.setter
    def positive(self, value):
        self._positive = value
        for k in self:
            self[k].positive = value

    def initialize(self, mu_dict):
        self.msg('Initializing with mu_dict:', *['{}: {}'.format(k, len(m)) for k, m in mu_dict.items()],
                 flush=True)
        for k, mu in mu_dict.items():
            self[k].initialize()
            self[k].bias.data = mu.squeeze() - 1

    def regularizer(self, readout_key, subs_idx=None):
        return self[readout_key].feature_l1(subs_idx=subs_idx) * self.gamma_features

    @property
    def pool_steps(self):
        return self._pool_steps

    @pool_steps.setter
    def pool_steps(self, value):
        self._pool_steps = value
        for k in self:
            self[k].poolsteps = value


class SpatialTransformerPooled3dReadout(PooledReadout, ModuleDict):
    def __init__(self, in_shape, neurons, positive=False, gamma_features=0, pool_steps=0, **kwargs):
        self.msg('Ignoring input', kwargs, 'when creating', self.__class__.__name__)
        super().__init__()

        self.in_shape = in_shape
        self.neurons = neurons
        self._positive = positive
        self.gamma_features = gamma_features
        self._pool_steps = pool_steps
        for k, neur in neurons.items():
            if isinstance(self.in_shape, dict):
                in_shape = self.in_shape[k]
            self.add_module(k, SpatialTransformerPooled3d(in_shape, neur, positive=positive, pool_steps=pool_steps))


class SpatialTransformer3dSharedGridReadout(PooledReadout, ModuleDict):
    def __init__(self, in_shape, neurons, positive=False, gamma_features=0, pool_steps=0, **kwargs):
        self.msg('Ignoring input', kwargs, 'when creating', self.__class__.__name__)
        super().__init__()

        self.in_shape = in_shape
        self.neurons = neurons
        self._positive = positive
        self.gamma_features = gamma_features
        self._pool_steps = pool_steps
        grid = None
        for k, neur in neurons.items():
            if isinstance(self.in_shape, dict):
                in_shape = self.in_shape[k]
            ro = SpatialTransformerPooled3d(in_shape, neur, positive=positive, pool_steps=pool_steps, grid=grid)
            grid = ro.grid
            self.add_module(k, ro)


class ST3dSharedGridStopGradientReadout(PooledReadout, ModuleDict):
    def __init__(self, in_shape, neurons, positive=False, gamma_features=0,
                 pool_steps=0, gradient_pass_mod=0, kernel_size=2, stride=2, **kwargs):
        self.msg('Ignoring input', kwargs, 'when creating', self.__class__.__name__)
        super().__init__()

        self.in_shape = in_shape
        self.neurons = neurons
        self._positive = positive
        self.gamma_features = gamma_features
        self._pool_steps = pool_steps
        old_neur = -1
        for ro_index, (k, neur) in enumerate(neurons.items()):
            if old_neur != neur:
                self.msg('Neuron change detected from', old_neur, 'to', neur, '! Resetting grid!', depth=1)
                grid = None
            old_neur = neur

            if isinstance(self.in_shape, dict):
                in_shape = self.in_shape[k]

            stop_grad = (ro_index % gradient_pass_mod) != 0
            if stop_grad:
                self.msg('Gradient for', k, 'will be blocked', depth=1)
            else:
                self.msg('Gradient for', k, 'will pass', depth=1)
            ro = SpatialTransformerPooled3d(in_shape, neur, positive=positive,
                                            pool_steps=pool_steps, grid=grid,
                                            stop_grad=stop_grad,
                                            kernel_size=kernel_size, stride=stride)
            grid = ro.grid
            self.add_module(k, ro)


class _SpatialTransformerPyramid(Readout, ModuleDict):
    _BaseReadout = None

    def __init__(self, in_shape, neurons, positive=False, gamma_features=0, scale_n=3, downsample=True,
                 type=None, _skip_upsampling=False, **kwargs):
        self.msg('Ignoring input', kwargs, 'when creating', self.__class__.__name__)
        super().__init__()

        self.in_shape = in_shape
        self.neurons = neurons
        self._positive = positive
        self.gamma_features = gamma_features
        for k, neur in neurons.items():
            if isinstance(self.in_shape, dict):
                in_shape = self.in_shape[k]
            self.add_module(k, self._BaseReadout(in_shape, neur, positive=positive, scale_n=scale_n,
                                                 _skip_upsampling=_skip_upsampling,
                                                 downsample=downsample, type=type))

    @property
    def positive(self):
        return self._positive

    @positive.setter
    def positive(self, value):
        self._positive = value
        for k in self:
            self[k].positive = value

    def initialize(self, mu_dict):
        self.msg('Initializing with mu_dict:', *['{}: {}'.format(k, len(m)) for k, m in mu_dict.items()],
                 flush=True)

        for k, mu in mu_dict.items():
            self[k].initialize()
            self[k].bias.data = mu.squeeze() - 1

    def regularizer(self, readout_key):
        return self[readout_key].feature_l1() * self.gamma_features


class SpatialTransformerPyramid2dReadout(_SpatialTransformerPyramid):
    _BaseReadout = SpatialTransformerPyramid2d


class SpatialTransformerPyramid3dReadout(_SpatialTransformerPyramid):
    _BaseReadout = SpatialTransformerPyramid3d


class FactorizedSTPyramid2dReadout(_SpatialTransformerPyramid):
    _BaseReadout = FactorizedSpatialTransformerPyramid2d

    def __init__(self, in_shape, neurons, positive=False, gamma_scale=0, gamma_channels=0,
                 gamma_features=0, scale_n=3, downsample=True, type=None, _skip_upsampling=False, **kwargs):
        if (gamma_scale > 0 or gamma_channels > 0) and gamma_features > 0:
            warnings.warn('Having non-zero gamma_features together with non-zero gamma_scale or gamma_channels '
                          'leads to redundant sparcity regularization')
        super().__init__(in_shape, neurons, positive=positive, gamma_feature=gamma_features,
                         scale_n=scale_n, downsample=downsample, type=type, _skip_upsampling=_skip_upsampling, **kwargs)
        self.gamma_scale = gamma_scale
        self.gamma_channels = gamma_channels

    def regularizer(self, readout_key):
        return super().regularizer(readout_key) + self[readout_key].scale_l1() * self.gamma_scale \
               + self[readout_key].channel_l1() * self.gamma_channels


class SimpleLinear(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.linear = nn.Linear(1, n, bias=True)

    def forward(self, input):
        batch, time = input.size()
        out = self.linear(input.contiguous().view(-1, 1))
        return out.view(batch, time, -1)


class SimpleLinearReadout(Readout, ModuleDict):
    def __init__(self, in_shape, neurons, gamma_linear=0, l=1, **kwargs):
        self.msg('Ignoring input', kwargs, 'when creating', self.__class__.__name__)
        super().__init__()

        self.in_shape = in_shape
        self.neurons = neurons
        self.gamma_linear = gamma_linear
        self.l = l

        for k, neur in neurons.items():
            self.add_module(k, SimpleLinear(neur))

    # def initialize(self, mu_dict):
    #     self.msg('Initializing with mu_dict:', *['{}: {}'.format(k, len(m)) for k, m in mu_dict.items()],
    #              flush=True)
    #     for k, mu in mu_dict.items():
    #         xavier_normal(self[k].linear.weight)
    #         self[k].linear.bias.data = mu.squeeze() - 1

    @staticmethod
    def load_data(datasets, readout_key):
        b, y = [np.stack(e) for e in zip(*[(b.numpy(), y.numpy()) for _, b, _, y in tqdm(datasets[readout_key])])]
        neurons = y.shape[-1]
        return b.reshape((-1, 3))[:, -1], y.reshape((-1, neurons))

    def initialize(self, trainsets, valsets=None):
        for readout_key in trainsets:
            v_train, y_train = self.load_data(trainsets, readout_key=readout_key)
            predictors = [LinearRegression().fit(v_train[:, None], e) for e in y_train.T]
            w = np.array([p.coef_[0] for p in predictors])
            b = np.array([p.intercept_ for p in predictors])

            self[readout_key].linear.weight.data = torch.from_numpy(w[:, None])
            self[readout_key].linear.bias.data = torch.from_numpy(b)

            if valsets is not None:
                v_val, y_val = self.load_data(valsets, readout_key=readout_key)
                y_hat = np.stack([p.predict(v_val[:, None]) for p in predictors], axis=1)
                self.msg('Initialized readout with linear regression. Validation correlation is',
                         corr(y_hat, y_val, axis=0).mean())

    def regularizer(self, readout_key):
        return self[readout_key].linear.weight.abs().pow(self.l).mean() * self.gamma_linear

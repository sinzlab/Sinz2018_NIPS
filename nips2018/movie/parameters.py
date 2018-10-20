from collections import OrderedDict, namedtuple, Counter
from itertools import product, count
from operator import add
from attorch.dataloaders import RepeatsBatchSampler
import torch
from functools import reduce
from torch.utils.data.sampler import SubsetRandomSampler, Sampler

from .transforms import Normalizer, Subsequence, ToTensor, Subsample
from ..utils.config import ConfigBase
from ..utils.logging import Messager
import datajoint as dj

import numpy as np
from torch.utils.data import DataLoader

from ..architectures import readouts, modulators, shifters, cores
from .data import MovieMultiDataset

# experiment = dj.create_virtual_module('experiment', 'pipeline_experiment')
# anatomy = dj.create_virtual_module('anatomy', 'pipeline_anatomy')
# xcorr = dj.create_virtual_module('xcorr', 'pipeline_xcorr')
# leaderboard = dj.create_virtual_module('leaderboard', 'pfahey_leaderboard')
# netflix = dj.create_virtual_module('netflix', 'pipeline_netflix')
schema = dj.schema('nips2018_parameters', locals())


# Datasets = namedtuple('Datasets', ['trainsets', 'valsets', 'testsets', 'n_neurons'])

@schema
class Seed(dj.Lookup):
    definition = """
    # random seed for training

    seed                 :  int # random seed
    ---
    """

    @property
    def contents(self):
        yield from zip([1009, 1215, 2606, 7797, 8142])


@schema
class CoreConfig(ConfigBase, dj.Lookup):
    _config_type = 'core'

    class Conv3dLinear(dj.Part):
        definition = """
        -> master
        ---
        hidden_channels       : int      # hidden channels
        input_kern            : int      # kernel size at input convolutional layers
        gamma_input           : double   # regularization constant for input  convolutional layers
        """

        @property
        def content(self):
            for p in product([16, 36], [13], [50]):
                d = dict(zip(self.heading.dependent_attributes, p))
                yield d

    class StackedFeatureGRU(dj.Part):
        definition = """
        -> master
        ---
        hidden_channels       : int      # hidden channels
        rec_channels          : int      # recurrent hidden channels
        input_kern            : int      # kernel size at input convolutional layers
        hidden_kern           : int      # kernel size at hidden convolutional layers
        rec_kern              : int      # kernel size at hidden convolutional layers
        layers                : int      # layers
        gamma_rec             : double   # regularization constant for recurrent bias term
        gamma_hidden          : double   # regularization constant for hidden layers in CNN
        gamma_input           : double   # regularization constant for input  convolutional layers
        skip                  : tinyint  # use skip connection to previous `skip` layers
        bias                  : bool     # use bias
        pad_input             : bool     # use padding
        momentum              : double     # use padding
        """

        @property
        def content(self):
            for p in product([9], [12], [7], [3], [3], [3], [0.0], [0.1], [50], [2], [False],
                             [True], [.1]):
                d = dict(zip(self.heading.dependent_attributes, p))
                yield d
            for p in product([8, 16], [16, 32], [7], [3], [3], [3], [0.0], [0.1], [50], [2], [False],
                             [True], [.1]):
                d = dict(zip(self.heading.dependent_attributes, p))
                yield d
            for p in product([12], [36], [7], [3], [3], [3], [0.0], [0.1], [50], [2], [False],
                             [True], [.1]):
                d = dict(zip(self.heading.dependent_attributes, p))
                yield d
            for p in product([16], [48], [7], [3], [3], [3], [0.0], [0.1], [50], [2], [False],
                             [True], [.1]):
                d = dict(zip(self.heading.dependent_attributes, p))
                yield d
            for p in product([12], [36], [13, 7, 5, 3], [1], [1], [3], [0.0], [0.1], [50], [2], [False],
                             [True], [.1]):
                d = dict(zip(self.heading.dependent_attributes, p))
                yield d

    class StackedFeatureStatic(dj.Part):
        definition = """
        -> master
        ---
        hidden_channels       : int      # hidden channels
        input_kern            : int      # kernel size at input convolutional layers
        hidden_kern           : int      # kernel size at hidden convolutional layers
        layers                : int      # layers
        gamma_hidden          : double   # regularization constant for hidden layers in CNN
        gamma_input           : double   # regularization constant for input  convolutional layers
        skip                  : tinyint  # use skip connection to previous `skip` layers
        bias                  : bool     # use bias
        pad_input             : bool     # use padding
        momentum              : double     # use padding
        """

        @property
        def content(self):
            yield dict(
                hidden_channels=12,
                input_kern=7,
                hidden_kern=3,
                layers=3,
                gamma_hidden=.1,
                gamma_input=50,
                skip=2,
                bias=False,
                pad_input=True,
                momentum=.1,
            )
            yield dict(
                hidden_channels=12,
                input_kern=13,
                hidden_kern=5,
                layers=5,
                gamma_hidden=.1,
                gamma_input=50,
                skip=2,
                bias=False,
                pad_input=True,
                momentum=.1,
            )

    def build(self, input_channels, key):
        core_key = self.parameters(key)
        core_name = '{}Core'.format(core_key.pop('core_type'))
        assert hasattr(cores, core_name), '''Cannot find core for {core_name}. 
                                             Core needs to be names "{core_name}Core" 
                                             in architectures.cores'''.format(core_name=core_name)
        Core = getattr(cores, core_name)
        return Core(input_channels=input_channels, **core_key)


@schema
class ReadoutConfig(ConfigBase, dj.Lookup):
    _config_type = 'ro'

    def build(self, in_shape, neurons, key):
        ro_key = self.parameters(key)
        ro_name = '{}Readout'.format(ro_key.pop('ro_type'))
        assert hasattr(readouts, ro_name), '''Cannot find readout for {ro_name}. 
                                             Core needs to be names "{ro_name}" 
                                             in architectures.readout'''.format(ro_name=ro_name)
        Readout = getattr(readouts, ro_name)
        return Readout(in_shape, neurons, **ro_key)

    class SpatialTransformerPooled3d(dj.Part):
        definition = """
        -> master
        ---
        gamma_features         : float # regularization constant for features
        positive               : bool  # whether the features will be restricted to be positive
        pool_steps             : tinyint  # number of pooling steps in the readout
        """

        @property
        def content(self):
            for p in product([.01, .1, 1.], [True, False], [4, 5]):
                d = dict(zip(self.heading.dependent_attributes, p))
                yield d

    class SpatialTransformer3dSharedGrid(dj.Part):
        definition = """
        -> master
        ---
        gamma_features         : float # regularization constant for features
        positive               : bool  # whether the features will be restricted to be positive
        pool_steps             : tinyint  # number of pooling steps in the readout
        """

        @property
        def content(self):
            for p in product([1., 5.], [False], [5]):
                d = dict(zip(self.heading.dependent_attributes, p))
                yield d

    class ST3dSharedGridStopGradient(dj.Part):
        definition = """
        -> master
        ---
        gamma_features         : float # regularization constant for features
        positive               : bool  # whether the features will be restricted to be positive
        pool_steps             : tinyint  # number of pooling steps in the readout
        gradient_pass_mod      : tinyint # all gradients of readout mod that number == 0 will pass
        kernel_size            : tinyint # kernel size for pooling
        stride                 : tinyint # stride for pooling
        """

        @property
        def content(self):
            for p in product([1.], [False], [3], [3], [3], [3]):
                d = dict(zip(self.heading.dependent_attributes, p))
                yield d
            for p in product([1.], [False], [5], [3], [2], [2]):
                d = dict(zip(self.heading.dependent_attributes, p))
                yield d
            for p in product([1.], [False], [2], [3], [4], [4]):
                d = dict(zip(self.heading.dependent_attributes, p))
                yield d

            for p in product([5.], [False], [2], [3], [4], [4]):
                d = dict(zip(self.heading.dependent_attributes, p))
                yield d


@schema
class ModulatorConfig(ConfigBase, dj.Lookup):
    _config_type = 'mod'

    def build(self, data_keys, input_channels, key):
        mod_key = self.parameters(key)
        mod_name = '{}Modulator'.format(mod_key.pop('mod_type'))
        assert hasattr(modulators, mod_name), '''Cannot find modulator for {mod_name}. 
                                             Core needs to be names "{mod_name}" 
                                             in architectures.readout'''.format(mod_name=mod_name)
        Modulator = getattr(modulators, mod_name)
        return Modulator(data_keys, input_channels, **mod_key)

    class GateGRU(dj.Part):
        definition = """
        -> master
        ---
        hidden_channels     : tinyint   # hidden channels of the GRU
        bias                : bool      # whether to use a bias or not
        gamma_modulator     : float     # regularization constant for sparse regularizer on fully connected part
        offset              : float     # offset added to exp(modsignal) 
        """

        @property
        def content(self):
            for p in product([5, 50], [True], [.0], [1.]):
                d = dict(zip(self.heading.dependent_attributes, p))
                yield d

    class No(dj.Part):
        definition = """
        -> master
        ---
        """

        @property
        def content(self):
            yield dict()


@schema
class ShifterConfig(ConfigBase, dj.Lookup):
    _config_type = 'shift'

    def build(self, data_keys, input_channels, key):
        shift_key = self.parameters(key)
        shift_name = '{}Shifter'.format(shift_key.pop('shift_type'))
        assert hasattr(shifters, shift_name), '''Cannot find modulator for {shift_name}. 
                                             Core needs to be names "{shift_name}" 
                                             in architectures.readout'''.format(shift_name=shift_name)
        Shifter = getattr(shifters, shift_name)
        return Shifter(data_keys, input_channels, **shift_key)

    class SharedGRU(dj.Part):
        definition = """
        -> master
        ---
        hidden_channels       : int   # kernel size at input convolutional layers
        gamma_shifter         : float # regularizer
        init_noise            : float # initialization noise
        bias                  : bool  # whether the GRU uses a bias
        """

        @property
        def content(self):
            for p in product([2], [0], [1e-3], [True]):
                d = dict(zip(self.heading.dependent_attributes, p))
                yield d

    class StaticAffine(dj.Part):
        definition = """
        -> master
        ---
        gamma_shifter         : float # regularizer
        bias                  : bool  # whether the GRU uses a bias
        """

        @property
        def content(self):
            for p in product([1e-3], [True]):
                d = dict(zip(self.heading.dependent_attributes, p))
                yield d

    class No(dj.Part):
        definition = """
        -> master
        ---
        """

        @property
        def content(self):
            yield dict()


class StimulusTypeMixin(Messager):
    _stimulus_type = None

    def add_transforms(self, key, datasets, tier, exclude=None):
        if exclude is not None:
            self.msg('Excluding', ','.join(exclude), 'from normalization')
        for k, dataset in datasets.items():
            transforms = []
            if tier == 'train':
                transforms.append(Subsequence(key['train_seq_len']))
            transforms.extend([Normalizer(dataset, stats_source=key['stats_source'], exclude=exclude), ToTensor()])
            dataset.transforms = transforms

        return datasets

    def get_constraint(self, dataset, stimulus_type, tier=None):
        constraint = np.zeros(len(dataset.types), dtype=bool)
        for const in map(lambda s: s.strip(), stimulus_type.split('|')):
            if const.startswith('~'):
                tmp = (dataset.types != const[1:])
            else:
                tmp = (dataset.types == const)
            constraint = constraint | tmp
        if tier is not None:
            constraint = constraint & (dataset.tiers == tier)
        return constraint

    def get_loaders(self, datasets, tier, batch_size, stimulus_types=None, balanced=False,
                    merge_noise_types=True, shrink_to_same_size=False):
        if stimulus_types is None:
            self.msg('Using', self._stimulus_type, 'as stimulus type for all datasets')
            stimulus_types = len(datasets) * [self._stimulus_type]
        if not isinstance(stimulus_types, list):
            self.msg('Using', stimulus_types, 'as stimulus type for all datasets')
            stimulus_types = len(datasets) * [stimulus_types]

        self.msg('Stimulus sources:', *stimulus_types)

        loaders = OrderedDict()
        constraints = [self.get_constraint(dataset, stimulus_type, tier=tier)
                       for dataset, stimulus_type in zip(datasets.values(), stimulus_types)]
        if shrink_to_same_size:
            if tier is None or tier in ('train', 'validation'):
                min_n = np.min([e.sum() for e in constraints])
                new_con = []
                for i, c, st in zip(count(), constraints, stimulus_types):
                    if not '|' in st:
                        c2 = c & (c.cumsum() <= min_n)
                    else:
                        c2 = c
                    untouched = c == c2
                    for c3 in constraints[i+1:]:
                        c3 &= untouched
                    new_con.append(c2)
                constraints = new_con
                self.msg('Shrinking each type in {} sets to same size of {}'.format(tier, min_n), depth=1)

        for (k, dataset), stimulus_type, constraint in zip(datasets.items(), stimulus_types, constraints):
            self.msg('Selecting trials from', stimulus_type, 'and tier=', tier)
            ix = np.where(constraint)[0]
            self.msg('Found {} active trials'.format(constraint.sum()), depth=1)
            if tier == 'train':
                if not balanced:
                    self.msg("Configuring random subset sampler for", k)
                    loaders[k] = DataLoader(dataset, sampler=SubsetRandomSampler(ix), batch_size=batch_size)
                else:
                    self.msg("Configuring balanced random subset sampler for", k)
                    if merge_noise_types:
                        self.msg("Balancing Clip vs. Rest", depth=1)
                        types = np.array([('Clip' if t == 'stimulus.Clip' else 'Noise') for t in dataset.types])
                    loaders[k] = DataLoader(dataset, sampler=BalancedSubsetSampler(ix, types), batch_size=batch_size)
                    self.msg('Number of samples in the loader will be', len(loaders[k].sampler), depth=1)
            else:
                self.msg("Configuring sequential subset sampler for", k)
                loaders[k] = DataLoader(dataset, sampler=SubsetSequentialSampler(ix), batch_size=batch_size)
                self.msg('Number of samples in the loader will be', len(loaders[k].sampler), depth=1)
            self.msg('Number of batches in the loader will be',
                     int(np.ceil(len(loaders[k].sampler) / loaders[k].batch_size)), depth=1)

        return loaders

    def load_data(self, key, tier=None, batch_size=1, key_order=None,
                  exclude_from_normalization=None, stimulus_types=None,
                  balanced=False, shrink_to_same_size=False):
        self.msg('Loading', self._stimulus_type, 'dataset with tier=', tier)
        datasets = MovieMultiDataset().fetch_data(key, key_order=key_order)
        for k, dat in datasets.items():
            if 'stats_source' in key:
                self.msg('Adding stats_source "{stats_source}" to dataset   '.format(**key))
                dat.stats_source = key['stats_source']

        self.msg('Using statistics source', key['stats_source'])
        datasets = self.add_transforms(key, datasets, tier, exclude=exclude_from_normalization)
        loaders = self.get_loaders(datasets, tier, batch_size, stimulus_types=stimulus_types,
                                   balanced=balanced, shrink_to_same_size=shrink_to_same_size)
        return datasets, loaders


class AreaLayerRawMixin(StimulusTypeMixin):
    def load_data(self, key, tier=None, batch_size=1, key_order=None, stimulus_types=None,
                  balanced=False, shrink_to_same_size=False):
        datasets, loaders = super().load_data(key, tier, batch_size, key_order,
                                              exclude_from_normalization=self._exclude_from_normalization,
                                              stimulus_types=stimulus_types,
                                              balanced=balanced, shrink_to_same_size=shrink_to_same_size)

        self.msg('Subsampling to layer "{layer}" and area "{brain_area}"'.format(**key))
        for readout_key, dataset in datasets.items():
            layers = dataset.neurons.layer
            areas = dataset.neurons.area
            idx = np.where((layers == key['layer']) & (areas == key['brain_area']))[0]
            dataset.transforms.insert(-1, Subsample(idx))
        return datasets, loaders


@schema
class DataConfig(ConfigBase, dj.Lookup, Messager):
    _config_type = 'data'

    def data_key(self, key):
        return dict(key, **self.parameters(key))

    def load_data(self, key, oracle=False, **kwargs):
        data_key = self.data_key(key)
        Data = getattr(self, data_key.pop('data_type'))
        datasets, loaders = Data().load_data(data_key, **kwargs)

        if oracle:
            self.msg('Placing oracle data samplers')
            for readout_key, loader in loaders.items():
                ix = loader.sampler.indices
                condition_hashes = datasets[readout_key].condition_hashes
                self.msg('Replacing', loader.sampler.__class__.__name__, 'with RepeatsBatchSampler', depth=1)
                loader.sampler = None

                datasets[readout_key].transforms = \
                    [tr for tr in datasets[readout_key].transforms if isinstance(tr, (Subsample, ToTensor))]
                loader.batch_sampler = RepeatsBatchSampler(condition_hashes, subset_index=ix)
        return datasets, loaders

    class Monet(dj.Part, StimulusTypeMixin):
        definition = """
        -> master
        ---
        stats_source            : varchar(50)  # normalization source
        train_seq_len           : smallint     # training sequence length in frames
        """
        _stimulus_type = 'stimulus.Monet'

        @property
        def content(self):
            for p in product([self._stimulus_type], [30 * 5]):
                yield dict(zip(self.heading.dependent_attributes, p))

    class Monet2(dj.Part, StimulusTypeMixin):
        definition = """
        -> master
        ---
        stats_source            : varchar(50)  # normalization source
        train_seq_len           : smallint     # training sequence length in frames
        """

        _stimulus_type = 'stimulus.Monet2'

        @property
        def content(self):
            for p in product([self._stimulus_type], [30 * 5]):
                yield dict(zip(self.heading.dependent_attributes, p))

    class Clip(dj.Part, StimulusTypeMixin):
        definition = """
        -> master
        ---
        stats_source            : varchar(50)  # normalization source
        train_seq_len           : smallint     # training sequence length in frames
        """
        _stimulus_type = 'stimulus.Clip'

        @property
        def content(self):
            for p in product([self._stimulus_type], [30 * 5]):
                yield dict(zip(self.heading.dependent_attributes, p))

    class EyetraceShuffledClip(dj.Part, StimulusTypeMixin):
        definition = """
        -> master
        ---
        stats_source            : varchar(50)  # normalization source
        train_seq_len           : smallint     # training sequence length in frames
        """
        _stimulus_type = 'stimulus.Clip'

        @property
        def content(self):
            for p in product([self._stimulus_type], [30 * 5]):
                yield dict(zip(self.heading.dependent_attributes, p))

        def load_data(self, key, tier=None, batch_size=1, key_order=None):
            datasets, loaders = \
                super().load_data(key, tier=tier, batch_size=batch_size, key_order=key_order)
            for k, dataset in datasets.items():
                ix = np.arange(len(dataset))
                if isinstance(loaders[k].sampler, (SubsetRandomSampler, SubsetSequentialSampler)):
                    idx = loaders[k].sampler.indices
                    self.msg('Shuffling subset of length n={}'.format(len(idx)), depth=1)
                    ix[idx] = np.random.permutation(idx)
                else:
                    self.msg('Shuffling all n={} trials'.format(len(ix)), depth=1)
                    ix = np.random.permutation(ix)
                dataset.shuffle_dims['eye_position'] = ix

            return datasets, loaders

    class EyetraceShuffledLayerClip(dj.Part, StimulusTypeMixin):
        definition = """
        -> master
        ---
        stats_source            : varchar(50)  # normalization source
        train_seq_len           : smallint     # training sequence length in frames
        -> experiment.Layer
        """
        _stimulus_type = 'stimulus.Clip'

        @property
        def content(self):
            for p in product([self._stimulus_type], [30 * 5], ['L2/3', 'L4']):
                yield dict(zip(self.heading.dependent_attributes, p))

        def load_data(self, key, tier=None, batch_size=1, key_order=None):
            datasets, loaders = \
                super().load_data(key, tier=tier, batch_size=batch_size, key_order=key_order)

            for readout_key, dataset in datasets.items():
                # --- select layers
                layers = dataset.neurons['layer'].value.astype(str)
                idx = np.where(layers == key['layer'])[0]
                dataset.transforms.insert(-1, Subsample(idx))

                # shuffle
                ix = np.arange(len(dataset))
                if isinstance(loaders[readout_key].sampler, (SubsetRandomSampler, SubsetSequentialSampler)):
                    idx = loaders[readout_key].sampler.indices
                    self.msg('Shuffling subset of length n={}'.format(len(idx)), depth=1)
                    ix[idx] = np.random.permutation(idx)
                else:
                    self.msg('Shuffling all n={} trials'.format(len(ix)), depth=1)
                    ix = np.random.permutation(ix)
                dataset.shuffle_dims['eye_position'] = ix

            return datasets, loaders

    class LayerClip(dj.Part, StimulusTypeMixin):
        definition = """
        -> master
        ---
        stats_source            : varchar(50)  # normalization source
        train_seq_len           : smallint     # training sequence length in frames
        -> experiment.Layer
        """
        _stimulus_type = 'stimulus.Clip'

        @property
        def content(self):
            for p in product([self._stimulus_type], [30 * 5], ['L2/3', 'L4']):
                yield dict(zip(self.heading.dependent_attributes, p))

        def load_data(self, key, tier=None, batch_size=1, key_order=None):
            datasets, loaders = super().load_data(key, tier, batch_size, key_order)
            for readout_key, dataset in datasets.items():
                layers = dataset.neurons.layer
                idx = np.where(layers == key['layer'])[0]
                dataset.transforms.insert(-1, Subsample(idx))
            return datasets, loaders

    class AreaLayerClip(dj.Part, StimulusTypeMixin):
        definition = """
        -> master
        ---
        stats_source            : varchar(50)  # normalization source
        train_seq_len           : smallint     # training sequence length in frames
        -> experiment.Layer
        -> anatomy.Area
        """
        _stimulus_type = 'stimulus.Clip'

        @property
        def content(self):
            for p in product([self._stimulus_type], [30 * 5], ['L2/3', 'L4'], ['V1']):
                yield dict(zip(self.heading.dependent_attributes, p))

    class AreaLayerClipRawInputResponse(dj.Part, AreaLayerRawMixin):
        definition = """
        -> master
        ---
        stats_source            : varchar(50)  # normalization source
        train_seq_len           : smallint     # training sequence length in frames
        -> experiment.Layer
        -> anatomy.Area
        """
        _stimulus_type = 'stimulus.Clip'
        _exclude_from_normalization = ['inputs', 'responses']

        @property
        def content(self):
            for p in product(['all'], [30 * 5], ['L2/3'], ['V1']):
                yield dict(zip(self.heading.dependent_attributes, p))

    class AreaLayerClipRawEyeShuffled(dj.Part, AreaLayerRawMixin):
        definition = """
        -> master
        ---
        stats_source            : varchar(50)  # normalization source
        train_seq_len           : smallint     # training sequence length in frames
        -> experiment.Layer
        -> anatomy.Area
        """
        _stimulus_type = 'stimulus.Clip'
        _exclude_from_normalization = ['inputs', 'responses']

        @property
        def content(self):
            for p in product(['all'], [30 * 5], ['L2/3'], ['V1']):
                yield dict(zip(self.heading.dependent_attributes, p))

        def load_data(self, key, tier=None, batch_size=1, key_order=None):
            datasets, loaders = \
                super().load_data(key, tier=tier, batch_size=batch_size, key_order=key_order)

            for readout_key, dataset in datasets.items():
                loader = loaders[readout_key]
                # shuffle
                ix = np.arange(len(loader.dataset))
                if isinstance(loader.sampler, (SubsetRandomSampler, SubsetSequentialSampler)):
                    idx = loader.sampler.indices
                    self.msg('Shuffling subset of length n={}'.format(len(idx)), depth=1)
                    ix[idx] = np.random.permutation(idx)
                else:
                    self.msg('Shuffling all n={} trials'.format(len(ix)), depth=1)
                    ix = np.random.permutation(ix)
                dataset.shuffle_dims['eye_position'] = ix

            return datasets, loaders

    # class AreaLayerLeaderboard(dj.Part, AreaLayerRawMixin):
    #     definition = """
    #     -> master
    #     ---
    #     stats_source            : varchar(50)  # normalization source
    #     train_seq_len           : smallint     # training sequence length in frames
    #     -> experiment.Layer
    #     -> anatomy.Area
    #     -> leaderboard.EnsembleCuration
    #     """
    #     _stimulus_type = 'stimulus.Clip'
    #     _exclude_from_normalization = ['inputs', 'responses']
    #
    #     @property
    #     def content(self):
    #         for p in product(['all'], [30 * 5], ['L2/3'], ['V1']):
    #             for k in leaderboard.EnsembleCuration().fetch("KEY"):
    #                 yield dict(zip(self.heading.dependent_attributes, p), **k)
    #
    #     def load_data(self, key, tier=None, batch_size=1, key_order=None):
    #         datasets, loaders = super().load_data(key, tier=tier, batch_size=batch_size, key_order=key_order)
    #         self.msg('Shrinking dataset according to ensemble selection')
    #         for rok in datasets:
    #             dataset, loader = datasets[rok], loaders[rok]
    #
    #             assert loader.sampler is not None, 'Only works with a loader with a sampler at the moment'
    #             assert MovieMultiDataset.Member() & key, 'Dataset and ensemble are not compatible'
    #
    #             condition_hashes = dataset.condition_hashes
    #             idx = loader.sampler.indices
    #             ensemble_conditions = (leaderboard.Ensemble.Condition() & key).fetch('condition_hash').astype('str')
    #
    #             selection = np.zeros(len(condition_hashes), dtype=bool)
    #             selection[idx] = True  # set all selections to true
    #             ensemble_selection = np.any(condition_hashes[:, None] == ensemble_conditions[None, :], axis=1)
    #
    #             if tier == 'validation':
    #                 new_selection = selection & ~ensemble_selection
    #                 self.msg('Dropping', (selection & ~new_selection).sum(), 'trials from dataset', depth=1)
    #             elif tier == 'test':
    #                 self.msg('Checking that test and ensemble do not overlap', depth=1)
    #                 assert ~np.any(selection & ensemble_selection), 'Test set and ensemble overlap'
    #                 new_selection = selection
    #             else:
    #                 new_selection = ensemble_selection
    #                 self.msg('New set of', new_selection.sum(), 'trials condigured', depth=1)
    #             assert np.any(new_selection), 'New dataset is empty'
    #             loader.sampler.indices = np.where(new_selection)[0]
    #         return datasets, loaders

    class GoodiesBadies(dj.Part, AreaLayerRawMixin):
        definition = """
        -> master
        ---
        stats_source            : varchar(50)  # normalization source
        train_seq_len           : smallint     # training sequence length in frames
        condition               : enum('goodies', 'badies', 'all')               
        -> experiment.Layer
        -> anatomy.Area
        """
        _stimulus_type = 'stimulus.Clip'
        _exclude_from_normalization = ['inputs', 'responses']

        @property
        def content(self):
            for p in product(['all'], [30 * 5], ['goodies','badies', 'all'], ['L2/3'], ['V1']):
                yield dict(zip(self.heading.dependent_attributes, p))

        _good = '83f3cbacdcaf870c62f8bf6ba290a6c2'
        _bad = 'eba6c8af262dbb8c1748b3ff701f950c'

        def load_data(self, key, tier=None, batch_size=1, key_order=None):
            datasets, loaders = super().load_data(key, tier=tier, batch_size=batch_size, key_order=key_order)
            self.msg('Shrinking dataset according to ensemble selection')
            repeats = (stimulus.Clip() * netflix.RepeatSet()).fetch('condition_hash').astype(str)
            goodies = (leaderboard.Ensemble.Condition() & dict(ensemble_hash=self._good)).fetch('condition_hash').astype(str)
            badies = (leaderboard.Ensemble.Condition() & dict(ensemble_hash=self._bad)).fetch('condition_hash').astype(str)
            for rok in datasets:
                dataset, loader = datasets[rok], loaders[rok]

                assert loader.sampler is not None, 'Only works with a loader with a sampler at the moment'
                assert MovieMultiDataset.Member() & key, 'Dataset and ensemble are not compatible'

                condition_hashes = dataset.condition_hashes
                idx = loader.sampler.indices

                selection = np.zeros(len(condition_hashes), dtype=bool)
                selection[idx] = True  # set all selections to true
                repeat_idx = np.any(condition_hashes[:, None] == repeats[None, :], axis=1)
                goodies_idx = np.any(condition_hashes[:, None] == goodies[None, :], axis=1)
                badies_idx = np.any(condition_hashes[:, None] == badies[None, :], axis=1)

                if tier == 'validation':
                    new_selection = selection & ~repeat_idx
                    self.msg('Dropping', (selection & ~new_selection).sum(), 'trials from dataset', depth=1)
                elif tier == 'test':
                    self.msg('Checking that test and ensemble do not overlap', depth=1)
                    assert ~np.any(selection & repeat_idx), 'Test set and ensemble overlap'
                    new_selection = selection
                else:
                    self.msg('Adding repeats to training set', depth=1)
                    new_selection = selection | repeat_idx # add repeats
                    if key['condition'] == 'goodies':
                        self.msg('dropping badies', depth=1)
                        new_selection = new_selection & ~badies_idx
                    elif key['condition'] == 'badies':
                        self.msg('dropping goodies', depth=1)
                        new_selection = new_selection & ~goodies_idx
                    self.msg('New set of', new_selection.sum(), 'trials condigured', depth=1)
                assert np.any(new_selection), 'New dataset is empty'
                loader.sampler.indices = np.where(new_selection)[0]
            return datasets, loaders

    class GoodiesBadies2(GoodiesBadies):
        _good = 'd6deb748c0a520b09bd82b30053e3eee'
        _bad = '95d7a7e7c99823ce2305c8a153ab355b'



    # class AreaLayerLeaderboardGenerations(dj.Part, AreaLayerRawMixin):
    #     definition = """
    #     -> master
    #     ---
    #     stats_source            : varchar(50)  # normalization source
    #     train_seq_len           : smallint     # training sequence length in frames
    #     -> experiment.Layer
    #     -> anatomy.Area
    #     -> leaderboard.Ensemble
    #     """
    #     _stimulus_type = 'stimulus.Clip'
    #     _exclude_from_normalization = ['inputs', 'responses']
    #
    #     @property
    #     def content(self):
    #         for p in product(['all'], [30 * 5], ['L2/3'], ['V1']):
    #             for k in leaderboard.Ensemble().fetch("KEY"):
    #                 yield dict(zip(self.heading.dependent_attributes, p), **k)
    #
    #     def load_data(self, key, tier=None, batch_size=1, key_order=None):
    #         datasets, loaders = super().load_data(key, tier=tier, batch_size=batch_size, key_order=key_order)
    #         self.msg('Shrinking dataset according to ensemble selection')
    #         for rok in datasets:
    #             dataset, loader = datasets[rok], loaders[rok]
    #
    #             assert loader.sampler is not None, 'Only works with a loader with a sampler at the moment'
    #             assert MovieMultiDataset.Member() & key, 'Dataset and ensemble are not compatible'
    #
    #             condition_hashes = dataset.condition_hashes
    #             idx = loader.sampler.indices
    #             ensemble_conditions = (leaderboard.Ensemble.Condition() & key).fetch('condition_hash').astype('str')
    #
    #             selection = np.zeros(len(condition_hashes), dtype=bool)
    #             selection[idx] = True  # set all selections to true
    #             ensemble_selection = np.any(condition_hashes[:, None] == ensemble_conditions[None, :], axis=1)
    #
    #             if tier == 'validation':
    #                 new_selection = selection & ~ensemble_selection
    #                 self.msg('Dropping', (selection & ~new_selection).sum(), 'trials from dataset', depth=1)
    #             elif tier == 'test':
    #                 self.msg('Checking that test and ensemble do not overlap', depth=1)
    #                 assert ~np.any(selection & ensemble_selection), 'Test set and ensemble overlap'
    #                 new_selection = selection
    #             else:
    #                 new_selection = ensemble_selection
    #                 self.msg('New set of', new_selection.sum(), 'trials condigured', depth=1)
    #             assert np.any(new_selection), 'New dataset is empty'
    #             loader.sampler.indices = np.where(new_selection)[0]
    #         return datasets, loaders

    class AreaLayerTrippyRawInputResponse(dj.Part, AreaLayerRawMixin):
        definition = """
        -> master
        ---
        stats_source            : varchar(50)  # normalization source
        train_seq_len           : smallint     # training sequence length in frames
        -> experiment.Layer
        -> anatomy.Area
        """
        _stimulus_type = 'stimulus.Trippy'
        _exclude_from_normalization = ['inputs', 'responses']

        @property
        def content(self):
            for p in product(['all'], [30 * 5], ['L2/3'], ['V1']):
                yield dict(zip(self.heading.dependent_attributes, p))

    class AreaLayerMonet2RawInputResponse(dj.Part, AreaLayerRawMixin):
        definition = """
        -> master
        ---
        stats_source            : varchar(50)  # normalization source
        train_seq_len           : smallint     # training sequence length in frames
        -> experiment.Layer
        -> anatomy.Area
        """
        _stimulus_type = 'stimulus.Monet2'
        _exclude_from_normalization = ['inputs', 'responses']

        @property
        def content(self):
            for p in product(['all'], [30 * 5], ['L2/3'], ['V1']):
                yield dict(zip(self.heading.dependent_attributes, p))

    class AreaLayerNoiseRawInputResponse(dj.Part, AreaLayerRawMixin):
        definition = """
        -> master
        ---
        stats_source            : varchar(50)  # normalization source
        train_seq_len           : smallint     # training sequence length in frames
        -> experiment.Layer
        -> anatomy.Area
        """
        _stimulus_type = '~stimulus.Clip'
        _exclude_from_normalization = ['inputs', 'responses']

        @property
        def content(self):
            for p in product(['all'], [30 * 5], ['L2/3'], ['V1']):
                yield dict(zip(self.heading.dependent_attributes, p))

    class AreaLayerSplitRawInputResponse(dj.Part, AreaLayerRawMixin):
        definition = """
        -> master
        ---
        stats_source            : varchar(50)  # normalization source
        train_seq_len           : smallint     # training sequence length in frames
        permute                 : tinyint      # cyclical shift of the datatypes among groups
        stimulus_types          : varchar(512) # stimulus source type for the different groups
        -> experiment.Layer
        -> anatomy.Area
        balanced                : bool         # whether sampling is balanced or not
        """
        _exclude_from_normalization = ['inputs', 'responses']

        @property
        def content(self):
            for p in product(['all'], [30 * 5], [0, 1],
                             ['stimulus.Clip,~stimulus.Clip',
                              'stimulus.Clip,~stimulus.Clip,stimulus.Clip|~stimulus.Clip'],
                             ['L2/3'], ['V1'], [0]):
                yield dict(zip(self.heading.dependent_attributes, p))
            for p in product(['all'], [30 * 5], [0],
                             ['stimulus.Clip,~stimulus.Clip,stimulus.Clip|~stimulus.Clip'],
                             ['L2/3'], ['V1'], [1]):
                yield dict(zip(self.heading.dependent_attributes, p))
            for p in product(['all'], [30 * 5], [0],
                             [
                                 'stimulus.Clip,~stimulus.Clip,stimulus.Clip|~stimulus.Clip,stimulus.Clip,~stimulus.Clip,stimulus.Clip|~stimulus.Clip'],
                             ['L2/3'], ['V1'], [1]):
                yield dict(zip(self.heading.dependent_attributes, p))
            for p in product(['all'], [30 * 5], [0],
                             ['~stimulus.Clip,stimulus.Clip,stimulus.Clip|~stimulus.Clip'],
                             ['L2/3'], ['V1'], [1]):
                yield dict(zip(self.heading.dependent_attributes, p))

        def load_data(self, key, tier=None, batch_size=1, key_order=None):
            t = [s.strip() for s in key['stimulus_types'].split(',')]
            T = len(t)
            n = len(MovieMultiDataset.Member() & key)
            permute = key['permute'] % T
            assert n // T >= 1, 'not enough member datasets to do splits'

            group_sizes = [n // T if i < T - 1 else n - (T - 1) * (n // T) for i in range(T)]
            stimulus_types = reduce(add, [g * [a] for g, a in zip(group_sizes, t[permute:] + t[:permute])])
            self.msg('Using stimulus types "{}"'.format('", "'.join(stimulus_types)))

            return super().load_data(key, tier, batch_size, key_order, stimulus_types=stimulus_types,
                                     balanced=bool(key['balanced']))

    class AreaLayerSplitRawSizeMatched(dj.Part, AreaLayerRawMixin):
        definition = """
        -> master
        ---
        stats_source            : varchar(50)  # normalization source
        train_seq_len           : smallint     # training sequence length in frames
        permute                 : tinyint      # cyclical shift of the datatypes among groups
        stimulus_types          : varchar(512) # stimulus source type for the different groups
        -> experiment.Layer
        -> anatomy.Area
        balanced                : bool         # whether sampling is balanced or not
        """
        _exclude_from_normalization = ['inputs', 'responses']

        @property
        def content(self):
            for p in product(['all'], [30 * 5], [0],
                             ['~stimulus.Clip,stimulus.Clip,stimulus.Clip|~stimulus.Clip',
                              'stimulus.Clip,~stimulus.Clip,stimulus.Clip|~stimulus.Clip'],
                             ['L2/3'], ['V1'], [1]):
                yield dict(zip(self.heading.dependent_attributes, p))

        def load_data(self, key, tier=None, batch_size=1, key_order=None):
            t = [s.strip() for s in key['stimulus_types'].split(',')]
            T = len(t)
            n = len(MovieMultiDataset.Member() & key)
            permute = key['permute'] % T
            assert n // T >= 1, 'not enough member datasets to do splits'

            group_sizes = [n // T if i < T - 1 else n - (T - 1) * (n // T) for i in range(T)]
            stimulus_types = reduce(add, [g * [a] for g, a in zip(group_sizes, t[permute:] + t[:permute])])
            self.msg('Using stimulus types "{}"'.format('", "'.join(stimulus_types)))

            return super().load_data(key, tier, batch_size, key_order, stimulus_types=stimulus_types,
                                     balanced=bool(key['balanced']), shrink_to_same_size=True)


def fill():
    DataConfig().fill()
    ReadoutConfig().fill()
    ModulatorConfig().fill()
    CoreConfig().fill()
    ShifterConfig().fill()


class SubsetSequentialSampler(Sampler):
    """Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class BalancedSubsetSampler(Sampler):
    """Samples elements randomly from a given list of indices, without replacement, balanced by occurence of types.

    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices, types, mode='shortest'):
        self.indices = indices
        c = Counter(types[indices])
        if mode == 'longest':
            self.num_samples = max(c.values())
            self.replacement = True
        elif mode == 'shortest':
            self.num_samples = min(c.values())
            self.replacement = False

        for e, n in c.items():
            c[e] = 1 / n
        self.weights = torch.DoubleTensor([c[types[i]] for i in indices])

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, self.replacement))

    def __len__(self):
        return self.num_samples

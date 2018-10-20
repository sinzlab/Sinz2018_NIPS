from collections import OrderedDict, namedtuple
from itertools import repeat
from pprint import pformat

from ._utils import Learner, CorePlusReadoutModel, compute_scores

from ..utils.git import gitlog

import datajoint as dj

from attorch.losses import PoissonLoss3d, PoissonLoss

from ..utils.logging import Messager
from .data import MovieScan
from ..utils import set_seed
import numpy as np
from .data import MovieMultiDataset, InputResponse
from .parameters import Seed, CoreConfig, ReadoutConfig, DataConfig, ShifterConfig, ModulatorConfig, ConfigBase
from .parameters import schema as parameter_schema
import torch


schema = dj.schema('nips2018_models', locals())

torch.backends.cudnn.benchmark = True

dj.config['external-model'] = dict(
    protocol='file',
    location='/external/movie-models/')


@parameter_schema
class TrainConfig(ConfigBase, dj.Lookup, Messager):
    _config_type = 'train'

    class Default(dj.Part, Messager):
        definition = """
        -> master
        ---
        batch_size             : int      # training and validation batchsize
        n_subsample=null       : int      # neuron subsample size
        n_subsample_test=null  : int      # neuron subsample size for test sets
        schedule               : longblob # learning rate schedule
        acc_gradient           : tinyint  # whether to accumulate gradient or not
        max_epoch              : int      # maximum number of epochs
        """

        @property
        def content(self):
            yield dict(batch_size=5, n_subsample_test=2000,
                       schedule=np.array([0.005, 0.001]), acc_gradient=1, max_epoch=500)

        def train(self, key, trainloaders, valloaders, n_neurons):
            img_shape = list(trainloaders.values())[0].dataset.img_shape

            max_neurons = np.max(list(n_neurons.values()))
            # set some parameters
            self.msg('Training sets')
            for dl in trainloaders.values():
                dl.batch_size = key['batch_size']
                self.msg(dl.dataset, depth=1)

            self.msg('Validation sets')
            for dl in valloaders.values():
                dl.batch_size = 1
                self.msg(dl.dataset, depth=1)
            n_subsample = key['n_subsample']
            n_subsample_test = key['n_subsample_test']

            # --- set some parameters

            criterion = PoissonLoss3d()

            def full_objective(model, readout_key, inputs, beh, eye_pos, targets):
                if n_subsample is not None:
                    subs_idx, _ = torch.randperm(n_neurons[readout_key])[:n_subsample].cuda().sort()
                else:
                    subs_idx = slice(None)

                outputs = model(inputs, readout_key, eye_pos=eye_pos, behavior=beh, subs_idx=subs_idx)
                return criterion(outputs, targets[..., subs_idx]) \
                       + model.core.regularizer() \
                       + model.readout.regularizer(readout_key, subs_idx=subs_idx) \
                       + (model.shifter.regularizer(readout_key) if model.shift else 0) \
                       + (model.modulator.regularizer(readout_key, subs_idx=subs_idx) if model.modulate else 0)

            # --- initialize
            stop_closure = Encoder().get_stop_closure(valloaders, subsamp_size=n_subsample_test)

            model = Encoder().build_model(key, img_shape=img_shape, n_neurons=n_neurons)
            mu_dict = {k: dl.dataset.mean_trial().responses for k, dl in trainloaders.items()}
            model.readout.initialize(mu_dict)
            model.core.initialize()
            if model.shifter is not None:
                biases = {k: -dl.dataset.mean_trial().eye_position for k, dl in trainloaders.items()}
                model.shifter.initialize(bias=biases)
            if model.modulator is not None:
                model.modulator.initialize()
            self.msg('Shipping model to GPU')
            model = model.cuda()
            print(model)

            epoch = 0
            # --- train core, modulator, and readout but not shifter
            schedule = key['schedule']
            self.msg('Full training'.ljust(30, '-'))
            for opt, lr in zip(repeat(torch.optim.Adam), schedule):
                self.msg('Training with learning rate', lr, depth=1)
                optimizer = opt(model.parameters(), lr=lr)

                model, epoch = Encoder().train(model, full_objective, optimizer,
                                               stop_closure, trainloaders,
                                               epoch=epoch, max_iter=key['max_epoch'],
                                               interval=max_neurons // n_subsample * 20 if n_subsample is not None else 20,
                                               patience=10, accumulate_gradient=key['acc_gradient']
                                               )
            model.eval()
            return model

    class MultiGPU(dj.Part, Messager):
        definition = """
        -> master
        ---
        batch_size      : int      # training and validation batchsize
        n_subsample_test=null  : int      # neuron subsample size for test sets
        schedule               : longblob # learning rate schedule
        acc_gradient           : tinyint  # whether to accumulate gradient or not
        max_epoch              : int      # maximum number of epochs
        """

        @property
        def content(self):
            yield dict(batch_size=5, n_subsample_test=500,
                       schedule=np.array([0.005, 0.001]), acc_gradient=1, max_epoch=500)
            yield dict(batch_size=8, n_subsample_test=2000,
                       schedule=np.array([0.005, 0.001]), acc_gradient=1, max_epoch=200)
            yield dict(batch_size=4, n_subsample_test=1000,
                       schedule=np.array([0.005, 0.001]), acc_gradient=1, max_epoch=200)
            yield dict(batch_size=3, n_subsample_test=1000,
                       schedule=np.array([0.005, 0.001]), acc_gradient=1, max_epoch=200)
            yield dict(batch_size=2, n_subsample_test=1000,
                       schedule=np.array([0.005, 0.001]), acc_gradient=1, max_epoch=500)
            yield dict(batch_size=8, schedule=np.array([0.005, 0.001]), acc_gradient=1, max_epoch=200)
            # reduced epoch training
            yield dict(batch_size=8, schedule=np.array([0.005]), acc_gradient=1, max_epoch=2)
            yield dict(batch_size=8, schedule=np.array([0.005]), acc_gradient=1, max_epoch=4)
            yield dict(batch_size=8, schedule=np.array([0.005]), acc_gradient=1, max_epoch=8)
            yield dict(batch_size=8, schedule=np.array([0.005]), acc_gradient=1, max_epoch=16)

        def train(self, key, trainloaders, valloaders, n_neurons):
            img_shape = list(trainloaders.values())[0].dataset.img_shape

            # set some parameters
            self.msg('Training sets')
            for dl in trainloaders.values():
                dl.batch_size = key['batch_size']
                self.msg(dl.dataset, depth=1)
                self.msg('Batchsize is', dl.batch_size, depth=1)

            self.msg('Validation sets')
            for dl in valloaders.values():
                dl.batch_size = 1
                self.msg(dl.dataset, depth=1)
            n_subsample_test = key['n_subsample_test']

            # --- get model
            stop_closure = Encoder().get_stop_closure(valloaders, subsamp_size=n_subsample_test)

            model = Encoder().build_model(key, img_shape=img_shape, n_neurons=n_neurons)
            mu_dict = OrderedDict([
                (k, dl.dataset.mean_trial().responses) for k, dl in trainloaders.items()
            ])
            model.readout.initialize(mu_dict)
            model.core.initialize()

            # --- set some parameters
            acc = key['acc_gradient']
            criterion = PoissonLoss3d()
            n_datasets = len(trainloaders)

            def full_objective(model, readout_key, inputs, beh, eye_pos, targets):
                outputs = model(inputs, readout_key, eye_pos=eye_pos, behavior=beh)
                return (criterion(outputs, targets) / n_datasets \
                        + model.core.regularizer() / n_datasets \
                        + model.readout.regularizer(readout_key).cuda(0) \
                        + (model.shifter.regularizer(readout_key) if model.shift else 0) \
                        + (model.modulator.regularizer(readout_key) if model.modulate else 0)) / acc

            # --- initialize
            if model.shifter is not None:
                biases = OrderedDict([
                    (k, -dl.dataset.mean_trial().eye_position) for k, dl in trainloaders.items()
                ])
                model.shifter.initialize(bias=biases)
            if model.modulator is not None:
                model.modulator.initialize()

            self.msg('Shipping model to GPU')
            model = model.cuda()
            print(model)

            epoch = 0
            # --- train core, modulator, and readout but not shifter
            schedule = key['schedule']
            if not isinstance(schedule, np.ndarray):
                schedule = np.array([schedule])
            self.msg('Full training')
            for opt, lr in zip(repeat(torch.optim.Adam), schedule):
                self.msg('Training with learning rate', lr, depth=1)
                optimizer = opt(model.parameters(), lr=lr)

                model, epoch = Encoder().train(model, full_objective, optimizer,
                                               stop_closure, trainloaders,
                                               epoch=epoch,
                                               max_iter=key['max_epoch'],
                                               interval=min(key['max_epoch'], 4),
                                               patience=4,
                                               accumulate_gradient=acc * n_datasets
                                               )
            model.eval()
            return model
            


    class MultiGPUStopGrad(dj.Part, Messager):
        definition = """
        -> master
        ---
        batch_size             : int      # training and validation batchsize
        n_subsample_test=null  : int      # neuron subsample size for test sets
        schedule               : longblob # learning rate schedule
        acc_gradient           : tinyint  # whether to accumulate gradient or not
        max_epoch              : int      # maximum number of epochs
        """

        @property
        def content(self):
            yield dict(batch_size=5, n_subsample_test=500,
                       schedule=np.array([0.005, 0.001]), acc_gradient=3, max_epoch=500)
            yield dict(batch_size=5, n_subsample_test=500,
                       schedule=np.array([0.005, 0.001]), acc_gradient=1, max_epoch=500)
            yield dict(batch_size=8, n_subsample_test=500,
                       schedule=np.array([0.005, 0.001]), acc_gradient=1, max_epoch=500)

        def train(self, key, trainloaders, valloaders, n_neurons):
            img_shape = list(trainloaders.values())[0].dataset.img_shape

            # set some parameters
            self.msg('Training sets')
            for dl in trainloaders.values():
                dl.batch_size = key['batch_size']
                self.msg(dl.dataset, depth=1)
                self.msg('Batchsize is', dl.batch_size, depth=1)

            self.msg('Validation sets')
            for dl in valloaders.values():
                dl.batch_size = 1
                self.msg(dl.dataset, depth=1)
            n_subsample_test = key['n_subsample_test']

            # --- get model
            stop_closure = Encoder().get_stop_closure(valloaders, subsamp_size=n_subsample_test)

            model = Encoder().build_model(key, img_shape=img_shape, n_neurons=n_neurons)
            mu_dict = OrderedDict([
                (k, dl.dataset.mean_trial().responses) for k, dl in trainloaders.items()
            ])
            model.readout.initialize(mu_dict)
            model.core.initialize()

            # --- set some parameters

            criterion = PoissonLoss3d()
            n_datasets = len(trainloaders)
            acc = key['acc_gradient']

            # --- setup objective
            grad_passes = 0
            for ro in model.readout.values():
                grad_passes += int(not ro.stop_grad)

            def full_objective(model, readout_key, inputs, beh, eye_pos, targets):
                outputs = model(inputs, readout_key, eye_pos=eye_pos, behavior=beh)
                return (criterion(outputs, targets)
                        + (model.core.regularizer() / grad_passes if not model.readout[readout_key].stop_grad else 0)
                        +  model.readout.regularizer(readout_key).cuda(0)
                        + (model.shifter.regularizer(readout_key) if model.shift else 0)
                        + (model.modulator.regularizer(readout_key) if model.modulate else 0)) / acc

            # --- initialize
            if model.shifter is not None:
                biases = OrderedDict([
                    (k, -dl.dataset.mean_trial().eye_position) for k, dl in trainloaders.items()
                ])
                model.shifter.initialize(bias=biases)
            if model.modulator is not None:
                model.modulator.initialize()

            self.msg('Shipping model to GPU')
            model = model.cuda()
            print(model)

            epoch = 0
            # --- train core, modulator, and readout but not shifter
            schedule = key['schedule']
            self.msg('Full training')
            for opt, lr in zip(repeat(torch.optim.Adam), schedule):
                self.msg('Training with learning rate', lr, depth=1)
                optimizer = opt(model.parameters(), lr=lr)

                model, epoch = Encoder().train(model, full_objective, optimizer,
                                               stop_closure, trainloaders,
                                               epoch=epoch,
                                               max_iter=key['max_epoch'],
                                               interval=4,
                                               patience=4,
                                               accumulate_gradient=acc * n_datasets
                                               )
            model.eval()
            return model

    def train_key(self, key):
        return dict(key, **self.parameters(key))

    def train(self, key, **kwargs):
        train_key = self.train_key(key)
        Trainer = getattr(self, train_key.pop('train_type'))
        return Trainer().train(train_key, **kwargs)


@schema
@gitlog
class Encoder(dj.Computed, Learner, CorePlusReadoutModel, Messager):
    definition = """
    -> MovieMultiDataset
    -> CoreConfig
    -> ReadoutConfig
    -> ShifterConfig
    -> ModulatorConfig
    -> DataConfig
    -> TrainConfig
    -> Seed
    ---
    val_corr                  : float       # validation correlation (single trial)
    model                     : external-model    # stored model
    """

    class TestScores(dj.Part):
        definition = """
        -> master
        -> MovieMultiDataset.Member
        ---
        neurons                  : int         # number of neurons
        pearson                  : float       # test correlation on single trial responses
        """

    class UnitTestScores(dj.Part):
        definition = """
        -> master.TestScores
        -> MovieScan.Unit
        ---
        pearson                  : float       # test correlation on single trial responses
        """

    def fill_test_scores(self, keys):
        for key in ((self & keys.proj()) - (self.TestScores & keys)).fetch(dj.key):
            testsets, testloaders = DataConfig().load_data(key, tier='test', batch_size=2)
            model = self.load_model(key).cuda()
            model.eval()
            scores, unit_scores = self.compute_test_score_tuples(key, testloaders, model)
            with self.connection.transaction:
                self.TestScores().insert(scores, ignore_extra_fields=True)
                self.UnitTestScores().insert(unit_scores, ignore_extra_fields=True)

    # @property
    # def key_source(self):
    #     ro_restr = [
    #         CoreConfig.StackedFeatureGRU().proj() * ReadoutConfig.SpatialTransformerPooled3d().proj('pool_steps') \
    #         & 'pool_steps=5',
    #         CoreConfig.StackedFeatureGRU().proj() * ReadoutConfig.SpatialTransformer3dSharedGrid().proj('pool_steps') \
    #         & 'pool_steps=5',
    #         CoreConfig.StackedFeatureGRU().proj() * ReadoutConfig.ST3dSharedGridStopGradient().proj('pool_steps') \
    #         & 'pool_steps=5',
    #         CoreConfig.StackedFeatureStatic().proj() * ReadoutConfig.SpatialTransformerPooled3d().proj('pool_steps') \
    #         & 'pool_steps=5',
    #         CoreConfig.Conv3dLinear().proj() * ReadoutConfig.SpatialTransformerPooled3d().proj('pool_steps') \
    #         & 'pool_steps=4',
    #     ]
    #     params = MovieMultiDataset() * CoreConfig() \
    #              * ReadoutConfig() \
    #              * DataConfig() * Seed() \
    #              * ShifterConfig() * ModulatorConfig() * TrainConfig()
    #     return params

    def _make_tuples(self, key):
        self.msg('Populating\n', pformat(key, indent=10), flush=True)
        # --- set seed
        set_seed((Seed() & key).fetch1('seed'))
        key0 = dict(key)

        # --- load data
        train_key = TrainConfig().train_key(key)
        trainsets, trainloaders = DataConfig().load_data(key, tier='train', batch_size=train_key['batch_size'])

        n_neurons = OrderedDict([(k, v.n_neurons) for k, v in trainsets.items()])
        valsets, valloaders = DataConfig().load_data(key, tier='validation', batch_size=1, key_order=trainsets)

        testsets, testloaders = DataConfig().load_data(key, tier='test', batch_size=1, key_order=trainsets)

        self.msg('Trainingsets\n', pformat(dict(trainsets), indent=10))
        model = TrainConfig().train(key, trainloaders=trainloaders,
                                    valloaders=valloaders,
                                    n_neurons=n_neurons)
        # --- test
        train_key = TrainConfig().train_key(key)
        val_closure = Encoder().get_stop_closure(valloaders,
                                                 subsamp_size=train_key['n_subsample_test'])
        key = self.update_key_with_validation_scores(key0, val_closure(model, avg=False))
        row = dict(key, model={k: v.cpu().numpy() for k, v in model.state_dict().items()})
        self.insert1(row)
        git_key = self.log_git(key)
        self.msg('Logging git key', pformat(git_key))
        scores, unit_scores = self.compute_test_score_tuples(key0, testloaders, model)
        self.TestScores().insert(scores, ignore_extra_fields=True)
        self.UnitTestScores().insert(unit_scores, ignore_extra_fields=True)
        print(80 * '=', flush=True)

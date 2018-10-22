import os
from _operator import attrgetter
from collections import OrderedDict, namedtuple
from contextlib import redirect_stdout
from itertools import chain
from pprint import pformat

import numpy as np
import torch
from attorch.dataset import to_variable
from attorch.layers import Elu1
from attorch.train import early_stopping, cycle_datasets
from scipy.stats import stats
from tqdm import tqdm

import datajoint as dj
from .data import MovieMultiDataset
from .parameters import CoreConfig, ReadoutConfig, Seed, ShifterConfig, ModulatorConfig, \
    DataConfig
from ..architectures.base import CorePlusReadout3d
from ..utils.logging import Messager
from ..utils.measures import corr

PerformanceScores = namedtuple('PerformanceScores', ['pearson'])


def spearm(pair):
    return stats.spearmanr(*pair)[0]


def variance_explained(y, y_hat, axis=0):
    s = y.var(axis=axis, ddof=1)
    zero_var = s < 1e-6
    vexpl = 1 - (y - y_hat).var(axis=0, ddof=1) / s
    if np.any(zero_var):
        vexpl[zero_var] = 0
    return vexpl


def slice_iter(n, step):
    k = 0
    while k < n - step:
        yield slice(k, k + step)
        k += step
    yield slice(k, None)


def compute_scores(y, y_hat, axis=0):
    # per_clip_pearson = np.array([corr(d, o, axis=axis) for d, o in zip(y, y_hat)])
    pearson = corr(y, y_hat, axis=axis)
    # with Pool(2) as p:
    #     per_clip_spearman = np.array([p.map(spearm, zip(d.T, o.T)) for d, o in zip(y, y_hat)])
    #     spearman = np.array(p.map(spearm, zip(np.vstack(y).T, np.vstack(y_hat).T)))
    #
    # vexpl = variance_explained(np.vstack(y), np.vstack(y_hat), axis=axis)
    # per_clip_vexpl = np.array([variance_explained(np.vstack(yy), np.vstack(yy_hat), axis=axis) for
    #                            yy, yy_hat in zip(y, y_hat)])
    return PerformanceScores(pearson=pearson)


class Learner(Messager):

    def update_key_with_validation_scores(self, key, corrs):
        key = dict(key)
        # corrs = stop(model, avg=False)
        corrs[np.isnan(corrs)] = 0
        key['val_corr'] = corrs.mean()
        return key

    @staticmethod
    def compute_predictions(loader, model, readout_key, reshape=True, stack=True, subsamp_size=None, return_lag=False):
        y, y_hat = [], []
        for x_val, beh_val, eye_val, y_val in tqdm(to_variable(loader, filter=(True, True, True, False),
                                                               cuda=True, volatile=True), desc='predictions'):
            neurons = y_val.size(-1)
            if subsamp_size is None:
                y_mod = model(x_val, readout_key, eye_pos=eye_val, behavior=beh_val).data.cpu().numpy()
            else:
                y_mod = []
                neurons = y_val.size(-1)
                for subs_idx in slice_iter(neurons, subsamp_size):
                    y_mod.append(
                        model(x_val, readout_key, eye_pos=eye_val,
                              behavior=beh_val, subs_idx=subs_idx).data.cpu().numpy())
                y_mod = np.concatenate(y_mod, axis=-1)

            lag = y_val.shape[1] - y_mod.shape[1]
            if reshape:
                y.append(y_val[:, lag:, :].numpy().reshape((-1, neurons)))
                y_hat.append(y_mod.reshape((-1, neurons)))
            else:
                y.append(y_val[:, lag:, :].numpy())
                y_hat.append(y_mod)
        if stack:
            y, y_hat = np.vstack(y), np.vstack(y_hat)
        if not return_lag:
            return y, y_hat
        else:
            return y, y_hat, lag

    def compute_test_scores(self, testloaders, model, readout_key):
        loader = testloaders[readout_key]

        y, y_hat = self.compute_predictions(loader, model, readout_key, reshape=True, stack=True, subsamp_size=None)
        return compute_scores(y, y_hat)  # scores is a named tuple

    def compute_test_score_tuples(self, key, testloaders, model):
        self.msg('Computing scores')
        scores, unit_scores = [], []
        for readout_key, testloader in testloaders.items():
            self.msg('for', readout_key, depth=1, flush=True)
            perf_scores = self.compute_test_scores(testloaders, model, readout_key)

            member_key = (MovieMultiDataset.Member() & key & dict(name=readout_key)).fetch1(dj.key)  # get other fields
            member_key.update(key)
            unit_ids = testloader.dataset.neurons.unit_ids
            member_key['neurons'] = len(unit_ids)
            member_key['pearson'] = perf_scores.pearson.mean()

            scores.append(member_key)
            unit_scores.extend([dict(member_key, unit_id=u, pearson=c) for u, c in zip(unit_ids, perf_scores.pearson)])
        return scores, unit_scores

    def get_stop_closure(self, valloaders, subsamp_size=None):

        def stop(mod, avg=True):
            ret = []
            train = mod.training
            mod.eval()
            for readout_key, loader in valloaders.items():
                y, y_hat = self.compute_predictions(loader, mod, readout_key,
                                                    reshape=True, stack=True, subsamp_size=subsamp_size)
                co = corr(y, y_hat, axis=0)
                self.msg(readout_key, 'correlation', co.mean(), depth=1)
                ret.append(co)
            ret = np.hstack(ret)
            if np.any(np.isnan(ret)):
                self.msg(' {}% NaNs '.format(np.isnan(ret).mean() * 100), depth=1, flush=True)
            ret[np.isnan(ret)] = 0
            # -- average if requested
            if avg:
                ret = ret.mean()
            mod.train(train)
            return ret

        return stop

    def train(self, model, objective, optimizer, stop_closure, trainloaders, epoch=0, post_epoch_hook=None,
              interval=1, patience=10, max_iter=10, maximize=True, tolerance=1e-6, cuda=True,
              restore_best=True, accumulate_gradient=1
              ):
        self.msg('Training models with', optimizer.__class__.__name__,
                 'gradient accumulation', accumulate_gradient,
                 'and state\n', pformat(model.state, indent=5))
        assert not isinstance(optimizer, torch.optim.LBFGS), "We don't BFGS at the moment. "
        optimizer.zero_grad()
        iteration = 0
        assert accumulate_gradient > 0, 'accumulate_gradient needs to be > 0'

        for epoch, val_obj in early_stopping(model, stop_closure,
                                             interval=interval, patience=patience,
                                             start=epoch, max_iter=max_iter, maximize=maximize,
                                             tolerance=tolerance, restore_best=restore_best):
            for batch_no, (readout_key, *data) in \
                    tqdm(enumerate(cycle_datasets(trainloaders, requires_grad=False, cuda=cuda)),
                         desc=self.__class__.__name__.ljust(25) + '  | Epoch {}'.format(epoch)):
                obj = objective(model, readout_key, *data)
                obj.backward()
                if iteration % accumulate_gradient == accumulate_gradient - 1:
                    optimizer.step()
                    optimizer.zero_grad()
                iteration += 1

            if post_epoch_hook is not None:
                model = post_epoch_hook(model, epoch)
        return model, epoch


class Model:
    def best_modulo(self):
        raise NotImplementedError('This function needs to be implemented by the subclasses')

    def load_model(self, key=None, img_shape=None, n_neurons=None):
        if key is None:
            key = self.fetch1(dj.key)
        model = self.build_model(key, img_shape=img_shape, n_neurons=n_neurons)
        state_dict = (self & key).fetch1('model')
        state_dict = {k: torch.from_numpy(state_dict[k][0]) for k in state_dict.dtype.names}
        mod_state_dict = model.state_dict()
        for k in set(mod_state_dict) - set(state_dict):
            self.msg('Could not find paramater', k, 'setting to initialization value', depth=1)
            state_dict[k] = mod_state_dict[k]
        model.load_state_dict(state_dict)
        return model

    @property
    def best(self):
        return self.best_modulo()


class CorePlusReadoutModel(Model):
    def best_modulo(self, *attrs):
        """
        Returns: best model according to validation error
        """
        from .models import TrainConfig
        pool_over = [CoreConfig(), ReadoutConfig(), Seed(), ShifterConfig(), ModulatorConfig(), TrainConfig()]
        h = self.heading.primary_key
        for e in chain(*map(attrgetter('heading.primary_key'), pool_over), attrs):
            if e not in attrs:
                h.remove(e)
        return self * dj.U(*h).aggr(self.proj('val_corr'), max_val='max(val_corr)') & 'val_corr = max_val'

    def build_model(self, key=None, img_shape=None, n_neurons=None, burn_in=15):
        """
        Builds a specified model
        Args:
            key:    key for CNNParameters used to load the parameter of the model. If None, (self & key) must
                    be non-empty so that key can be inferred.
            img_shape: image shape to figure out the size of the readouts
            n_neurons: dictionary with readout sizes (number of neurons)

        Returns:
            an uninitialized MultiCNN
        """
        if key is None:
            key = self.fetch1(dj.key)

        # --- load datasets
        if img_shape is None and n_neurons is None:
            with redirect_stdout(open(os.devnull, "w")):
                trainsets, _ = DataConfig().load_data(key)
            n_neurons = OrderedDict([(k, v.n_neurons) for k, v in trainsets.items()])
            img_shape = list(trainsets.values())[0].img_shape

        core = CoreConfig().build(img_shape[1], key)

        ro_in_shape = CorePlusReadout3d.get_readout_in_shape(core, img_shape)
        readout = ReadoutConfig().build(ro_in_shape, n_neurons, key)

        shifter = ShifterConfig().build(n_neurons, input_channels=2, key=key)

        modulator = ModulatorConfig().build(n_neurons, input_channels=3, key=key)

        # --- initialize
        return CorePlusReadout3d(core, readout, nonlinearity=Elu1(), shifter=shifter,
                                 modulator=modulator, burn_in=burn_in)

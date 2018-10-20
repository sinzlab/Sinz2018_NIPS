from itertools import product

import numpy as np
import torch
from attorch.dataloaders import RepeatsBatchSampler

import datajoint as dj
from .._utils import Learner
from ..data import MovieMultiDataset
from ..models import Encoder
from ..parameters import DataConfig
from ..transforms import Subsequence
from ...utils.git import gitlog
from ...utils.measures import corr

schema = dj.schema('nips2018_analysis_performance', locals())


def avg_pearson(y, y_hat):
    # y_hat, y have shape stimuli x repeats x time x neurons
    y = np.vstack([a.mean(axis=0, keepdims=True) for a in y])
    y_hat = np.vstack([a.mean(axis=0, keepdims=True) for a in y_hat])
    return corr(np.vstack(y), np.vstack(y_hat), axis=0)


def fev(y, y_hat):
    # y_hat, y have shape stimuli x repeats x time x neurons
    evary = np.mean([np.var(yy, axis=0) for yy in y], axis=(0, 1))
    varres = np.var(np.vstack(y) - np.vstack(y_hat), axis=(0, 1))
    vary = np.var(np.vstack(y), axis=(0, 1))
    return 1 - (varres - evary) / (vary - evary)


def pearson(y, y_hat):
    return corr(np.vstack(y), np.vstack(y_hat), axis=0)


def poisson(y, y_hat, bias=1e-16):
    y, y_hat = np.vstack(y), np.vstack(y_hat)
    return (y_hat - y * np.log(y_hat + bias)).mean(axis=0)


def make_loaders_repeated(loaders):
    for k, loader in loaders.items():
        ix = loader.sampler.indices
        dataset = loader.dataset
        condition_hashes = dataset.condition_hashes
        loader.sampler = None
        loader.batch_sampler = RepeatsBatchSampler(condition_hashes, subset_index=ix)
    return loaders


class PerformanceMeasurer(Learner):
    def compute_test_score_tuples(self, key, testloaders, model, scorers, reshape=False, stack=True, subsamp_size=250,
                                  readout_keys=None):
        self.msg('Computing ', *scorers)
        scores, unit_scores = [], []
        if readout_keys is None:
            readout_keys = list(testloaders)
        self.msg('Testing readouts:', readout_keys)
        for readout_key in readout_keys:
            testloader = testloaders[readout_key]
            self.msg('for', readout_key, depth=1, flush=True)
            loader = testloaders[readout_key]

            y, y_hat = self.compute_predictions(loader, model, readout_key,
                                                reshape=reshape, stack=stack, subsamp_size=subsamp_size)

            score_vals = {}
            for score_name, score in scorers.items():
                score_vals[score_name] = score(y, y_hat)

            member_key = (MovieMultiDataset.Member() & key & dict(name=readout_key)).fetch1(dj.key)  # get other fields
            member_key.update(key)
            unit_ids = testloader.dataset.neurons.unit_ids
            member_key['neurons'] = len(unit_ids)

            for score_name, val in score_vals.items():
                member_key[score_name] = val.mean()
            scores.append(member_key)

            for i, u in enumerate(unit_ids):
                ukey = dict(member_key, unit_id=u)

                for score_name, val in score_vals.items():
                    ukey[score_name] = val[i]
                unit_scores.append(ukey)
        return scores, unit_scores


@schema
class DataLink(dj.Lookup):
    definition = """
    -> DataConfig
    (test_data_hash) -> DataConfig 
    ---
    """

    @property
    def contents(self):
        remap = dict(test_data_hash='data_hash')
        confs = [DataConfig.AreaLayerNoiseRawInputResponse(),
                 DataConfig.AreaLayerClipRawInputResponse()]
        for conf, conf2 in product(confs, confs):
            rel = conf.proj() * conf2.proj(**remap)
            yield from rel.proj('test_data_hash').fetch(as_dict=True)

        rel = (DataConfig() & 'data_type like "%%SplitRaw%%"') * (DataConfig() & confs).proj(test_data_hash='data_hash')
        yield from rel.proj('test_data_hash').fetch(as_dict=True)


@schema
@gitlog
class XPearson(dj.Computed, PerformanceMeasurer):
    definition = """
    -> Encoder
    -> DataLink
    ---
    """

    class Scores(dj.Part):
        definition = """
        -> master
        -> MovieMultiDataset.Member
        ---
        neurons                  : int         # number of neurons
        pearson                  : float       # test correlation on single trial responses
        """

    class UnitScores(dj.Part):
        definition = """
        -> master.Scores
        -> MovieScan.Unit
        ---
        pearson                  : float       # test correlation on single trial responses
        """

    def make(self, key):
        data_key = dict(key)
        data_key['data_hash'] = data_key.pop('test_data_hash')

        orig_data = (DataConfig() & key).fetch1('data_type')
        test_data = (DataConfig() & data_key).fetch1('data_type')
        self.msg('Testing model trained on', orig_data, 'on', test_data)

        testsets, testloaders = DataConfig().load_data(data_key, tier='test')
        model = Encoder().load_model(key).cuda()
        model.eval()

        self.insert1(key)
        self.log_git(key)
        scores, unit_scores = self.compute_test_score_tuples(key, testloaders, model, {'pearson': pearson},
                                                             reshape=False, stack=True, subsamp_size=None)
        self.Scores().insert(scores, ignore_extra_fields=True)
        self.UnitScores().insert(unit_scores, ignore_extra_fields=True)


@schema
class Grid(dj.Lookup):
    definition = """
    # grid points for convex combination

    lambda_movies       : decimal(4,2) # combination factor for (movies - all)
    lambda_noise        : decimal(4,2) # combination factor for (noise  - all)
    ---
    """

    @property
    def contents(self):
        yield from [dict(lambda_movies=l1, lambda_noise=l2) for l1, l2 in
                    product(np.linspace(0, 1, 11), np.linspace(0, 1, 11))]


@schema
@gitlog
class ReadoutConvexComination(dj.Computed, PerformanceMeasurer):
    definition = """
    -> Encoder
    -> DataLink
    ---
    """

    key_source = Encoder() * DataLink()

    class Scores(dj.Part):
        definition = """
        -> master
        -> MovieMultiDataset.Member
        -> Grid
        ---
        neurons                  : int         # number of neurons
        poisson                  : float       # train Poisson loss on single trial responses
        pearson                  : float       # train Pearson correlation on single trial responses
        """

    def make(self, key):
        data_key = dict(key)
        data_key['data_hash'] = data_key.pop('test_data_hash')

        orig_data = (DataConfig() & key).fetch1('data_type')
        test_data = (DataConfig() & data_key).fetch1('data_type')
        self.msg('Testing model trained on', orig_data, 'on', test_data)

        testsets, testloaders = DataConfig().load_data(data_key, tier='train')  # we check the loss on the training set
        for rok, testset in testsets.items():
            testset.transforms = [tr for tr in testset.transforms if not isinstance(tr, Subsequence)]
        model = Encoder().load_model(key).cuda()
        model.eval()

        assert len(model.readout) == 3, 'only test on triple models'

        movie, noise, all = model.readout.keys()
        all0 = torch.cuda.FloatTensor(model.readout[all].features.data)
        dma = torch.cuda.FloatTensor(model.readout[movie].features.data - all0)
        dna = torch.cuda.FloatTensor(model.readout[noise].features.data - all0)

        self.insert1(key)
        self.log_git(key)
        for lmov, lnoi in map(lambda t: tuple(float(tt) for tt in t),
                              zip(*Grid().fetch('lambda_movies', 'lambda_noise'))):
            model.readout[all].features.data = all0 + lmov * dma + lnoi * dna
            keyins = dict(key, lambda_movies=lmov, lambda_noise=lnoi)
            scores, unit_scores = self.compute_test_score_tuples(keyins, testloaders, model,
                                                                 {'poisson': poisson, 'pearson': pearson},
                                                                 reshape=False, stack=True,
                                                                 subsamp_size=None,
                                                                 readout_keys=[all])
            assert len(scores) == 1, 'Returned more than one scores dict'
            self.msg('Testing @ movie={} and noise={}: pearson={pearson} and poisson={poisson}'.format(
                lmov, lnoi, **scores[0]))
            self.Scores().insert(scores, ignore_extra_fields=True)

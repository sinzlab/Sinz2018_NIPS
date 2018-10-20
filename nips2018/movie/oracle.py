from multiprocessing.pool import Pool

import datajoint as dj
from tqdm import tqdm

from ..utils.measures import corr
import numpy as np
from scipy import stats
from .parameters import DataConfig

from .data import MovieMultiDataset, MovieScan

from ..utils.logging import Messager

schema = dj.schema('nips2018_oracle', locals())


def spearm(pair):
    return stats.spearmanr(*pair)[0]


@schema
class MovieOracle(dj.Computed, Messager):
    definition = """
    # oracle computation for hollymonet data

    -> MovieMultiDataset
    -> DataConfig
    ---
    """

    @property
    def key_source(self):
        return MovieMultiDataset() * (
                    DataConfig() & [DataConfig.Clip(), DataConfig.Monet2(), DataConfig.AreaLayerClip(), DataConfig.AreaLayerClipRawInputResponse()])

    class TotalScores(dj.Part):
        definition = """
        -> master
        -> MovieMultiDataset.Member
        ---
        pearson           : float     # mean test correlation
        spearman          : float     # mean test spearman correlation
        """

    class PerClipScores(dj.Part):
        definition = """
        -> master
        -> MovieMultiDataset.Member
        ---
        pearson           : float     # mean test correlation
        spearman          : float     # mean test spearman correlation
        """

    class TotalUnitScores(dj.Part):
        definition = """
        -> master.TotalScores
        -> MovieScan.Unit
        ---
        pearson           : float     # mean test correlation
        spearman          : float     # mean test spearman correlation
        """

    class PerClipUnitScores(dj.Part):
        definition = """
        -> master.PerClipScores
        -> MovieScan.Unit
        ---
        pearson           : float     # mean test correlation
        spearman          : float     # mean test spearman correlation
        """

    def _make_tuples(self, key):
        self.msg('Populating', key)
        # --- load data
        testsets, testloaders = DataConfig().load_data(key, tier='test', oracle=True)

        self.insert1(dict(key))
        for readout_key, loader in testloaders.items():
            self.msg('Computing oracle for', readout_key, depth=1)
            oracles, data = [], []
            for inputs, *_, outputs in loader:
                inputs = inputs.numpy()
                assert np.all(np.abs(np.diff(inputs, axis=0)) == 0), \
                    'Video inputs of oracle trials does not match'
                outputs = outputs.numpy()
                new_shape = (-1, outputs.shape[-1])
                r, _, n = outputs.shape  # responses X neurons
                mu = outputs.mean(axis=0, keepdims=True)
                oracle = (mu - outputs / r) * r / (r - 1)
                oracles.append(oracle.reshape(new_shape))
                data.append(outputs.reshape(new_shape))
                # oracles.append(oracle)
                # data.append(outputs)

            per_clip_pearson = np.array([corr(d, o, axis=0) for d, o in zip(data, oracles)])
            pearson = corr(np.vstack(data), np.vstack(oracles), axis=0)

            with Pool(10) as p:
                per_clip_spearman = np.array([p.map(spearm, zip(d.T, o.T)) for d, o in zip(data, oracles)])
                spearman = np.array(p.map(spearm, zip(np.vstack(data).T, np.vstack(oracles).T)))
            member_key = (MovieMultiDataset.Member() & key & dict(name=readout_key)).fetch1(dj.key)
            member_key = dict(member_key, **key)
            self.TotalScores().insert1(dict(member_key, pearson=np.mean(pearson), spearman=np.mean(spearman)),
                                       ignore_extra_fields=True)
            self.PerClipScores().insert1(dict(member_key, pearson=np.mean(per_clip_pearson),
                                              spearman=np.mean(per_clip_spearman)),
                                         ignore_extra_fields=True)
            unit_ids = testsets[readout_key].neurons.unit_ids
            assert len(unit_ids) == len(pearson) == len(spearman) == outputs.shape[-1], 'Neuron numbers do not add up'
            self.TotalUnitScores().insert(
                [dict(member_key, pearson=c, spearman=sc, unit_id=u) \
                 for u, c, sc in tqdm(zip(unit_ids, pearson, spearman), total=len(unit_ids))],
                ignore_extra_fields=True)
            self.PerClipUnitScores().insert(
                [dict(member_key, pearson=c, spearman=sc, unit_id=u) \
                 for u, c, sc in tqdm(zip(unit_ids, per_clip_pearson.mean(axis=0),
                                          per_clip_spearman.mean(axis=0)), total=len(unit_ids))],
                ignore_extra_fields=True)

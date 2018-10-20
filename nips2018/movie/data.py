from collections import OrderedDict
from pprint import pformat

import numpy as np
from attorch.dataset import H5SequenceSet

import datajoint as dj
from .transforms import Subsequence
from ..utils.data import h5cached
from ..utils.logging import Messager

dj.config['external-file'] = dict(
    protocol='file',
    location='/external/movie-data/')

schema = dj.schema('nips2018_data', locals())


class ComputeStub:

    def make(selfk, key):
        pass


@schema
class MovieScan(dj.Computed, ComputeStub):
    definition = """
    # smaller primary key table for data

    animal_id            : int                          # id number
    session              : smallint                     # session index for the mouse
    scan_idx             : smallint                     # number of TIFF stack file
    ---
    -> fuse.ScanDone
    """

    class Unit(dj.Part):
        definition = """
        # smaller primary key table for data
        -> master
        unit_id              : int                          # unique per scan & segmentation method
        ---
        -> fuse.ScanSet.Unit
        """


@h5cached('/external/cache/', mode='groups', transfer_to_tmp=False,
          file_format='{animal_id}-{session}-{scan_idx}-preproc{preproc_id}.h5')
@schema
class InputResponse(dj.Computed, Messager):
    definition = """
    # responses of one neuron to the stimulus

    -> MovieScan
    -> Preprocessing
    ---
    """

    class Input(dj.Part):
        definition = """
            -> master
            -> stimulus.Trial
            -> MovieClips
            ---
            """

    class ResponseBlock(dj.Part):
        definition = """
            -> master
            -> master.Input
            ---
            responses           : external-file   # reponse of one neurons for all bins
            """

    class ResponseKeys(dj.Part):
        definition = """
            -> master.ResponseBlock
            -> fuse.Activity.Trace
            row_id           : int             # row id in the response block
            ---
            """

    def compute_data(self, key):
        # dummy function for constructor
        pass


@schema
class Eye(dj.Computed, ComputeStub):
    definition = """
    # eye movement data

    -> InputResponse.Input
    -> pupil.FittedContour
    ---
    pupil              : external-file   # pupil dilation trace
    dpupil             : external-file   # derivative of pupil dilation trace
    center             : external-file   # center position of the eye
    """


@schema
class Treadmill(dj.Computed, ComputeStub):
    definition = """
    # eye movement data

    -> InputResponse.Input
    -> treadmill.Treadmill
    ---
    treadmill          : external-file   # treadmill speed (|velcolity|)
    """


@schema
class MovieMultiDataset(dj.Manual, Messager):
    definition = """
    # defines a group of movie datasets

    group_id    : smallint  # index of group
    ---
    description : varchar(255) # short description of the data
    """

    class Member(dj.Part):
        definition = """
        -> master
        -> InputResponse
        ---
        name                    : varchar(50) unique # string description to be used for training
        """

    def fetch_data(self, key, key_order=None):
        assert len(self & key) == 1, 'Key must refer to exactly one multi dataset'
        ret = OrderedDict()
        self.msg('Fetching data for\n', pformat(key, indent=10))
        for mkey in (self.Member() & key).fetch(dj.key,
                                                order_by='animal_id ASC, session ASC, scan_idx ASC, preproc_id ASC'):
            name = (self.Member() & mkey).fetch1('name')
            include_behavior = bool(Eye() * Treadmill() & mkey)
            data_names = ['inputs', 'responses'] if not include_behavior \
                else ['inputs',
                      'behavior',
                      'eye_position',
                      'responses']
            self.msg('Data will be ({})'.format(','.join(data_names)))

            h5filename = InputResponse().get_hdf5_filename(mkey)
            self.msg('Loading dataset', name, '-->', h5filename)
            ret[name] = MovieSet(h5filename, *data_names)
        if key_order is not None:
            self.msg('Reordering datasets according to given key order', flush=True, depth=1)
            ret = OrderedDict([
                (k, ret[k]) for k in key_order
            ])
        return ret


class AttributeTransformer:
    def __init__(self, name, h5_handle, transforms):
        assert name in h5_handle, '{} must be in {}'.format(name, h5_handle)
        self.name = name
        self.h5_handle = h5_handle
        self.transforms = transforms

    def __getattr__(self, item):
        if not item in self.h5_handle[self.name]:
            raise AttributeError('{} is not among the attributes'.format(item))
        else:
            ret = self.h5_handle[self.name][item].value
            if ret.dtype.char == 'S':  # convert bytes to univcode
                ret = ret.astype(str)
            for tr in self.transforms:
                ret = tr.column_transform(ret)
            return ret


class MovieSet(H5SequenceSet):
    def __init__(self, filename, *data_keys, transforms=None, cache_raw=False, stats_source=None):
        super().__init__(filename, *data_keys, transforms=transforms)
        self.shuffle_dims = {}
        self.cache_raw = cache_raw
        self.last_raw = None
        self.stats_source = stats_source if stats_source is not None else 'all'

    @property
    def n_neurons(self):
        return self[0].responses.shape[1]

    @property
    def neurons(self):
        return AttributeTransformer('neurons', self._fid, self.transforms)

    @property
    def img_shape(self):
        return (1,) + self[0].inputs.shape

    def mean_trial(self, stats_source=None):
        if stats_source is None:
            stats_source = self.stats_source

        tmp = [np.atleast_1d(self.statistics['{}/{}/mean'.format(dk, stats_source)].value)
               for dk in self.data_groups]
        return self.transform(self.data_point(*tmp), exclude=Subsequence)

    def rf_base(self, stats_source='all'):
        N, c, t, w, h = self.img_shape
        t = min(t, 150)
        mean = lambda dk: self.statistics['{}/{}/mean'.format(dk, stats_source)].value
        d = dict(
            inputs=np.ones((1, c, t, w, h)) * np.array(mean('inputs')),
            eye_position=np.ones((1, t, 1)) * mean('eye_position')[None, None, :],
            behavior=np.ones((1, t, 1)) * mean('behavior')[None, None, :],
            responses=np.ones((1, t, 1)) * mean('responses')[None, None, :]
        )
        return self.transform(self.data_point(*[d[dk] for dk in self.data_groups]), exclude=Subsequence)

    def rf_noise_stim(self, m, t, stats_source='all'):
        """
        Generates a Gaussian white noise stimulus filtered with a 3x3 Gaussian filter
        for the computation of receptive fields. The mean and variance of the Gaussian
        noise are set to the mean and variance of the stimulus ensemble.

        The behvavior, eye movement statistics, and responses are set to their respective means.
        Args:
            m: number of noise samples
            t: length in time

        Returns: tuple of input, behavior, eye, and response

        """
        N, c, _, w, h = self.img_shape
        stat = lambda dk, what: self.statistics['{}/{}/{}'.format(dk, stats_source, what)].value
        mu, s = stat('inputs', 'mean'), stat('inputs', 'std')
        h_filt = np.float64([
            [1 / 16, 1 / 8, 1 / 16],
            [1 / 8, 1 / 4, 1 / 8],
            [1 / 16, 1 / 8, 1 / 16]]
        )
        noise_input = np.stack([convolve2d(np.random.randn(w, h), h_filt, mode='same')
                                for _ in range(m * t * c)]).reshape((m, c, t, w, h)) * s + mu

        mean_beh = np.ones((m, t, 1)) * stat('behavior', 'mean')[None, None, :]
        mean_eye = np.ones((m, t, 1)) * stat('eye_position', 'mean')[None, None, :]
        mean_resp = np.ones((m, t, 1)) * stat('responses', 'mean')[None, None, :]

        d = dict(
            inputs=noise_input.astype(np.float32),
            eye_position=mean_eye.astype(np.float32),
            behavior=mean_beh.astype(np.float32),
            responses=mean_resp.astype(np.float32)
        )

        return self.transform(self.data_point(*[d[dk] for dk in self.data_groups]), exclude=Subsequence)

    def __getitem__(self, item):
        x = self.data_point(*(np.array(self._fid[g][
                                           str(item if g not in self.shuffle_dims else self.shuffle_dims[g][item])])
                              for g in self.data_groups))
        if self.cache_raw:
            self.last_raw = x
        for tr in self.transforms:
            x = tr(x)
        return x

    def __repr__(self):
        return 'MovieSet m={}:\n\t({})'.format(len(self), ', '.join(self.data_groups)) \
               + '\n\t[Transforms: ' + '->'.join([repr(tr) for tr in self.transforms]) + ']' \
               + (
                   ('\n\t[Shuffled Features: ' + ', '.join(self.shuffle_dims) + ']') if len(
                       self.shuffle_dims) > 0 else '') + \
               ('\n\t[Stats source: {}]'.format(self.stats_source) if self.stats_source is not None else '')


schema.spawn_missing_classes()

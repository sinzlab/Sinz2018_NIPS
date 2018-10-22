import hashlib
import os.path as op
import pathlib
import pickle
from shutil import copyfile
from warnings import warn

from tqdm import tqdm


def list_hash(values):
    """
    Returns MD5 digest hash values for a list of values
    """
    hashed = hashlib.md5()
    for v in values:
        hashed.update(str(v).encode())
    return hashed.hexdigest()


from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
import numpy as np
import h5py
import os
from collections import OrderedDict


class SplineCurve:
    def __init__(self, t, s, **kwargs):
        t0 = np.nanmedian(t)  # center for numerical stability
        self.t0 = t0
        if len(t.shape) == 1:
            print('Using one common time traces per spline')
            self.splines = [NaNSpline(t - t0, si, **kwargs) for si in s]
        else:
            print('Using separate time traces per spline')
            self.splines = [NaNSpline(ti - t0, si, **kwargs) for ti, si in zip(t, s)]

    def __len__(self):
        return len(self.splines)

    def __call__(self, t, log=False):
        if log:
            return np.vstack([s(t - self.t0) for s in tqdm(self.splines, desc='Computing splines')])
        else:
            return np.vstack([s(t - self.t0) for s in self.splines])


class SplineMovie:
    def __init__(self, t, m, **kwargs):
        self.mshape = m.shape
        self.splines = [InterpolatedUnivariateSpline(t, mi, **kwargs) for mi in m.reshape((-1, self.mshape[-1]))]

    def __call__(self, t):
        return np.vstack([s(t) for s in self.splines]).reshape(self.mshape[:2] + (len(t),))


def merge(*args, **kwargs):
    """
    Merges several dictionaries. Checks for consistent values.

    :param args: dictionaries
    :return: merged dict
    """
    ret = dict(args[0])
    for d in args[1:] + (kwargs,):
        for k in d:
            if k in ret:
                if d[k] != ret[k]:
                    raise ValueError('Inconsistent values for key', k)
            else:
                ret[k] = d[k]
    return ret


def key_hash(key):
    """
    32-byte hash used for lookup of primary keys of jobs
    """
    hashed = hashlib.md5()
    for k, v in sorted(key.items()):
        hashed.update(str(v).encode())
    return hashed.hexdigest()


def fill_nans(x, preserve_gap=None):
    """
    :param x:  1D array  -- will
    :return: the array with nans interpolated
    The input argument is modified.
    """
    if preserve_gap is not None:
        assert preserve_gap % 2 == 1, 'can only efficiently preserve odd gaps'
        keep = np.convolve(np.convolve(1 * np.isnan(x), np.ones(preserve_gap), mode='same') == preserve_gap,
                           np.ones(preserve_gap, dtype=bool), mode='same')
    else:
        keep = np.zeros(len(x), dtype=bool)

    nans = np.isnan(x)

    x[nans] = 0 if nans.all() else np.interp(nans.nonzero()[0], (~nans).nonzero()[0], x[~nans])
    x[keep] = np.nan
    return x


def hamming(M):
    """
    Hamming window of lenth M

    Args:
        M: length of hamming window 

    Returns: numpy array with hamming window

    """
    n = np.arange(M)
    h = 0.54 - 0.46 * np.cos(2.0 * np.pi * n / (M - 1))
    return h / h.sum()


def dhamming(M):
    """
    Derivative of hamming window

    Args:
        M: length of hamming window 

    Returns:  numpy array with hamming window

    """
    n = np.arange(M)
    z = hamming(M).sum()
    return 0.46 * np.sin(2.0 * np.pi * n / (M - 1)) / z


class cached_data:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir

    def __call__(self, cls):
        if not hasattr(cls, 'compute_data'):
            raise AttributeError(
                'Class {name} needs a method compute_data(key, **kwargs) to be decorated with cached_data'.format(
                    name=cls.__name__))

        def fetch1_data(oself, key=None, **kwargs):
            if key is None:
                key = {}

            if not (oself & key):
                raise ValueError('Dataset not found!')
            if len(oself & key) > 1:
                raise ValueError('Can only return one dataset!')

            k = (oself & key).proj().fetch1()
            hash = key_hash(dict(k, **kwargs))
            filename = os.path.join(self.cache_dir, '{hash}.pkl'.format(hash=hash))

            if not os.path.isfile(filename):
                print('Computing data and saving to', filename, flush=True)
                data = oself.compute_data(key, **kwargs)
                with open(filename, 'wb') as fid:
                    pickle.dump(data, fid, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                print('Loading data from', filename, flush=True)
                with open(filename, 'rb') as fid:
                    data = pickle.load(fid)
            return data

        cls.fetch1_data = fetch1_data

        return cls


class h5cached:
    def __init__(self, cache_dir, mode='array', transfer_to_tmp=True, file_format=None):
        self.cache_dir = cache_dir
        pathlib.Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.transfer_to_tmp = transfer_to_tmp
        self.file_format = file_format

    def __call__(self, cls):
        if not hasattr(cls, 'compute_data'):
            raise AttributeError(
                'Class {name} needs a method compute_data(key, **kwargs) to be decorated with cached_data'.format(
                    name=cls.__name__))

        def _get_filename(oself, key=None, **kwargs):
            if key is None:
                key = {}

            if not (oself & key):
                raise ValueError('Dataset not found!')
            if len(oself & key) > 1:
                raise ValueError('Can only return one dataset!')

            k = (oself & key).proj().fetch1()

            if self.file_format is None:
                hash = key_hash(dict(k, **kwargs))  # TODO add in class name eventually
                filename = '{hash}.h5'.format(hash=hash)
            else:
                filename = self.file_format.format(**k)

            cachefile = op.join(self.cache_dir, filename)

            if op.exists(cachefile):
                tmpfile = op.join('/tmp', filename)
                if self.transfer_to_tmp:
                    if not op.exists(tmpfile) or (os.stat(cachefile).st_mtime - os.stat(tmpfile).st_mtime > 1):
                        print('Copying {} to {}'.format(cachefile, tmpfile), flush=True)
                        copyfile(cachefile, tmpfile)
                    return tmpfile
            return cachefile

        def fetch1_data(oself, key=None, **kwargs):
            filename = oself._get_filename(key, **kwargs)

            if not os.path.isfile(filename):
                print('Computing data and saving to', filename, flush=True)
                data = oself.compute_data(key, **kwargs)
                save_dict_to_hdf5(data, filename)

            # --- copy to tmp dir
            tmpfile = op.join('/tmp', filename.split('/')[-1])
            if self.transfer_to_tmp:
                if not op.exists(tmpfile) or (os.stat(filename).st_mtime - os.stat(tmpfile).st_mtime > 1):
                    print('Copying {} to {}'.format(filename, tmpfile), flush=True)
                    copyfile(filename, tmpfile)
                print('Loading data from', tmpfile, flush=True)
                data = load_dict_from_hdf5(tmpfile)
            else:
                print('Loading data from', filename, flush=True)
                data = load_dict_from_hdf5(filename)

            return data

        def get_hdf5_filename(oself, key=None, **kwargs):
            filename = oself._get_filename(key, **kwargs)
            if not os.path.isfile(filename):
                print('Computing data and saving to', filename, flush=True)
                data = oself.compute_data(key, **kwargs)
                save_dict_to_hdf5(data, filename)
            return filename

        cls.fetch1_data = fetch1_data
        cls._get_filename = _get_filename
        cls.get_hdf5_filename = get_hdf5_filename

        return cls


class FilterMixin:
    @staticmethod
    def get_filter(duration, d, type='hamming', warning=True):
        """
        Returns a hamming filter for downsampling a signal with period d to a signal with sampling
        period duration.

        Args:
            duration: target sampling period
            d:        source sampling period
            type:     filte type (only hamming at the moment)

        Returns: filter

        """
        if duration < d:
            if warning:
                warn('Target sampling period < source sampling period! Returning identity.')
            h = np.ones(1)
        else:
            if type == 'hamming':
                h = hamming(2 * int(duration // d) + 1)
            elif type == 'dhamming':
                h = dhamming(2 * int(duration // d) + 1)
            else:
                raise NotImplementedError('Filter {}  not implemented'.format(filter))
        return h


def save_dict_to_hdf5(dic, filename):
    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic)


def recursively_save_dict_contents_to_group(h5file, path, dic):
    for key, item in dic.items():
        if isinstance(item, list) or (isinstance(item, np.ndarray) and item.dtype is np.dtype('O')):
            for i, v in enumerate(item):
                h5file[path + key + '/' + str(i)] = v
            h5file[path + key].attrs['_iterable'] = True
        elif isinstance(item, (np.ndarray, np.int64, np.float64, np.float32, str, bytes)):
            h5file[path + key] = item
        elif isinstance(item, (dict, OrderedDict)):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        else:
            print(item, flush=True)
            raise ValueError('Cannot save %s type' % type(item))


def load_dict_from_hdf5(filename):
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')


def recursively_load_dict_contents_from_group(h5file, path):
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item.value
        elif isinstance(item, h5py._hl.group.Group):
            if '_iterable' in item.attrs and item.attrs['_iterable']:
                ans[key] = [item[str(i)] for i in range(len(item))]
            else:
                ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans


class FilterMixin:
    @staticmethod
    def get_filter(duration, d, type='hamming', warning=True):
        if duration < d:
            if warning:
                warn('Target sampling period < source sampling period! Returning identity.')
            h = np.ones(1)
        else:
            if type == 'hamming':
                h = hamming(2 * int(duration // d) + 1)
            elif type == 'dhamming':
                h = dhamming(2 * int(duration // d) + 1)
            else:
                raise NotImplementedError('Filter {}  not implemented'.format(filter))
        return h


def to_native(key):
    if isinstance(key, (dict, OrderedDict)):
        for k, v in key.items():
            if hasattr(v, 'dtype') and len(v) < 2:
                key[k] = v.item()
    else:
        for k, v in enumerate(key):
            if hasattr(v, 'dtype') and len(v) < 2:
                key[k] = v.item()
    return key


class NaNSpline(InterpolatedUnivariateSpline):

    def __init__(self, x, y, **kwargs):
        xnan = np.isnan(x)
        if np.any(xnan):
            warn('Found nans in the x-values. Replacing them with linear interpolation')
        ynan = np.isnan(y)
        w = xnan | ynan  # get nans
        x, y = map(np.array, [x, y])  # copy arrays
        y[ynan] = 0
        x[xnan] = np.interp(np.where(xnan)[0], np.where(~xnan)[0], x[~xnan])
        super().__init__(x[~w], y[~w], **kwargs)  # assign zero weight to nan positions

        self.nans = interp1d(x, 1 * w, kind='linear')

    def __call__(self, x, **kwargs):
        ret = np.zeros_like(x)
        newnan = np.zeros_like(x)

        old_nans = np.isnan(x)
        newnan[old_nans] = 1
        newnan[~old_nans] = self.nans(x[~old_nans])

        idx = newnan > 0
        ret[idx] = np.nan
        ret[~idx] = super().__call__(x[~idx], **kwargs)
        return ret

from collections import OrderedDict

import numpy as np
import torch
from attorch.dataset import MultiTensorDataset, to_variable
from scipy.interpolate import interp1d
from torch.utils.data import DataLoader
from tqdm import tqdm

import datajoint as dj
from ..data import Preprocessing, MovieMultiDataset
from ..models import Encoder
from ..parameters import DataConfig
from ...utils.logging import Messager

schema = dj.schema('nips2018_analysis_tuning', locals())
stimulus = dj.create_virtual_module('stimulus', 'pipeline_stimulus')

dj.config['external-data'] = dict(
    protocol='file',
    location='/external/movie-analysis/')


@schema
class MonetData(dj.Computed, Messager):
    definition = """
    -> stimulus.Monet2
    -> Preprocessing
    ---
    frames         : external-data
    """

    @property
    def key_source(self):
        return stimulus.Monet2() * Preprocessing() & dict(fps=30, resample_freq=30)

    def make(self, key):
        pass

    def load_data(self, network_key):
        dk = DataConfig().data_key(network_key)
        data, load = DataConfig().load_data(network_key)
        if 'stats_source' in dk:
            stats_source = dk['stats_source']
        else:
            stats_source = 'all'

        new_data = OrderedDict()
        condition_hashes = OrderedDict()
        for rok, dat in data.items():
            preproc = MovieMultiDataset.Member() & network_key & dict(name=rok)
            X, ch = (self & network_key & preproc).fetch('frames', 'condition_hash')
            X = np.stack(X, axis=0)[:, None, ...]

            _, beh, eye, _ = dat.mean_trial(stats_source=stats_source)
            beh = torch.ones(X.shape[0], X.shape[-3], 1) * beh[None, None, :]
            eye = torch.ones(X.shape[0], X.shape[-3], 1) * eye[None, None, :]
            new_data[rok] = MultiTensorDataset(X, beh.numpy(), eye.numpy())
            condition_hashes[rok] = ch

        return new_data, condition_hashes


@schema
class MonetResponse(dj.Computed, Messager):
    definition = """
    -> Encoder
    ---
    """

    class Block(dj.Part):
        definition = """
        -> master
        -> MonetData
        ---
        responses            : external-data # response of the model neuron on that trial
        """

    class Unit(dj.Part):
        definition = """
            -> master
            -> MovieScan.Unit
            ---
            row_id             : smallint       # row in the response
            """

    def make(self, key):
        self.msg('Computing responses')

        datasets, loaders = DataConfig().load_data(key)
        monet_data, condition_hashes = MonetData().load_data(key)

        model = Encoder().load_model(key).cuda()
        model.eval()

        self.insert1(key)
        for rok in datasets:
            self.msg('for', rok, depth=1, flush=True)

            dl = DataLoader(monet_data[rok], batch_size=1)
            responses = []
            for x, beh, eye in tqdm(to_variable(dl, filter=(True, True, True),
                                                cuda=True, volatile=True),
                                    desc='predictions'):
                y = model(x, rok, behavior=beh, eye_pos=eye)
                lag = x.shape[-3] - y.shape[1]

                y = y.cpu().data.numpy().squeeze()
                y = np.vstack((np.nan * np.ones((lag, y.shape[1])), y))  # fill up lag
                responses.append(y)

            member_key = (MovieMultiDataset.Member() & key & dict(name=rok)).fetch1(dj.key)  # get other fields
            member_key.update(key)
            unit_ids = datasets[rok].neurons.unit_ids

            for ch, r in tqdm(zip(condition_hashes[rok], responses), total=len(condition_hashes), desc='Inserting'):
                self.Block().insert1(dict(member_key,
                                          responses=r.astype(np.float32).T,
                                          condition_hash=ch), ignore_extra_fields=True)
            self.Unit().insert([dict(member_key, unit_id=ui, row_id=i) for i, ui in enumerate(unit_ids)],
                               ignore_extra_fields=True)


class MonetTrial:
    @staticmethod
    def _generate_monet_motion_trials(key):
        """
        generate dicts containing orientation trials from monet stimuli with
        onset and offset times and motion directions in radians.
        """
        monet = (stimulus.Monet2() * MonetData()).proj(
            'onsets', 'directions', 'speed', 'fps',
            ori_on_secs='round(1000*duration/n_dirs*ori_fraction)/1000')
        for onsets, directions, ori_duration, fps in zip(
                *(monet & key & 'speed>0').fetch(
                    'onsets', 'directions', 'ori_on_secs', 'fps', squeeze=True)):
            directions = np.round(directions / 180 * np.pi, decimals=4)
            for tup in zip(onsets, onsets + ori_duration, directions):
                yield dict(zip(('onset', 'offset', 'direction'), tup))


@schema
class MonetOri(dj.Computed, MonetTrial):
    definition = """
    # Monet orientation tuning predicted cell activity
    -> MonetResponse
    ori_type  : enum('ori','dir')    # orientation (180 degrees) or direction (360 degrees)
    ---
    latency      : float     # (s) screen-to-brain latency
    duration     : float     # (s) total duration of applicable trials
    """

    class Cell(dj.Part):
        definition = """
        -> master
        -> MonetResponse.Unit
        ----
        variance         : float    # total trace variance
        angle            : float  # (degrees) preferred orientation or direction
        selectivity      : float  # [0, 1]
        r2               : float  # fraction of variance explained
        """

    def make(self, key):
        # settings
        latency = 0.05
        trace_keys = (MonetResponse.Unit() & key).fetch("KEY", order_by='row_id')

        for ori_type, angle_multiplier in zip(('ori', 'dir'), (2, 1)):
            key['ori_type'] = ori_type

            designs, blocks = [], []
            for block_key in tqdm((MonetResponse.Block() & key).fetch("KEY"), desc="computing design matrix"):
                block = (MonetResponse.Block() & block_key).fetch1('responses')
                assert block.shape[0] == len(trace_keys)

                use = np.any(~np.isnan(block), axis=0)
                design = duration = 0.0
                fps = float((stimulus.Monet2() & block_key).fetch1('fps'))
                frame_times = np.arange(block.shape[1]) / fps

                for trial in self._generate_monet_motion_trials(block_key):
                    duration += trial['offset'] - trial['onset']
                    design += np.exp(angle_multiplier * 1j * trial['direction']) \
                              * (trial['onset'] + latency < frame_times) * (frame_times < trial['offset'] + latency)
                designs.append(design[use])
                blocks.append(block[:, use])
            traces = np.hstack(blocks)
            design = np.hstack(designs)

            # reduce analysis to periods of stimulus
            mag = abs(design) ** 2
            ix = mag.max(axis=tuple(range(design.ndim - 1))) > 1e-4 * mag.max()
            design = design[..., ix]
            traces = traces[..., ix]

            # remove orientation bias and normalize
            design -= design.mean(axis=-1, keepdims=True)  # remove orientation bias
            design /= np.sqrt((abs(design) ** 2).mean(axis=-1, keepdims=True))
            means = traces.mean(axis=-1)
            traces -= means[..., None]
            variances = (traces ** 2).mean(axis=-1)
            traces /= np.sqrt(variances[..., None])
            means /= np.sqrt(variances)

            # compute response
            xc = np.tensordot(traces, design, ([-1], [-1])) / design.shape[-1]
            self.insert1(dict(key, latency=latency, duration=duration))
            self.Cell().insert(
                dict(key, **trace_key, selectivity=abs(x) / m, variance=v,
                     angle=np.angle(x) / angle_multiplier, r2=abs(x) ** 2)
                for x, m, v, trace_key in zip(xc, means, variances, trace_keys))


@schema
class MonetCurve(dj.Computed, MonetTrial):
    definition = """
    # Monet orientation tuning predicted cell activity
    -> MonetResponse
    ---
    latency      : float     # (s) screen-to-brain latency
    duration     : float     # (s) total duration of applicable trials
    directions   : longblob  # directions presented
    n            : longblob  # number of sample points per direction
    """

    class Cell(dj.Part):
        definition = """
        -> master
        -> MonetResponse.Unit
        ----
        curve        : longblob
        std          : longblob
        """

    def make(self, key):
        # settings
        latency = 0.05
        trace_keys = (MonetResponse.Unit() & key).fetch("KEY", order_by='row_id')

        designs, blocks = [], []
        for block_key in tqdm((MonetResponse.Block() & key).fetch("KEY"), desc="computing design matrix"):
            block = (MonetResponse.Block() & block_key).fetch1('responses')
            assert block.shape[0] == len(trace_keys)

            use = np.any(~np.isnan(block), axis=0)
            design = duration = 0.0
            fps = float((stimulus.Monet2() & block_key).fetch1('fps'))
            frame_times = np.arange(block.shape[1]) / fps

            for trial in self._generate_monet_motion_trials(block_key):
                duration += trial['offset'] - trial['onset']
                design += trial['direction'] * (trial['onset'] + latency < frame_times) * (
                        frame_times < trial['offset'] + latency)
            designs.append(design[use])
            blocks.append(block[:, use])

        traces = np.hstack(blocks)
        design = np.hstack(designs)
        directions = np.unique(design)
        directions.sort()
        design = directions[:, None] == design[None, :]
        n = design.sum(axis=1, keepdims=True)
        design = design / n
        curves = np.tensordot(traces, design, ([-1], [-1]))
        std = np.sqrt(np.tensordot(traces ** 2, design, ([-1], [-1])) - curves ** 2)

        self.insert1(dict(key, latency=latency, duration=duration, directions=directions, n=n.squeeze()))
        self.Cell().insert(dict(key, **trace_key, curve=c, std=s) for c, s, trace_key in zip(curves, std, trace_keys))


@schema
class STA(dj.Computed):
    definition = """ # spike-triggered average receptive field maps
    -> MonetResponse
    ---
    nbins           : tinyint       # number of lags at which maps were calculated
    bin_size        : decimal(3,3)  # (secs) size of the bins.
    total_duration  : decimal(6,2)  # (secs) total duration of included trials
    vmax            : float         # correlation value of int8 level at 127
    """

    class Map(dj.Part):
        definition = """ # receptive field map at different lags

        -> master
        -> MonetResponse.Unit
        ---
        map         : external-data  # h x w x nbins
        """

    def make(self, key):
        # Set params
        num_lags = 5
        bin_size = 0.1
        vmax = 0.4
        cutoff_period = 2 * bin_size

        trace_keys = (MonetResponse.Unit() & key).fetch("KEY", order_by='row_id')
        trace_mean = np.zeros(len(trace_keys))
        trace_meansq = np.zeros(len(trace_keys))
        maps = 0  # num_traces x height x width x num_lags
        movie_mean = 0  # 1 x height x width x num_lags
        movie_meansq = 0  # 1 x height x width x num_lags

        total_duration = .0
        for block_key in tqdm((MonetResponse.Block() & key).fetch("KEY"), desc="computing STA"):
            block = (MonetResponse.Block() & block_key).fetch1('responses')
            movie = (MonetData & block_key).fetch1('frames')
            assert block.shape[0] == len(trace_keys)

            # get framerate
            fps = float((stimulus.Monet2() & block_key).fetch1('fps'))

            # get rid of nans because of model lag
            use = np.any(~np.isnan(block), axis=0)
            movie = movie[use, ...]
            block = block[..., use]

            # generate fake flip times and sample points
            flip_times = np.arange(block.shape[1]) / fps
            sample_secs = np.arange(flip_times[0], flip_times[-1], bin_size)

            # movie statistics and put movie time dimension at the end
            movie = movie.astype('float32').transpose([1, 2, 0]) / 127.5 - 1  # -1 to 1
            movie = interp1d(flip_times, movie)(sample_secs)

            duration = sample_secs[-1] - sample_secs[0] - (num_lags - 1) * bin_size
            total_duration += duration

            h = np.hamming(2 * int(fps * cutoff_period) + 1)
            block = np.apply_along_axis(np.convolve, 1, block, h, mode='same')
            block = interp1d(flip_times, block)(sample_secs)

            trace_mean += block.mean(axis=1) * duration
            trace_meansq += (block ** 2).mean(axis=1) * duration

            maps += STA.compute_sta(block, movie, num_lags) * duration

            ones = np.ones([1, len(sample_secs)])
            movie_mean += STA.compute_sta(ones, movie, num_lags) * duration
            movie_meansq += STA.compute_sta(ones, movie ** 2, num_lags) * duration

        # Normalize as correlation
        xy = maps / total_duration
        xx = movie_meansq / total_duration
        mx = movie_mean / total_duration
        yy = trace_meansq[:, None, None, None] / total_duration
        my = trace_mean[:, None, None, None] / total_duration
        g = (xy - mx * my) / np.sqrt((xx - mx ** 2) * (yy - my ** 2))
        g = np.clip(127 * g / vmax, -128, 127).astype('int8')

        # Insert
        self.insert1({**key, 'nbins': num_lags, 'bin_size': bin_size, 'vmax': vmax,
                      'total_duration': total_duration})
        self.Map().insert([{**key, **tk, 'map': rf} for tk, rf in zip(trace_keys, g)])

    @staticmethod
    def compute_sta(traces, movie, num_lags):
        """ Spike-triggered average at diff lags."""
        num_timepoints = movie.shape[-1] - (num_lags - 1)  # length of movie minus lag time
        weighted_sums = [np.tensordot(traces[..., lag:lag + num_timepoints], movie[..., :num_timepoints],
                                      axes=(-1, -1)) for lag in range(num_lags)]
        stas = np.stack(weighted_sums, -1) / num_timepoints  # num_traces x height x width x num_lags

        return stas


@schema
class STAQual(dj.Computed):
    definition = """

    -> STA.Map
    ---
    snr         : float  # RF contrast measurement
    """

    @property
    def key_source(self):
        return STA()

    def make(self, key):
        map_keys, maps = (STA.Map & key).fetch('KEY', 'map')
        snrs = [np.max(abs(map_[:, :, 1])) / map_.std() for map_ in maps]
        self.insert({**mk, 'snr': snr} for mk, snr in zip(map_keys, snrs))


@schema
class Ori(dj.Computed):
    definition = """
    # Orientation tuning for cells including monet and trippy conditions
    animal_id            : int                          # id number
    session              : smallint                     # session index for the mouse
    scan_idx             : smallint                     # number of TIFF stack file
    pipe_version         : smallint                     # 
    field                : tinyint                      # 
    channel              : tinyint                      # 
    segmentation_method  : tinyint                      # 
    spike_method         : tinyint                      # spike inference method
    stimulus_type        : varchar(30)                  # 
    ori_type             : enum('ori','dir')            # orientation (180 degrees) or direction (360 degrees)
    ---
    ori_version          : tinyint                      # in case variants must be compared
    latency              : float                        # (s) screen-to-brain latency
    duration             : float                        # (s) total duration of applicable trials
    """

    class Cell(dj.Part):
        definition = """
        -> master
        unit_id              : int                          # unique per scan & segmentation method
        ---
        variance             : float                        # total trace variance
        angle                : float                        # (degrees) preferred orientation or direction
        selectivity          : float                        # [0, 1]
        r2                   : float                        # fraction of variance explained
        """

    def make(self, key):
        pass


@schema
class DirCurve(dj.Computed):
    definition = """
    # Direction tuning curve for cells on monet
    animal_id            : int                          # id number
    session              : smallint                     # session index for the mouse
    scan_idx             : smallint                     # number of TIFF stack file
    pipe_version         : smallint                     # 
    field                : tinyint                      # 
    channel              : tinyint                      # 
    segmentation_method  : tinyint                      # 
    spike_method         : tinyint                      # spike inference method
    stimulus_type        : varchar(30)                  # 
    ---
    ori_version          : tinyint                      # in case variants must be compared
    directions           : longblob                     # base directions in radians
    samples              : longblob                     # sample points per condition
    latency              : float                        # (s) screen-to-brain latency
    duration             : float                        # (s) total duration of applicable trials    
    """

    class Cell(dj.Part):
        definition = """
        # 
        -> master
        unit_id              : int                          # unique per scan & segmentation method
        ---
        curve                : longblob                     # tuning curve
        std                  : longblob                     # std of the curve        
        """

    def make(self, key):
        pass


@schema
class NeuroSTA(dj.Computed):
    definition = """
    # Spike-triggered average receptive field maps
    animal_id            : int                          # id number
    session              : smallint                     # session index for the mouse
    scan_idx             : smallint                     # number of TIFF stack file
    pipe_version         : smallint                     # 
    field                : tinyint                      # 
    channel              : tinyint                      # 
    segmentation_method  : tinyint                      # 
    spike_method         : tinyint                      # spike inference method
    stimulus_type        : varchar(30)                  # 
    ---
    nbins                : tinyint                      # number of bins
    bin_size             : decimal(3,3)                 # (s)
    total_duration       : decimal(6,2)                 # total duration of included trials
    vmax                 : float                        # correlation value of int8 level at 127    
    """

    class Map(dj.Part):
        definition = """
        # receptive field map
        -> master
        unit_id              : int                          # unique per scan & segmentation method
        ---
        map                  : longblob                     # receptive field map
        """

    def make(self, key):
        pass


@schema
class NeuroSTAQual(dj.Computed):
    definition = """
    -> NeuroSTA.Map
    ---
    snr                  : float                        # RF contrast measurement    
    """

    def make(self, key):
        pass

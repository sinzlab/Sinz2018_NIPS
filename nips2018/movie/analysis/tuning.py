import numpy as np

import datajoint as dj
from nips2018.utils import ComputeStub
from ...utils.logging import Messager

schema = dj.schema('nips2018_analysis_tuning', locals())

dj.config['external-data'] = dict(
    protocol='file',
    location='/external/movie-analysis/')


@schema
class MonetData(dj.Computed, Messager, ComputeStub):
    definition = """
    -> stimulus.Monet2
    -> Preprocessing
    ---
    frames         : external-data
    """

    def make(self, key):
        pass


@schema
class MonetResponse(dj.Computed, Messager, ComputeStub):
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


@schema
class MonetOri(dj.Computed, ComputeStub):
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


@schema
class MonetCurve(dj.Computed, ComputeStub):
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


@schema
class STA(dj.Computed, ComputeStub):
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
class Ori(dj.Computed, ComputeStub):
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


@schema
class DirCurve(dj.Computed, ComputeStub):
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


@schema
class NeuroSTA(dj.Computed, ComputeStub):
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


@schema
class NeuroSTAQual(dj.Computed, ComputeStub):
    definition = """
    -> NeuroSTA.Map
    ---
    snr                  : float                        # RF contrast measurement    
    """

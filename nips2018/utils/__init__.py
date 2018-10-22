import functools

import numpy as np
import torch
import random



def set_seed(seed, cuda=True):
    print('Setting numpy and torch seed to', seed, flush=True)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(int(seed))
    if cuda:
        torch.cuda.manual_seed(int(seed))

def rename(rel, prefix='new_', exclude=[]):
    attrs = list(rel.heading.attributes.keys())
    original = [x for x in attrs if x not in exclude]
    name_map = {prefix+x: x for x in original}
    return rel.proj(**name_map)


class ComputeStub:

    def make(selfk, key):
        pass
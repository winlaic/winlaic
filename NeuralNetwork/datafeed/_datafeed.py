import torch
import numpy as np
__all__ = [
    'namedtuple_collate',
]
def namedtuple_collate(batch):
    transposed = list(map(list, zip(*batch)))
    ret = []
    for col in transposed:
        if isinstance(col[0], torch.Tensor):
            ret.append(torch.stack(col, dim=0))
        else:
            ret.append(col)
    return batch[0].__class__(*col)
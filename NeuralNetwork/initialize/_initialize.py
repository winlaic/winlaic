import torch
import torch.nn as nn
import random
import numpy as np

__all__ = [
    'deterministic', 'general_initialize'
]

def deterministic(torch_seed=1, numpy_seed=1, python_seed=1):
    random.seed(python_seed)
    np.random.seed(numpy_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(torch_seed)

def general_initialize(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
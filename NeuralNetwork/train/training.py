import torch
import torch.nn as nn

__all__ = [
    'adjust_hyperparams'
]

def adjust_hyperparams(param_groups: list, model: nn.Module, weight_lr_mult=1.0, bias_lr_mult=2.0, weight_decay_mult=1.0, bias_decay_mult=0):
    """Adjust learning rates by multiplying a constant.
    """
    named_parameters = model.named_parameters()
    ret = []
    for group in param_groups:
        weights = []; biases = []; others = []
        for param in group['params']:
            for param_name, param_to_be_matched in named_parameters:
                if param is param_to_be_matched:
                    if param_name.endswith('.weight'):
                        weights.append(param)
                    elif param_name.endswith('.bias'):
                        biases.append(param)
                    else:
                        others.append(param)
                    break
                raise Exception('Not all parameters is in given model.')
        new_groups = []
        if len(weights) > 0:
            weights_group = {'params': weights, 'lr': group['lr'] * weight_lr_mult, 'weight_decay': group['weight_decay'] * weight_decay_mult}
            new_groups.append(weights_group)
        if len(biases) > 0:
            biases_group = {'params': biases, 'lr': group['lr'] * bias_lr_mult, 'weight_decay': group['weight_decay'] * bias_decay_mult}
            new_groups.append(biases_group)
        if len(others) > 0: 
            others_group = {'params': others}
            new_groups.append(others_group)

        for g in new_groups:
            for k in group:
                if k not in g:
                    g[k] = group[k]
        ret += new_groups
    return ret
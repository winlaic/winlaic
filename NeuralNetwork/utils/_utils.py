import os
import torch
import typing
import numpy as np
import re
__all__ = [
    'start_tensorboard',
    'get_avaliable_devices',
    'Averager',
    'ParameterManager'
]

def start_tensorboard(log_dir, tensorboard='tensorboard'):
    ret_kill = os.system('pkill tensorboard')
    command = tensorboard + ' --logdir=' + log_dir + ' > /dev/null 2>&1'
    # if ret_kill:
    #     print('Failed to kill previous TensorBoard process.')
    #     print('You may need to manually launch it.')
    #     print_command('TensorBoard Command', command)
    ret_open = os.popen(command)

def get_avaliable_devices():
    n_cuda_device = torch.cuda.device_count()
    if n_cuda_device > 0:
        return [torch.device('cuda', i) for i in range(n_cuda_device)]
    else:
        return [torch.device('cpu')]

class Averager:
    def __init__(self):
        self.sum = 0
        self.count = 0
    def clear(self):
        self.__init__()

    def __iadd__(self, value):
        self.sum += float(value)
        self.count += 1
        return self

    @property
    def mean(self):
        return self.sum / self.count if self.count != 0 else 0.0


class ParameterManager:

    def __init__(self, named_parameters, train_parameters=None) -> None:
        if isinstance(named_parameters, typing.Mapping):
            self.__dict__['_named_parameters'] = {k: v for k, v in named_parameters.items()}
        else:
            self.__dict__['_named_parameters'] = {k: v for k, v in named_parameters}
        self.__dict__['_train_parameters'] = {k: {} for k in
                                              self._named_parameters} if train_parameters is None else train_parameters

    def __repr__(self):
        ret = ''
        max_len_key = max(len(item) for item in self._named_parameters.keys())
        max_len_shape = max(len(str(tuple(item.shape))) for item in self._named_parameters.values())
        max_len_params = max(len(str(item)) for item in self._train_parameters.values())
        n_param = 0
        ret += '-' * (3 + max_len_key + max_len_shape + max_len_params + 9) + '\n'
        for i, key in enumerate(self._named_parameters):
            ret += str(i).rjust(3) + ' | '
            ret += str(key).ljust(max_len_key) + ' | '
            param_shape = tuple(self._named_parameters[key].shape)
            n_param += np.prod(param_shape)
            ret += str(param_shape).ljust(max_len_shape) + ' | '
            ret += str(self._train_parameters[key]).ljust(max_len_params)
            ret += '\n'
        ret += '-' * (3 + max_len_key + max_len_shape + max_len_params + 9) + '\n'
        ret += 'Number of parameter: {:.3e}'.format(float(n_param)) + '\n'
        ret += '-' * (3 + max_len_key + max_len_shape + max_len_params + 9) + '\n'
        return ret[:-1]

    def __sub__(self, other):
        assert isinstance(other, self.__class__)
        matched_keys = [key for key in self._named_parameters.keys() if key not in other._named_parameters.keys()]
        return ParameterManager(
            {k: v for k, v in self._named_parameters.items() if k in matched_keys},
            {k: v for k, v in self._train_parameters.items() if k in matched_keys}
        )

    def __add__(self, other):
        assert isinstance(other, self.__class__)
        
        named_parameters = {}
        named_parameters.update(self._named_parameters)
        named_parameters.update(other._named_parameters)

        train_parameters = {}
        train_parameters.update(self._train_parameters)
        train_parameters.update(other._train_parameters)

        return ParameterManager(named_parameters, train_parameters)

    def __getitem__(self, item):
        matched_keys = []
        if isinstance(item, str):
            for key in self._named_parameters:
                if re.match('^' + item, key) is not None:
                    matched_keys.append(key)
        elif isinstance(item, slice):
            matched_keys += list(self._named_parameters.keys())[item]
        elif isinstance(item, int):
            matched_keys.append(list(self._named_parameters.keys())[item])
        else:
            raise TypeError('Index type must be str, slice or int.')

        return ParameterManager(
            {k: v for k, v in self._named_parameters.items() if k in matched_keys},
            {k: v for k, v in self._train_parameters.items() if k in matched_keys}
        )

    def __setattr__(self, key, value):
        if key not in self.__dict__:
            assert key != 'params', 'Cannot specify params!'
            for k in self._train_parameters:
                self._train_parameters[k][key] = value
        else:
            super().__setattr__(key, value)

    def update_optimizer(self, optimizer, **kwargs):
        assert isinstance(optimizer, torch.optim.Optimizer)
        param_groups = self.parse(**kwargs)
        optimizer.param_groups.clear()
        for param_group in param_groups:
            optimizer.add_param_group(param_group)

    def parse(self, squeeze_zero_lr=False):
        unique_train_params = []
        param_groups = []
        for k in self._named_parameters:
            train_param = self._train_parameters[k]
            if squeeze_zero_lr:
                # Skip parameters whose lr is zero and drop them from grad map.
                if 'lr' in train_param and train_param['lr'] == 0.0:
                    self._named_parameters[k].requires_grad = False
                    continue
            if train_param not in unique_train_params:
                unique_train_params.append(train_param)
                param_group = {'params': [self._named_parameters[k]]}
                param_group.update(self._train_parameters[k])
                param_groups.append(param_group)
            else:
                index = unique_train_params.index(train_param)
                param_groups[index]['params'].append(self._named_parameters[k])
        return param_groups

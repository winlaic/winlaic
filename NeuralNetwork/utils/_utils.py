import os
import torch

__all__ = [
    'start_tensorboard',
    'get_avaliable_devices',
    'Averager'
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
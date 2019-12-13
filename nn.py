import torch
import torch.nn as nn
import numpy as np
import math
import random
from .utils import Averager, print_command
from .fs import ensuredir
from collections import deque
import os

"""
Note: In pytorch, epoch starts from 0.
In init process with "last_epoch = -1", 
learning rate schedular will save all base_learning rate,
set self.last_epoch to 0.
However, the calculation of learning rate is based on self.last_epoch.
"""

class StepLRWithWarmingUp(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, step_size, T_warm_up, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        self.T_warm_up = T_warm_up
        super().__init__(optimizer, last_epoch)
    def get_single_lr(self, base_lr):
        # Always use "<" operator in case that T_warm_up is set to 0.
        if self.last_epoch < self.T_warm_up:
            return base_lr / (self.T_warm_up + 1) * (self.last_epoch + 1)
        else:
            return base_lr * self.gamma ** ((self.last_epoch - self.T_warm_up) // self.step_size)
    def get_lr(self):
        return [self.get_single_lr(base_lr) for base_lr in self.base_lrs]

class ExponentialLRWithWarmingUp(torch.optim.lr_scheduler._LRScheduler):
    
    def __init__(self, optimizer, gamma, T_warm_up, last_epoch=-1):
        self.gamma = gamma
        self.T_warm_up = T_warm_up
        super().__init__(optimizer, last_epoch)

    @staticmethod
    def half_when(epoch):
        return 0.5 ** (1 / epoch)

    @staticmethod
    def deci_when(epoch):
        return 0.1 ** (1 / epoch)
    
    def get_single_lr(self, base_lr):
        # Always use "<" operator in case that T_warm_up is set to 0.
        if self.last_epoch < self.T_warm_up:
            return base_lr / (self.T_warm_up + 1) * (self.last_epoch + 1)
        else:
            return base_lr * self.gamma ** (self.last_epoch - self.T_warm_up)
    
    def get_lr(self):
        return [self.get_single_lr(base_lr) for base_lr in self.base_lrs]

class CosineAnnealingLRWithWarmingUp(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_warm_up, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.T_warm_up = T_warm_up
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_single_lr(self, base_lr):
        # Always use "<" operator in case that T_warm_up is set to 0.
        if self.last_epoch < self.T_warm_up:
            return base_lr / (self.T_warm_up + 1) * (self.last_epoch + 1)
        else:
            return self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * (self.last_epoch - self.T_warm_up) / self.T_max)) / 2

    def get_lr(self):
        return [self.get_single_lr(base_lr) for base_lr in self.base_lrs]
    

def general_initialize(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def start_tensorboard(log_dir, tensorboard='tensorboard'):
    ret_kill = os.system('pkill tensorboard')
    command = tensorboard + ' --logdir=' + log_dir + ' > /dev/null 2>&1'
    # if ret_kill:
    #     print('Failed to kill previous TensorBoard process.')
    #     print('You may need to manually launch it.')
    #     print_command('TensorBoard Command', command)
    ret_open = os.popen(command)

def reproducible(random_seed=1):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)



class Trainer:
    r"""Trainer base class.

    You should at least implement method "criterion()".
    
    Args: 
        net (torch.nn.Module): Network to train.
        optimizer (torch.optim.Optimizer): The optimizer.
        dataloader (torch.utils.data.DataLoader): The dataloader.
        using_tqdm (bool): When called, return a tqdm handler to show progress.
        loss_smooth (int): When print loss, implement mean smooth with window-size of specfied.
    """

    def __init__(self, 
        net: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        dataloader: torch.utils.data.DataLoader,
        schedular: torch.optim.lr_scheduler._LRScheduler = None,
        using_tqdm: bool = False,
        tqdm_report_period: int = 10):
    
        self.net = net
        self.mode = None
        self.optimizer = optimizer
        self.schedular = schedular
        self.dataloader = dataloader
        self.loss_collector = deque()
        self.loss_reporter = deque(maxlen=tqdm_report_period)
        if using_tqdm:
            from tqdm import tqdm
            self.tqdm = tqdm
        else:
            self.tqdm = None

    def resume(self, n, checkpoint):
        self.__load_checkpoint(checkpoint)
        self.__n = n
        if self.tqdm is not None:
            self.tqdm_progress = self.tqdm(self, initial=self.__i)
            return self.tqdm_progress
        else:
            return self

    def epoch(self, n):
        self.mode = 'epoch'
        self.__i = 0; self.__n = n
        if self.tqdm is not None:
            self.tqdm_progress = self.tqdm(self)
            return self.tqdm_progress
        else:
            return self
    
    def iter(self, n):
        self.mode = 'iter'
        self.__i = 0; self.__n = n
        if self.tqdm is not None:
            self.tqdm_progress = self.tqdm(self)
            return self.tqdm_progress
        else:
            return self

    @property
    def progress(self):
        return self.__i
    
    def save_checkpoint(self, path):
        checkpoint = {}
        checkpoint['mode'] = self.mode
        checkpoint['optimizer'] = self.optimizer.state_dict()
        checkpoint['net'] = self.net.state_dict()
        checkpoint['progress'] = self.__i
        if self.schedular is not None: 
            checkpoint['schedular'] = self.schedular.state_dict()
        torch.save(checkpoint, path)

    def __load_checkpoint(self, checkpoint):
        if isinstance(checkpoint, str):
            checkpoint = torch.load(checkpoint)
        self.mode = checkpoint['mode']
        self.net.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'schedular' in checkpoint: 
            self.schedular.load_state_dict(checkpoint['schedular'])
        self.__i = checkpoint['progress']
        
    def __len__(self):
        return self.__n
    

    def __iter__(self):
        loss_report = 0.0
        if self.mode == 'epoch':
            while self.__i < self.__n:
                self.net.train()
                self.loss_collector.clear()
                for i_batch, sequence in enumerate(self.dataloader):
                    self.optimizer.zero_grad()
                    loss = self.criterion(self.net, sequence)
                    loss_plain = loss.tolist()
                    self.loss_collector.append(loss_plain)
                    if self.tqdm is not None:
                        lr_groups = [item['lr'] for item in self.optimizer.param_groups]
                        lr_str = '[' + ' '.join(map(lambda x:'{:.2e}'.format(x), lr_groups)) +']'
                        self.loss_reporter.append(loss_plain)
                        if len(self.loss_reporter) == self.loss_reporter.maxlen:
                            loss_report = sum(self.loss_reporter) / len(self.loss_reporter)
                            self.loss_reporter.clear()
                        self.tqdm_progress.set_description(
                            'LOSS: %.3e LR: %s PROG: %d/%d %.0f%%' % (loss_report, lr_str, i_batch, len(self.dataloader), i_batch / len(self.dataloader) * 100.0))
                    loss.backward()
                    self.optimizer.step()
                if self.schedular is not None:
                    self.schedular.step()
                self.__i += 1
                yield sum(self.loss_collector) / len(self.loss_collector)
            raise StopIteration
        elif self.mode == 'iter':
            progress = iter(self.dataloader)
            while self.__i < self.__n:
                self.net.train()
                try: sequence = next(progress)
                except StopIteration: 
                    progress = iter(self.dataloader)
                    sequence = next(progress)
                self.optimizer.zero_grad()
                loss = self.criterion(self.net, sequence)
                loss_plain = loss.tolist()
                if self.tqdm is not None:
                    self.loss_reporter.append(loss_plain)
                    if len(self.loss_reporter) == self.loss_reporter.maxlen:
                        loss_report = sum(self.loss_reporter) / len(self.loss_reporter)
                        self.loss_reporter.clear()
                        self.tqdm_progress.set_description('LOSS: %e' % loss_report)
                loss.backward()
                self.optimizer.step()
                if self.schedular is not None:
                    self.schedular.step()
                self.__i += 1
                yield loss_plain
            raise StopIteration


    def criterion(self, net, sequence):
        # TODO: Convert output to loss scalar and return it.
        raise NotImplementedError

Kunrensya = Trainer

# class Trainer:
#     r"""Trainer base class.

#     You should at least implement method "criterion()".
    
#     Args: 
#         net (torch.nn.Module): Network to train.
#         optimizer (torch.optim.Optimizer): The optimizer.
#         dataloader (torch.utils.data.DataLoader): The dataloader.
#         using_tqdm (bool): When called, return a tqdm handler to show progress.
#         loss_smooth (int): When print loss, implement mean smooth with window-size of specfied.
#     """
#     def __init__(self, 
#         net: torch.nn.Module, 
#         optimizer: torch.optim.Optimizer, 
#         dataloader: torch.utils.data.DataLoader,
#         schedular: torch.optim.lr_scheduler._LRScheduler = None,
#         using_tqdm: bool = False, 
#         loss_smooth: int = 1):
    
#         self.net = net
#         self.optimizer = optimizer
#         self.schedular = schedular
#         self.dataloader = dataloader
#         self.loss_collector = Averager()
#         self.loss_smoother = deque(maxlen=loss_smooth)
#         if using_tqdm:
#             from tqdm import tqdm
#             self.tqdm = tqdm
#         else:
#             self.tqdm = None

#     @property
#     def epoch(self):
#         return self.__i_epoch

#     def save_checkpoint(self, path):
#         checkpoint = {}
#         checkpoint['optimizer'] = self.optimizer.state_dict()
#         checkpoint['net'] = self.net.state_dict()
#         checkpoint['epoch'] = self.epoch
#         if self.schedular is not None: 
#             checkpoint['schedular'] = self.schedular.state_dict()
#         torch.save(checkpoint, path)

#     def __load_checkpoint(self, checkpoint):
#         if isinstance(checkpoint, str):
#             checkpoint = torch.load(checkpoint)
#         self.net.load_state_dict(checkpoint['net'])
#         self.optimizer.load_state_dict(checkpoint['optimizer'])
#         if 'schedular' in checkpoint: 
#             self.schedular.load_state_dict(checkpoint['schedular'])
#         self.__i_epoch = checkpoint['epoch']


#     def __call__(self, n_epoch, checkpoint=None):

#         if checkpoint is None:
#             self.__i_epoch = 0
#         else:
#             self.__load_checkpoint(checkpoint)
#         self.__n_epoch = n_epoch
#         if self.tqdm is not None:
#             self.tqdm_progress = self.tqdm(self, initial=self.__i_epoch)
#             return self.tqdm_progress
#         else:
#             return self

#     def __len__(self):
#         return self.__n_epoch
    
#     def step(self):
#         '''
#         Define how to update parameters of network.
#         Implement if needed.
#         '''
#         self.optimizer.step()

#     def __iter__(self):
#         while self.__i_epoch < self.__n_epoch:
#             self.net.train()
#             self.loss_collector.clear()
#             for i_batch, sequence in enumerate(self.dataloader):
#                 self.optimizer.zero_grad()
#                 loss = self.criterion(self.net, sequence)
#                 loss_plain = loss.tolist()
#                 self.loss_collector += loss_plain
#                 if self.tqdm is not None:
#                     self.loss_smoother.append(loss_plain)
#                     self.tqdm_progress.set_description('LOSS: %e PROG: %d/%d %.1f%%' % (sum(self.loss_smoother)/len(self.loss_smoother), i_batch, len(self.dataloader), i_batch / len(self.dataloader) * 100.0))
#                 loss.backward()
#                 self.step()
#             if self.schedular is not None:
#                 self.schedular.step()
#             self.__i_epoch += 1
#             yield self.loss_collector.mean
#         raise StopIteration

#     def criterion(self, net, sequence):
#         # TODO: Convert output to loss scalar and return it.
#         raise NotImplementedError

def adjust_hyperparams(param_groups: list, model: nn.Module, weight_lr_mult=1.0, bias_lr_mult=2.0, weight_decay_mult=1.0, bias_decay_mult=0):
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


def get_avaliable_devices():
    n_cuda_device = torch.cuda.device_count()
    if n_cuda_device > 0:
        return [torch.device('cuda', i) for i in range(n_cuda_device)]
    else:
        return [torch.device('cpu')]


class Saver:
    def __init__(self, trainer, time_stamp, logger=None, save_dir='models', default_best_module_name='BEST', period_checkpoint=None):
        self.period_checkpoint = period_checkpoint
        self.save_dir = save_dir
        self.trainer = trainer
        self.time_stamp = time_stamp
        self.last_bests = dict()
        self.logger = logger

    def save_checkpoint(self):
        if self.period_checkpoint is not None:
            if self.trainer.progress % self.period_checkpoint == 0:
                save_path = ensuredir(self.save_dir, self.time_stamp, 
                        file_name='EPOCH_{}.pkl'.format(self.trainer.progress))
                self.trainer.save_checkpoint(save_path)
                if self.logger is not None:
                    self.logger.i = 'EPOCH: {}, Checkpoint saved at "{}".'.format(self.trainer.progress, save_path)
        else:
            warn_info = 'Checkpoint save period is not specified. Saving is ignored!'
            if self.logger is not None:
                self.logger.e = warn_info
            else: 
                print(warn_info)

    def save_maximum(self, criterion_value, criterion_name=''):
        self.save_best(criterion_value=criterion_value, mode='maximum', criterion_name=criterion_name)
    
    def save_minimum(self, criterion_value, criterion_name=''):
        self.save_best(criterion_value=criterion_value, mode='minimum', criterion_name=criterion_name)    
    
    def save_best(self, criterion_value, mode, criterion_name=''):
        if (criterion_name not in self.last_bests) or \
            (mode == 'maximum' and criterion_value >= self.last_bests[criterion_name]) or \
            (mode == 'minimum' and criterion_value <= self.last_bests[criterion_name]):
                self.last_bests[criterion_name] = criterion_value
                save_path = ensuredir(self.save_dir, self.time_stamp, file_name='BEST_{}.pkl'.format(criterion_name))
                self.trainer.save_checkpoint(save_path)
                if self.logger is not None:
                    self.logger.w = 'EPOCH: {}, {} achieved {:.3f}, succeed last best value. Model saved at "{}".'.format(
                        self.trainer.progress, criterion_name, criterion_value, save_path)
                
                
            

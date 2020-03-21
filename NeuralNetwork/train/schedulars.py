import torch
import math

__all__ = [
    'StepLRWithWarmingUp',
    'ExponentialLRWithWarmingUp',
    'CosineAnnealingLRWithWarmingUp',
]

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
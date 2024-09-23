import torch
import math

class LinearLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, start_epoch, end_epoch, start_lr, end_lr, last_epoch=-1):
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.num_iterations = self.end_epoch - self.start_epoch
        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.end_epoch and self.last_epoch >= self.start_epoch:
            return [self.start_lr + self.last_epoch * ((self.end_lr - self.start_lr) / self.num_iterations)]
        else:
            return [group['lr'] for group in self.optimizer.param_groups]

class CosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, start_epoch, end_epoch, start_lr, end_lr, last_epoch=-1):
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.start_lr = start_lr
        self.end_lr = end_lr
        super(CosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.end_epoch and self.last_epoch >= self.start_epoch:
            return [.5 * (self.start_lr - self.end_lr) * math.cos(self.last_epoch * (math.pi / (self.end_epoch - self.start_epoch)) - self.start_epoch * (math.pi / (self.end_epoch - self.start_epoch))) + .5 * (self.start_lr - self.end_lr) + self.end_lr]
        else:
            return [group['lr'] for group in self.optimizer.param_groups]

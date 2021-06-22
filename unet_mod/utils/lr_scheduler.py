import math


class ClosedFormCosineLRScheduler(object):
    def __init__(self, optimizer, init_lr, total_steps, resume_step=0, end_lr=0.):
        assert init_lr>end_lr and end_lr>=0
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.end_lr = end_lr
        self.total_steps = total_steps
        self.lr_list = [(init_lr-end_lr) * 0.5 * (1. + math.cos(math.pi * step / total_steps)) for step in range(total_steps)]
        self._step = resume_step
    
    def step(self, current_step=None):
        param_groups = self.optimizer.param_groups
        if current_step is None:
            current_lr = self.lr_list[self._step]
            self._step += 1
        else:
            current_lr = self.lr_list[current_step]
        for param_group in param_groups:
            param_group['lr'] = current_lr

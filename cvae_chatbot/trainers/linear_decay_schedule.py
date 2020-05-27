# -*- coding: utf-8 -*-
# @Time        : 2020/5/23 23:37
# @Author      : ssxy00
# @File        : linear_decay_schedule.py
# @Description :

class LinearDecaySchedule:
    def __init__(self, total_steps, max_lr, optimizer):
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.optimizer = optimizer
        self._step = 0

    def state_dict(self):
        return {'step': self._step,
                'optimizer': self.optimizer.state_dict()}

    def load_state_dict(self, state_dict):
        self._step = state_dict['step']
        self.optimizer.load_state_dict(state_dict['optimizer'])

    def zero_grad(self):
        return self.optimizer.zero_grad()

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.max_lr * (1 - step / self.total_steps)

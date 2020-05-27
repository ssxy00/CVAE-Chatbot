# -*- coding: utf-8 -*-
# @Time        : 2020/5/24 13:39
# @Author      : ssxy00
# @File        : bow_module.py
# @Description :

import torch.nn as nn
import torch.nn.functional as F

class BOWModule(nn.Module):
    """predict bow of response sequence"""
    def __init__(self, in_dim, out_dim, mid_dim=400):
        super(BOWModule, self).__init__()
        self.linear_1 = nn.Linear(in_dim, mid_dim)
        self.activation = F.relu
        self.linear_2 = nn.Linear(mid_dim, out_dim)

    def forward(self, x):
        x = self.linear_2(self.activation(self.linear_1(x)))
        return x


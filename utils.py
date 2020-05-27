# -*- coding: utf-8 -*-
# @Time        : 2020/5/23 14:19
# @Author      : ssxy00
# @File        : utils.py
# @Description : training utils

import random
import torch


def set_seed(seed):
    """set random seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

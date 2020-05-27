# -*- coding: utf-8 -*-
# @Time        : 2020/5/24 01:58
# @Author      : ssxy00
# @File        : model_utils.py
# @Description : modeling utils

import torch

def compute_KL(recognition_mu, recognition_logvar, prior_mu, prior_logvar):
    """compute KL of two gaussian distributions"""
    kl = -0.5 * (1 + (recognition_logvar - prior_logvar) - torch.exp(recognition_logvar - prior_logvar)
                 - torch.pow((recognition_mu - prior_mu), 2) / torch.exp(prior_logvar))
    kl = torch.sum(kl, dim=1)
    return torch.mean(kl)

def sample_z(mu, logvar):
    """sample from gaussian distribution"""
    random_sample = torch.normal(mean=0., std=1., size=logvar.shape, device=mu.device)
    random_sample = mu + random_sample * torch.exp(0.5 * logvar)
    return random_sample


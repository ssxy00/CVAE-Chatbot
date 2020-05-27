# -*- coding: utf-8 -*-
# @Time        : 2020/5/24 01:53
# @Author      : ssxy00
# @File        : prior_module.py
# @Description : prior network

import torch
import torch.nn as nn

class PriorModule(nn.Module):
    """(p, c) -> z"""
    def __init__(self, embed_dim, z_dim):
        super(PriorModule, self).__init__()
        self.z_dim = z_dim
        self.linear = nn.Linear(embed_dim * 2, z_dim * 2)

    def forward(self, persona_embedding, context_embedding):
        hidden_states = torch.cat([persona_embedding, context_embedding], dim=1)
        hidden_states = self.linear(hidden_states)
        mu, logvar = hidden_states.split(self.z_dim, dim=1)
        return mu, logvar
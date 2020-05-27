# -*- coding: utf-8 -*-
# @Time        : 2020/5/24 00:25
# @Author      : ssxy00
# @File        : sent_embedding_module.py
# @Description :

import torch
import torch.nn as nn

class SentEmbeddingModule(nn.Module):
    """compute sequence embedding"""
    def __init__(self, core_module):
        super(SentEmbeddingModule, self).__init__()
        self.core_module = core_module

    def forward(self, input_ids, cls_positions):
        hidden_states = self.core_module(input_ids=input_ids)[0]
        indices_offset = torch.arange(start=0, end=hidden_states.shape[0], device=hidden_states.device) * hidden_states.shape[1]
        indices = indices_offset + cls_positions
        sequence_embeddings = hidden_states.view(-1, hidden_states.shape[-1]).index_select(dim=0, index=indices)
        return sequence_embeddings

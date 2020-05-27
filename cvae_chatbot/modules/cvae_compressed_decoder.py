# -*- coding: utf-8 -*-
# @Time        : 2020/5/25 20:40
# @Author      : ssxy00
# @File        : cvae_compressed_decoder.py
# @Description :


import torch
import torch.nn as nn


class CVAECompressedDecoder(nn.Module):
    """decoder for CVAECompressedModel, reconstruct response"""

    def __init__(self, core_module, pad_id, z_dim, embed_dim):
        super(CVAECompressedDecoder, self).__init__()
        self.core_module = core_module
        self.pad_id = pad_id
        self.z_dim = z_dim
        self.embed_dim = embed_dim
        self.n_layer = core_module.config.n_layer

        # initialize and tile lm head
        self.lm_head = nn.Linear(embed_dim, core_module.wte.weight.size(0), bias=False)
        self.lm_head.weight = core_module.wte.weight

        # initialize latent head
        self.latent_head = nn.Linear(z_dim, embed_dim * self.n_layer, bias=False)

    def forward(self, persona_embedding, context_embedding, input_ids, latent_sample=None):
        if latent_sample is None:
            # compressed_decoder
            extra_hidden_states = [torch.cat([persona_embedding.unsqueeze(1), context_embedding.unsqueeze(1)],
                                             dim=1)] * self.n_layer
        else:
            # compressed_cvae
            extra_hidden_states = self.latent_head(latent_sample)
            extra_hidden_states = [h.unsqueeze(1) for h in extra_hidden_states.split(self.embed_dim, dim=1)]
            extra_hidden_states = [torch.cat([persona_embedding.unsqueeze(1), context_embedding.unsqueeze(1),
                                              z_embedding], dim=1) for z_embedding in extra_hidden_states]
        hidden_states = self.core_module(input_ids=input_ids, extra_hidden_states=extra_hidden_states)[0]
        logits = self.lm_head(hidden_states)
        return logits

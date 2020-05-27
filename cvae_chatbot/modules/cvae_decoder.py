# -*- coding: utf-8 -*-
# @Time        : 2020/5/24 02:43
# @Author      : ssxy00
# @File        : cvae_decoder.py
# @Description :

import torch.nn as nn


class CVAEDecoder(nn.Module):
    """decoder, reconstruct response"""

    def __init__(self, core_module, pad_id, z_dim, embed_dim, inject_type):
        super(CVAEDecoder, self).__init__()
        self.inject_type = inject_type
        self.core_module = core_module
        self.pad_id = pad_id
        self.z_dim = z_dim
        self.embed_dim = embed_dim

        # initialize and tie lm head
        self.lm_head = nn.Linear(embed_dim, core_module.wte.weight.size(0), bias=False)
        self.lm_head.weight = core_module.wte.weight

        # initialize latent head
        if inject_type == "memory":
            self.latent_head = nn.Linear(z_dim, embed_dim * core_module.config.n_layer, bias=False)
        if inject_type == "embedding":
            self.latent_head = nn.Linear(z_dim, embed_dim, bias=False)

    def forward(self, input_ids, type_ids, latent_sample=None):
        if latent_sample is None:
            # decoder
            hidden_states = self.core_module(input_ids=input_ids, token_type_ids=type_ids)[0]
        else:
            if self.inject_type == "memory":
                # cvae_memory
                extra_hidden_states = self.latent_head(latent_sample)
                extra_hidden_states = [h.unsqueeze(1) for h in extra_hidden_states.split(self.embed_dim, dim=1)]
                hidden_states = self.core_module(input_ids=input_ids, token_type_ids=type_ids,
                                                 extra_hidden_states=extra_hidden_states)[0]
            elif self.inject_type == "embedding":
                # cvae embedding
                extra_embedding = self.latent_head(latent_sample).unsqueeze(1)
                hidden_states = self.core_module(input_ids=input_ids, token_type_ids=type_ids,
                                                 extra_embedding=extra_embedding)[0]
            else:
                raise ValueError("unknown injection type")

        logits = self.lm_head(hidden_states)
        return logits

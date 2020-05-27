# -*- coding: utf-8 -*-
# @Time        : 2020/5/24 00:08
# @Author      : ssxy00
# @File        : cvae_model.py
# @Description :

import torch
import torch.nn as nn

from cvae_chatbot.modules.sent_embedding_module import SentEmbeddingModule
from cvae_chatbot.modules.recognition_module import RecognitionModule
from cvae_chatbot.modules.prior_module import PriorModule
from cvae_chatbot.modules.cvae_decoder import CVAEDecoder
from cvae_chatbot.modules.bow_module import BOWModule
from cvae_chatbot.models.model_utils import compute_KL, sample_z


class CVAEModel(nn.Module):
    def __init__(self, core_module, pad_id, z_dim, model_type, bow=False):
        super(CVAEModel, self).__init__()
        # parse model_type
        if model_type == "decoder":
            self.cvae = False
            self.inject_type = None
        elif model_type == "cvae_memory":
            self.cvae = True
            self.inject_type = "memory"
        elif model_type == "cvae_embedding":
            self.cvae = True
            self.inject_type = "embedding"
        else:
            raise ValueError("Unknown model type!")
        self.pad_id = pad_id
        self.embed_dim = core_module.config.n_embd
        self.sent_embedding_module = SentEmbeddingModule(core_module)
        self.recognition_module = RecognitionModule(embed_dim=self.embed_dim, z_dim=z_dim)  # (p, c, x) -> z
        self.prior_module = PriorModule(embed_dim=self.embed_dim, z_dim=z_dim)  # (p, c) -> z
        self.decoder = CVAEDecoder(core_module=core_module, pad_id=pad_id, z_dim=z_dim, embed_dim=self.embed_dim,
                                   inject_type=self.inject_type)
        self.bow = bow
        if bow:
            self.bow_module = BOWModule(in_dim=self.embed_dim + z_dim, out_dim=core_module.wte.weight.size(0))
            # bow loss
            self.bow_criterion = nn.CrossEntropyLoss(ignore_index=self.pad_id)
        # reconstruction loss
        self.seq_criterion = nn.CrossEntropyLoss(ignore_index=self.pad_id)

    def forward(self, input_ids, type_ids, labels, context, persona, response, context_cls_position,
                persona_cls_position, response_cls_position):
        """forward function during training"""
        if self.cvae:
            # embedding
            persona_embedding = self.sent_embedding_module(input_ids=persona, cls_positions=persona_cls_position)
            context_embedding = self.sent_embedding_module(input_ids=context, cls_positions=context_cls_position)
            response_embedding = self.sent_embedding_module(input_ids=response, cls_positions=response_cls_position)

            # recognition
            recognition_mu, recognition_logvar = self.recognition_module(persona_embedding=persona_embedding,
                                                                         context_embedding=context_embedding,
                                                                         response_embedding=response_embedding)
            # prior
            prior_mu, prior_logvar = self.prior_module(persona_embedding=persona_embedding,
                                                       context_embedding=context_embedding)
            kl_loss = compute_KL(recognition_mu=recognition_mu, recognition_logvar=recognition_logvar,
                                 prior_mu=prior_mu, prior_logvar=prior_logvar)

            # sample z with recognition module during training
            latent_sample = sample_z(mu=recognition_mu, logvar=recognition_logvar)
        else:
            kl_loss = torch.tensor(0., dtype=torch.float, device=input_ids.device)
            latent_sample = None

        if self.bow:
            if not self.cvae:
                raise ValueError("bow loss can only be computed in CVAE model")
            bow_logits = self.bow_module(torch.cat([context_embedding, latent_sample], dim=1)) \
                .unsqueeze(1).repeat(1, labels.shape[1] - 1, 1)  # 不计算最后一位 eos
            bow_loss = self.bow_criterion(bow_logits.view(-1, bow_logits.shape[-1]),
                                          labels[:, :-1].contiguous().view(-1))
        else:
            bow_loss = torch.tensor(0., dtype=torch.float, device=input_ids.device)

        logits = self.decoder(input_ids=input_ids, type_ids=type_ids, latent_sample=latent_sample)
        seq_loss = self.seq_criterion(logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]),
                                      labels[:, 1:].contiguous().view(-1))

        return logits, seq_loss, kl_loss, bow_loss

    def sample_z_for_inference(self, context, persona, context_cls_position, persona_cls_position, n_samples):
        # sample a list of lantent z for decoding
        persona_embedding = self.sent_embedding_module(input_ids=persona, cls_positions=persona_cls_position)
        context_embedding = self.sent_embedding_module(input_ids=context, cls_positions=context_cls_position)
        prior_mu, prior_logvar = self.prior_module(persona_embedding=persona_embedding,
                                                   context_embedding=context_embedding)
        samples = []
        for _ in range(n_samples):
            latent_sample = sample_z(mu=prior_mu, logvar=prior_logvar)
            samples.append(latent_sample)
        return samples

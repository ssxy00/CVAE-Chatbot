# -*- coding: utf-8 -*-
# @Time        : 2020/5/24 22:44
# @Author      : ssxy00
# @File        : predict_utils.py
# @Description : beam_search function is modified from # modified from https://github.com/atselousov/transformer_chatbot/blob/agent/model/transformer_model.py

from collections import Counter
import torch
import torch.nn.functional as F

def compute_f1(gold_list, predict_list):
    epsilon = 1e-5
    common = Counter(gold_list) & Counter(predict_list)
    num_same = sum(common.values())
    precision = num_same / max(len(predict_list), epsilon)
    recall = num_same / max(len(gold_list), epsilon)
    f1 = (2 * precision * recall) / max((precision + recall), epsilon)
    return f1


def beam_search(input_ids, type_ids, vocab, model, max_len, beam_size, latent_sample=None):
    with torch.no_grad():
        batch_size = input_ids.shape[0]
        device = input_ids.device

        beam_scores = torch.zeros(batch_size, beam_size, device=device)
        beam_lens = torch.ones(batch_size, beam_size, dtype=torch.long, device=device)
        is_end = torch.zeros(batch_size, beam_size, dtype=torch.bool, device=device)

        prevs = torch.full((batch_size * beam_size, 1), fill_value=vocab.speaker2_id, dtype=torch.long,
                           device=device)

        mask = torch.tensor([0] + [-1e+6] * (len(vocab) - 1), device=device)

        if latent_sample is None:
            beam_latent_sample = None
        else:
            beam_latent_sample = latent_sample.squeeze(1).repeat(1, beam_size, 1).view(-1, latent_sample.shape[-1])


        for i in range(max_len):
            beam_input_ids = torch.cat((input_ids.squeeze(1).repeat(1, beam_size, 1).view(-1, input_ids.shape[-1]),
                                        prevs), dim=1)
            prevs_type = torch.full(prevs.shape, fill_value=vocab.speaker2_id, dtype=torch.long, device=device)
            beam_type_ids = torch.cat((type_ids.repeat(1, beam_size, 1).view(-1, type_ids.shape[-1]),
                                       prevs_type), dim=1)

            logits = model.decoder(input_ids=beam_input_ids, type_ids=beam_type_ids,
                                   latent_sample=beam_latent_sample)[:, -1, :]

            log_probs = F.log_softmax(logits, dim=-1)
            log_probs = log_probs.view(batch_size, beam_size, -1)

            # if a beam reaches end, we only want to keep one branch of this beam
            beam_scores = beam_scores.unsqueeze(-1) + log_probs * (1 - is_end.float().unsqueeze(-1)) + is_end.float().unsqueeze(-1) * mask

            if i == 0:
                beam_scores = beam_scores[:, 0, :]  # 因为初始每个束内容相同
                beam_scores, idxs = beam_scores.topk(beam_size, dim=-1)  # bsz, beam_size
                beam_idxs = torch.zeros((batch_size, beam_size), dtype=torch.long, device=device)  # 表示选择哪几个束
            else:
                beam_scores = beam_scores.view(batch_size, beam_size, -1)


                beam_scores = beam_scores.view(batch_size, -1)
                beam_scores, idxs = beam_scores.topk(beam_size, dim=-1)
                beam_idxs = idxs / log_probs.shape[-1]


            sym_idxs = torch.fmod(idxs, log_probs.shape[-1])  # selected next idxs
            is_end = torch.gather(is_end, 1, beam_idxs)
            beam_lens = torch.gather(beam_lens, 1, beam_idxs)

            sym_idxs[is_end] = vocab.pad_id
            beam_lens[~is_end] += 1
            is_end[sym_idxs == vocab.eos_id] = 1  # <eos> means end of sentence
            sym_idxs = sym_idxs.view(batch_size * beam_size, 1)
            prevs = prevs.view(batch_size, beam_size, -1)
            prevs = torch.gather(prevs, 1, beam_idxs.unsqueeze(-1).repeat(1, 1, prevs.shape[-1]))
            prevs = prevs.view(batch_size * beam_size, -1)
            prevs = torch.cat([prevs, sym_idxs], dim=1)

            if all(is_end.view(-1)):
                break


        sorted_values, sorted_indices = beam_scores.squeeze(0).sort(dim=-1, descending=True)

        indices_offset = (torch.arange(0, batch_size, device=sorted_indices.device) * beam_size).unsqueeze(-1).repeat(1, beam_size)
        sorted_indices = (sorted_indices + indices_offset).view(-1)
        results = prevs.index_select(dim=0, index=sorted_indices).view(batch_size, beam_size, -1)

    return results


def beam_search_for_compressed(persona_embedding, context_embedding, vocab, model, max_len, beam_size,
                               latent_sample=None):
    with torch.no_grad():
        batch_size = persona_embedding.shape[0]
        device = persona_embedding.device

        beam_scores = torch.zeros(batch_size, beam_size, device=device)
        beam_lens = torch.ones(batch_size, beam_size, dtype=torch.long, device=device)
        is_end = torch.zeros(batch_size, beam_size, dtype=torch.bool, device=device)

        prevs = torch.full((batch_size * beam_size, 1), fill_value=vocab.speaker2_id, dtype=torch.long,
                           device=device)

        mask = torch.tensor([0] + [-1e+6] * (len(vocab) - 1), device=device)

        beam_persona_embedding = persona_embedding.squeeze(1).repeat(1, beam_size, 1).view(-1, persona_embedding.shape[-1])
        beam_context_embedding = context_embedding.squeeze(1).repeat(1, beam_size, 1).view(-1, persona_embedding.shape[-1])
        if latent_sample is None:
            beam_latent_sample = None
        else:
            beam_latent_sample = latent_sample.squeeze(1).repeat(1, beam_size, 1).view(-1, latent_sample.shape[-1])


        for i in range(max_len):

            logits = model.decoder(persona_embedding=beam_persona_embedding,
                                   context_embedding=beam_context_embedding,
                              input_ids=prevs, latent_sample=beam_latent_sample)[:, -1, :]

            log_probs = F.log_softmax(logits, dim=-1)
            log_probs = log_probs.view(batch_size, beam_size, -1)

            # if a beam reaches end, we only want to keep one branch of this beam
            beam_scores = beam_scores.unsqueeze(-1) + log_probs * (1 - is_end.float().unsqueeze(-1)) + is_end.float().unsqueeze(-1) * mask

            if i == 0:
                beam_scores = beam_scores[:, 0, :]  # 因为初始每个束内容相同
                beam_scores, idxs = beam_scores.topk(beam_size, dim=-1)  # bsz, beam_size
                beam_idxs = torch.zeros((batch_size, beam_size), dtype=torch.long, device=device)  # 表示选择哪几个束
            else:
                beam_scores = beam_scores.view(batch_size, beam_size, -1)


                beam_scores = beam_scores.view(batch_size, -1)
                beam_scores, idxs = beam_scores.topk(beam_size, dim=-1)
                beam_idxs = idxs / log_probs.shape[-1]


            sym_idxs = torch.fmod(idxs, log_probs.shape[-1])  # selected next idxs
            is_end = torch.gather(is_end, 1, beam_idxs)
            beam_lens = torch.gather(beam_lens, 1, beam_idxs)

            sym_idxs[is_end] = vocab.pad_id
            beam_lens[~is_end] += 1
            is_end[sym_idxs == vocab.eos_id] = 1  # <eos> means end of sentence
            sym_idxs = sym_idxs.view(batch_size * beam_size, 1)
            prevs = prevs.view(batch_size, beam_size, -1)
            prevs = torch.gather(prevs, 1, beam_idxs.unsqueeze(-1).repeat(1, 1, prevs.shape[-1]))
            prevs = prevs.view(batch_size * beam_size, -1)
            prevs = torch.cat([prevs, sym_idxs], dim=1)

            if all(is_end.view(-1)):
                break

        sorted_values, sorted_indices = beam_scores.squeeze(0).sort(dim=-1, descending=True)

        indices_offset = (torch.arange(0, batch_size, device=sorted_indices.device) * beam_size).unsqueeze(-1).repeat(1,
                                                                                                                      beam_size)
        sorted_indices = (sorted_indices + indices_offset).view(-1)
        results = prevs.index_select(dim=0, index=sorted_indices).view(batch_size, beam_size, -1)


    return results
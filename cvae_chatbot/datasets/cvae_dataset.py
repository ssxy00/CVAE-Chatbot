# -*- coding: utf-8 -*-
# @Time        : 2020/5/23 18:42
# @Author      : ssxy00
# @File        : cvae_dataset.py
# @Description :

import logging
from itertools import chain

import torch
from torch.utils.data import Dataset


class CVAEDataset(Dataset):
    def __init__(self, cache_data_path, vocab, max_seq_len, max_context_len, max_persona_len, max_response_len,
                 max_history=2):
        super(CVAEDataset, self).__init__()
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__file__)
        self.vocab = vocab
        self.max_history = max_history
        # load data from cache
        dialogs = torch.load(cache_data_path)
        self.logger.info("building model inputs")
        data = [self.build_input(dialog=dialog, max_history=max_history, vocab=vocab, max_seq_len=max_seq_len,
                                 max_context_len=max_context_len, max_persona_len=max_persona_len,
                                 max_response_len=max_response_len)
                for dialog in dialogs]
        self.logger.info("padding and converting to tensor")
        self.pad_data = self.get_padding_data(data)

    def __len__(self):
        return self.pad_data["input_ids"].shape[0]

    def __getitem__(self, item):
        return {"input_ids": self.pad_data["input_ids"][item, :],  # seq_len
                "type_ids": self.pad_data["type_ids"][item, :],  # seq_len
                "labels": self.pad_data["labels"][item, :],  # seq_len
                "context": self.pad_data["context"][item, :],  # context_len
                "persona": self.pad_data["persona"][item, :],  # persona_len
                "response": self.pad_data["response"][item, :],  # response_len
                "context_cls_position": self.pad_data["context_cls_position"][item],
                "persona_cls_position": self.pad_data["persona_cls_position"][item],
                "response_cls_position": self.pad_data["response_cls_position"][item],
                }

    @staticmethod
    def build_input(dialog, max_history, vocab, max_seq_len, max_context_len, max_persona_len, max_response_len):
        context = dialog['context'][-(2 * max_history + 1):]
        input_ids = [[vocab.speaker2_id if i % 2 else vocab.speaker1_id] + s for i, s in enumerate(context)]
        type_ids = [vocab.speaker2_id if i % 2 else vocab.speaker1_id for i, s in enumerate(input_ids) for _ in s]
        input_ids = list(chain(*input_ids))
        assert len(input_ids) == len(type_ids)

        context = list(chain(*context))
        persona = list(chain(*dialog['persona']))
        response = dialog["response"]

        input_ids = persona + input_ids + [vocab.speaker2_id] + response
        type_ids = [vocab.speaker2_id] * len(persona) + type_ids + [vocab.speaker2_id] * (len(response) + 1)

        if max_seq_len == -1:
            input_ids = [vocab.bos_id] + input_ids + [vocab.eos_id]
            type_ids = [vocab.speaker2_id] + type_ids + [vocab.speaker2_id]
        else:
            input_ids = [vocab.bos_id] + input_ids[-(max_seq_len - 2):] + [vocab.eos_id]
            type_ids = [vocab.speaker2_id] + type_ids[-(max_seq_len - 2):] + [vocab.speaker2_id]
        # ground truth label of each token, pad_token not included in loss
        labels = [vocab.pad_id] * (len(input_ids) - len(response) - 1) + dialog["response"] + [vocab.eos_id]
        assert len(labels) == len(input_ids)
        assert len(input_ids) == len(type_ids)

        # final hidden state of 【eos】 as sentence representation
        if max_persona_len == -1:
            persona = persona + [vocab.eos_id]
        else:
            persona = persona[-(max_persona_len - 1):] + [vocab.eos_id]
        if max_context_len == -1:
            context = context + [vocab.eos_id]
        else:
            context = context[-(max_context_len - 1):] + [vocab.eos_id]
        if max_response_len == -1:
            response = response + [vocab.eos_id]
        else:
            response = response[-(max_response_len - 1):] + [vocab.eos_id]

        persona_cls_position = len(persona) - 1
        context_cls_position = len(context) - 1
        response_cls_position = len(response) - 1

        return {'input_ids': input_ids,  # List[int], input of GPT2 for decoding
                'type_ids': type_ids,  # List[int], input of GPT2 for decoding
                'labels': labels,  # List[int], golden labels
                'context': context,  # List[int], input of context sentence embedding
                'persona': persona,  # List[int], input of persona sentence embedding
                'response': response,  # List[int], input of response sentence embedding
                'context_cls_position': context_cls_position,  # int, 【eos】position in context sequence
                'persona_cls_position': persona_cls_position,  # int,【eos】position in persona sequence
                'response_cls_position': response_cls_position,  # int,【eos】position in response sequence
                }

    def get_padding_data(self, data):
        """padding all data in advance"""
        pad_data = {"input_ids": [],  # n_samples, seq_len
                    "type_ids": [],  # n_samples, seq_len
                    "labels": [],  # n_samples, seq_len
                    "context": [],  # n_samples, seq_len
                    "persona": [],  # n_samples, seq_len
                    "response": [],  # n_samples, seq_len
                    "context_cls_position": [],  # n_samples
                    "persona_cls_position": [],  # n_samples
                    "response_cls_position": [],  # n_samples
                    }
        for instance in data:
            pad_data["input_ids"].append(instance["input_ids"])
            pad_data["type_ids"].append(instance["type_ids"])
            pad_data["labels"].append(instance["labels"])
            pad_data["context"].append(instance["context"])
            pad_data["persona"].append(instance["persona"])
            pad_data["response"].append(instance["response"])
            pad_data["context_cls_position"].append(instance["context_cls_position"])
            pad_data["persona_cls_position"].append(instance["persona_cls_position"])
            pad_data["response_cls_position"].append(instance["response_cls_position"])

        # pad data
        max_seq_len = max(len(sequence) for sequence in pad_data["input_ids"])
        for key in ["input_ids", "type_ids", "labels"]:
            pad_data[key] = self.pad_and_convert_to_tensor(pad_data[key], pad_id=self.vocab.pad_id,
                                                           max_seq_len=max_seq_len)

        pad_data["context"] = self.pad_and_convert_to_tensor(pad_data["context"], pad_id=self.vocab.pad_id)
        pad_data["persona"] = self.pad_and_convert_to_tensor(pad_data["persona"], pad_id=self.vocab.pad_id)
        pad_data["response"] = self.pad_and_convert_to_tensor(pad_data["response"], pad_id=self.vocab.pad_id)
        pad_data["context_cls_position"] = torch.tensor(pad_data["context_cls_position"], dtype=torch.long)
        pad_data["persona_cls_position"] = torch.tensor(pad_data["persona_cls_position"], dtype=torch.long)
        pad_data["response_cls_position"] = torch.tensor(pad_data["response_cls_position"], dtype=torch.long)

        return pad_data

    def collate_func(self, instances):
        batch_data = {}
        for key in instances[0]:
            batch_data[key] = torch.stack([instance[key] for instance in instances])
        return batch_data

    @staticmethod
    def pad_and_convert_to_tensor(sequences, pad_id, max_seq_len=None):
        if max_seq_len is None:
            max_seq_len = max(len(sequence) for sequence in sequences)
        tensor_data = [seq + [pad_id] * (max_seq_len - len(seq)) for seq in sequences]
        tensor_data = torch.tensor(tensor_data, dtype=torch.long)
        return tensor_data

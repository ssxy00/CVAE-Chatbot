# -*- coding: utf-8 -*-
# @Time        : 2020/5/23 16:16
# @Author      : ssxy00
# @File        : gpt2_tokenizer.py
# @Description :

from transformers import GPT2Tokenizer
from cvae_chatbot.datasets.base_tokenizer import BaseTokenizer

ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                         'additional_special_tokens': ["<speaker1>", "<speaker2>"]}
SPECIAL_TOKENS = ['<bos>', '<eos>', '<pad>', "<speaker1>", "<speaker2>"]


class GPT2Vocab(BaseTokenizer):
    """wrapper of transformers GPT2Tokenizer"""

    def __init__(self, model_path):
        super(GPT2Vocab, self).__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self._add_special_tokens()

    def _add_special_tokens(self):
        """ Add special tokens to the tokenizer and the model if they have not already been added. """
        orig_num_tokens = len(self.tokenizer.encoder)
        num_added_tokens = self.tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)
        self.special_tokens = SPECIAL_TOKENS

    def __len__(self):
        return len(self.tokenizer)

    @property
    def bos_id(self):
        """start of sequence"""
        return self.tokenizer.bos_token_id

    @property
    def eos_id(self):
        """end of sequence"""
        return self.tokenizer.eos_token_id

    @property
    def pad_id(self):
        """padding"""
        return self.tokenizer.pad_token_id

    @property
    def speaker1_id(self):
        """represent partner"""
        return self.tokenizer.convert_tokens_to_ids(["<speaker1>"])[0]

    @property
    def speaker2_id(self):
        """represent self"""
        return self.tokenizer.convert_tokens_to_ids(["<speaker2>"])[0]

    def string2ids(self, string):
        """convert string to list of indices"""
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(string, add_prefix_space=True))

    def ids2string(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=True):
        """convert list of indices to string"""
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens,
                                     clean_up_tokenization_spaces=clean_up_tokenization_spaces)

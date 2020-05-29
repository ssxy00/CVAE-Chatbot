# -*- coding: utf-8 -*-
# @Time        : 2020/5/23 16:32
# @Author      : ssxy00
# @File        : data_preprocessor.py
# @Description :

import json
import logging
from tqdm import tqdm
import torch


class DataPreprocessor:
    """
    raw data preprocessor
    """

    def __init__(self, vocab_path):
        """
        :param vocab_path: path to pretrained model vocab file
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__file__)
        self.vocab_path = vocab_path



    def save_json(self, json_data, json_data_path):
        """save data to json file"""
        self.logger.info("write to json file")
        with open(json_data_path, 'w') as fout:
            json.dump(json_data, fout)

    def save_cache(self, indexed_data, cache_data_path):
        """save indexed data to cache path"""
        self.logger.info("write to cache file")
        torch.save(indexed_data, cache_data_path)

    def parse_data(self, data_path):
        """
        :param data_path: path to raw dataset file
        :return: List[{'persona': List[str],
                       'context': List[str],
                       'response': str}]
        """
        self.logger.info("parsing dialogs from file")
        with open(data_path, 'r', encoding='utf-8') as fin:
            data = []
            dialog = {"persona": [], "dialog": []}
            for line in tqdm(fin):
                line = line.strip()
                if len(line) == 0:
                    continue
                line_idx, line_content = line.split(' ', 1)

                if int(line_idx) == 1 and len(dialog["persona"]):
                    # start line of a new dialog
                    # process previous dialog
                    data += self.split_dialog(dialog)
                    # record new dialog
                    dialog = {"persona": [], "dialog": []}

                if line_content.startswith("your persona: "):  # persona line
                    dialog["persona"].append(line_content.replace("your persona: ", ''))
                else:  # utterance line
                    dialog["dialog"] += [utterance.strip() for utterance in line_content.split('\t')]
            # process last dialog
            data += self.split_dialog(dialog)
            return data

    def split_dialog(self, dialog):
        """
        split a single multi-turn dialog into several single-turn dialogs
        dialog: {"persona": [], "dialog": []}
        return: [{'persona': List[str], 'context': List[str], 'response': str}]
        """
        # self.logger.info("splitting a multi-turn dialog into single-turn dialogs")
        # assert len(dialog["persona"]) > 0
        assert len(dialog["dialog"]) > 0
        assert not len(dialog["dialog"]) % 2  # dialog should have even lines, start from partner, end by self
        single_turn_dialogs = []

        for turn in range(len(dialog['dialog']) // 2):
            single_turn_dialogs.append({'persona': dialog['persona'],
                                        'context': dialog['dialog'][: 2 * turn + 1],
                                        'response': dialog['dialog'][2 * turn + 1]})
        return single_turn_dialogs

    def tokenize_and_index_data(self, data, vocab):
        """tokenize string and convert tokens to indices"""
        self.logger.info("tokenize and index data")
        def tokenize(obj):
            if isinstance(obj, str):
                return vocab.string2ids(obj)
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)
        indexed_data = []
        for dialog in tqdm(data):
            indexed_data.append(tokenize(dialog))
        return indexed_data

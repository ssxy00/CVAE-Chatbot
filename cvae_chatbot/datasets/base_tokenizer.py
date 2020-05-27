# -*- coding: utf-8 -*-
# @Time        : 2020/5/23 16:21
# @Author      : ssxy00
# @File        : base_tokenizer.py
# @Description :

class BaseTokenizer:
    def __init__(self):
        pass

    def string2ids(self, string):
        raise NotImplementedError

    def ids2string(self, ids):
        raise NotImplementedError
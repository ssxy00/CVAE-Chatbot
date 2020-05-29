# -*- coding: utf-8 -*-
# @Time        : 2020/5/23 14:29
# @Author      : ssxy00
# @File        : preprocess_data.py
# @Description :

import argparse
from prepare_data.data_preprocessor import DataPreprocessor
from cvae_chatbot.datasets.gpt2_tokenizer import GPT2Vocab

def main(args):
    preprocessor = DataPreprocessor(vocab_path=args.gpt2_vocab_path)

    # load data
    json_data = preprocessor.parse_data(data_path=args.raw_data)

    # tokenize and index data
    vocab = GPT2Vocab(model_path=args.gpt2_vocab_path)
    indexed_data = preprocessor.tokenize_and_index_data(data=json_data, vocab=vocab)
    preprocessor.save_cache(indexed_data=indexed_data, cache_data_path=args.cache_data)



def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data", default="/home/Parlai/data/ConvAI2/train_self_original_no_cands.txt",
                        help="path to raw dataset file"
                        )
    parser.add_argument("--cache_data",
                        default="./datasets/train_self_original_no_cands.cache",
                        help="path to save json format dataset")
    parser.add_argument("--gpt2_vocab_path",
                        default="./gpt2/tokenizer",
                        help="path to GPT2 tokenizer vocab file")

    args = parser.parse_args()
    main(args)

if __name__ == "__main__":
    cli_main()
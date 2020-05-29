# -*- coding: utf-8 -*-
# @Time        : 2020/5/24 21:57
# @Author      : ssxy00
# @File        : predict.py
# @Description :

import argparse
import logging

import torch


from cvae_chatbot.datasets.gpt2_tokenizer import GPT2Vocab
from cvae_chatbot.datasets.cvae_dataset import CVAEDataset
from cvae_chatbot.predictors.cvae_predictor import CVAEPredictor
from cvae_chatbot.modules.custom_gpt2_module import CustomGPT2Module
from cvae_chatbot.models.cvae_model import CVAEModel
from cvae_chatbot.models.cvae_compressed_model import CVAECompressedModel
from utils import set_seed

def main(args):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)

    set_seed(args.seed)
    device = torch.device(args.device)

    # initialize dataset
    vocab = GPT2Vocab(model_path=args.gpt2_vocab_path)


    logger.info("loading valid dataset")
    valid_dataset = CVAEDataset(cache_data_path=args.valid_dataset, vocab=vocab, max_history=2,
                                max_seq_len=args.max_seq_len, max_context_len=args.max_context_len,
                                max_persona_len=args.max_persona_len, max_response_len=args.max_response_len)

    # initialize model
    gpt2_module = CustomGPT2Module.from_pretrained(args.gpt2_model_dir)
    gpt2_module.resize_token_embeddings(new_num_tokens=len(vocab))
    if "compressed" in args.model_type:
        model = CVAECompressedModel(core_module=gpt2_module, pad_id=vocab.pad_id, speaker2_id=vocab.speaker2_id,
                                    z_dim=args.z_dim, model_type=args.model_type)
    else:
        model = CVAEModel(core_module=gpt2_module, pad_id=vocab.pad_id, z_dim=args.z_dim, model_type=args.model_type)
    # initialize evaluator
    predictor = CVAEPredictor(args=args, device=device, model=model, valid_dataset=valid_dataset, vocab=vocab)

    if "compressed" in args.model_type:
        predictor.predict_compressed()
    else:
        predictor.predict()



def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt2_model_dir", default="./gpt2/model", help="path to GPT2 pretrained model parameters")
    parser.add_argument("--gpt2_vocab_path", default="./gpt2/tokenizer", help="path to GPT2 tokenizer vocab file")
    parser.add_argument("--valid_dataset", default="./datasets/valid_self_original_no_cands.cache",
                        help="cache valid_dataset path")
    parser.add_argument("--output_path", default="./result.jsonl", help="path to output prediction results")
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--max_seq_len", default=-1, type=int, help="max concat sequence length")
    parser.add_argument("--max_context_len", default=-1, type=int, help="max context sequence length")
    parser.add_argument("--max_persona_len", default=-1, type=int, help="max persona sequence length")
    parser.add_argument("--max_response_len", default=-1, type=int, help="max response sequence length")
    parser.add_argument("--max_predict_len", default=32, type=int, help="max predicted response sequence length")
    parser.add_argument("--n_outputs", default=3, type=int, help="how many candidates to generate")
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("--device", default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument("--z_dim", default=200, help="dim of z")
    parser.add_argument("--checkpoint_path", default="", help="path to load model checkpoint")
    parser.add_argument("--model_type", type=str, default="decoder",
                        help="decoder, cvae_memory, cvae_embedding, compressed_decoder or compressed_cvae")
    args = parser.parse_args()
    main(args)

if __name__ == "__main__":
    cli_main()
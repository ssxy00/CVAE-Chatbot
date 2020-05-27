# -*- coding: utf-8 -*-
# @Time        : 2020/5/22 23:26
# @Author      : ssxy00
# @File        : train.py
# @Description : training script

import argparse
import logging
import torch

from cvae_chatbot.datasets.gpt2_tokenizer import GPT2Vocab
from cvae_chatbot.datasets.cvae_dataset import CVAEDataset
from cvae_chatbot.trainers.cvae_trainer import CVAETrainer
from cvae_chatbot.modules.custom_gpt2_module import CustomGPT2Module
from cvae_chatbot.models.cvae_model import CVAEModel
from cvae_chatbot.models.cvae_compressed_model import CVAECompressedModel
from utils import set_seed


def main(args):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)

    # set random seed
    set_seed(args.seed)

    # initialize dataset
    vocab = GPT2Vocab(model_path=args.gpt2_vocab_path)
    logger.info("loading training dataset")
    train_dataset = CVAEDataset(cache_data_path=args.train_dataset, vocab=vocab, max_history=args.max_history,
                                max_seq_len=args.max_seq_len, max_context_len=args.max_context_len,
                                max_persona_len=args.max_persona_len, max_response_len=args.max_response_len)

    logger.info("loading valid dataset")
    valid_dataset = CVAEDataset(cache_data_path=args.valid_dataset, vocab=vocab, max_history=args.max_history,
                                max_seq_len=args.max_seq_len, max_context_len=args.max_context_len,
                                max_persona_len=args.max_persona_len, max_response_len=args.max_response_len)

    # initialize model
    gpt2_module = CustomGPT2Module.from_pretrained(args.gpt2_model_dir)
    gpt2_module.resize_token_embeddings(new_num_tokens=len(vocab))
    if "compressed" in args.model_type:
        model = CVAECompressedModel(core_module=gpt2_module, pad_id=vocab.pad_id, speaker2_id=vocab.speaker2_id,
                                    z_dim=args.z_dim, bow=args.bow, model_type=args.model_type)
    else:
        model = CVAEModel(core_module=gpt2_module, pad_id=vocab.pad_id, z_dim=args.z_dim, bow=args.bow,
                          model_type=args.model_type)

    # initialize trainer
    device = torch.device(args.device)
    trainer = CVAETrainer(args=args, device=device, model=model, train_dataset=train_dataset,
                          valid_dataset=valid_dataset, vocab=vocab)

    # begin to train
    trainer.train()


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt2_model_dir", default="./gpt2/model",
                        help="path to GPT2 pretrained model parameters")
    parser.add_argument("--gpt2_vocab_path", default="./gpt2/tokenizer",
                        help="path to GPT2 tokenizer vocab file")
    parser.add_argument("--train_dataset",
                        default="./datasets/train_self_original_no_cands.cache",
                        help="cache train_dataset path")
    parser.add_argument("--valid_dataset",
                        default="./datasets/valid_self_original_no_cands.cache",
                        help="cache valid_dataset path")
    parser.add_argument("--max_seq_len", default=160, type=int, help="max sequence length fed into GPT2")
    parser.add_argument("--max_history", default=2, type=int, help="max number of historical conversation turns to use")
    parser.add_argument("--max_context_len", default=100, type=int,
                        help="max context sequence length for sentence embedding")
    parser.add_argument("--max_persona_len", default=70, type=int,
                        help="max persona sequence length for sentence embedding")
    parser.add_argument("--max_response_len", default=30, type=int,
                        help="max response sequence length for sentence embedding")
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("--device", default='cuda' if torch.cuda.is_available() else 'cpu', type=str,
                        help="cpu or cuda")
    parser.add_argument("--z_dim", default=200, help="latent hidden state dim (z)")
    parser.add_argument("--n_epochs", default=1, type=int, help="number of training epochs")
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--lr", default=6.25e-5, type=float, help="learning rate")
    parser.add_argument("--gradient_accumulate_steps", default=1, type=int, help="accumulate gradient on several steps")
    parser.add_argument("--clip_grad", type=float, default=1.0, help="clip gradient threshold")
    parser.add_argument("--save_model_dir", default="./checkpoints", help="path to save model checkpoints")
    parser.add_argument("--log_dir", default="", help="path to save logs, no log output by default")
    parser.add_argument("--save_interval", default=1)
    parser.add_argument("--model_type", type=str, default="compressed_cvae",
                        help="decoder, cvae_memory, cvae_embedding, compressed_decoder or compressed_cvae")
    parser.add_argument("--bow", action="store_true", help="add bow loss or not")
    parser.add_argument("--kl_coef", default=1.0, type=float, help="kl loss coef")
    parser.add_argument("--bow_coef", default=1.0, type=float, help="bow loss coef")

    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()

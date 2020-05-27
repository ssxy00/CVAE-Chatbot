# -*- coding: utf-8 -*-
# @Time        : 2020/5/23 23:36
# @Author      : ssxy00
# @File        : cvae_trainer.py
# @Description :

import os
import math
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from transformers import AdamW

from cvae_chatbot.trainers.linear_decay_schedule import LinearDecaySchedule


class CVAETrainer:
    def __init__(self, args, device, model, train_dataset, valid_dataset, vocab):
        self.device = device
        self.model = model.to(device)
        self.vocab = vocab
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

        self.n_epochs = args.n_epochs
        self.batch_size = args.batch_size
        self.gradient_accumulate_steps = args.gradient_accumulate_steps
        self.clip_grad = args.clip_grad

        # checkpoint
        self.save_model_dir = args.save_model_dir
        self.save_interval = args.save_interval

        # criterion
        self.seq_criterion = nn.CrossEntropyLoss(ignore_index=self.vocab.pad_id)
        self.kl_coef = args.kl_coef
        self.bow_coef = args.bow_coef

        # optimizer
        lr = args.lr
        base_optimizer = AdamW(model.parameters(), lr=lr, correct_bias=True)
        total_steps = math.ceil(self.n_epochs * len(self.train_dataset) / self.batch_size)
        # using linear decay learning rate schedule
        self.optimizer = LinearDecaySchedule(total_steps, lr, base_optimizer)

        # log
        self.writer = SummaryWriter(args.log_dir) if args.log_dir else None

    def state_dict(self):
        return {'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict()}

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'], strict=False)
        self.optimizer.load_state_dict(state_dict['optimizer'])

    def _eval_train(self, epoch):
        self.model.train()
        ave_seq_loss = 0
        ave_kl_loss = 0
        ave_bow_loss = 0
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                      collate_fn=self.train_dataset.collate_func, num_workers=4)

        tqdm_data = tqdm(train_dataloader, desc='Train (epoch #{})'.format(epoch))
        n_samples = len(tqdm_data)

        for i, data in enumerate(tqdm_data):
            data = {key: data[key].to(self.device) for key in data}
            seq_logits, seq_loss, kl_loss, bow_loss = self.model(input_ids=data['input_ids'],
                                                                 type_ids=data['type_ids'],
                                                                 labels=data['labels'],
                                                                 context=data["context"],
                                                                 persona=data["persona"],
                                                                 response=data["response"],
                                                                 context_cls_position=data["context_cls_position"],
                                                                 persona_cls_position=data["persona_cls_position"],
                                                                 response_cls_position=data["response_cls_position"]
                                                                 )

            loss = (seq_loss + kl_loss * self.kl_coef + bow_loss * self.bow_coef) / self.gradient_accumulate_steps
            loss.backward()
            if self.clip_grad is not None:
                for group in self.optimizer.param_groups:
                    nn.utils.clip_grad_norm_(group['params'], self.clip_grad)

            if (i + 1) % self.gradient_accumulate_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            ave_seq_loss = (ave_seq_loss * i + seq_loss.item()) / (i + 1)
            ave_kl_loss = (ave_kl_loss * i + kl_loss.item()) / (i + 1)
            ave_bow_loss = (ave_bow_loss * i + bow_loss.item()) / (i + 1)

            tqdm_data.set_postfix({'seq_loss': seq_loss.item(),
                                   'ave_seq_loss': ave_seq_loss,
                                   'kl_loss': kl_loss.item(),
                                   'ave_kl_loss': ave_kl_loss,
                                   'bow_loss': bow_loss.item(),
                                   'ave_bow_loss': ave_bow_loss,
                                   'lr': self.optimizer.param_groups[0]['lr']})
            if self.writer is not None:
                self.writer.add_scalar("Train/seq_loss", seq_loss.item(), (epoch - 1) * n_samples + i)
                self.writer.add_scalar("Train/ave_seq_loss", ave_seq_loss, (epoch - 1) * n_samples + i)
                self.writer.add_scalar("Train/kl_loss", kl_loss.item(), (epoch - 1) * n_samples + i)
                self.writer.add_scalar("Train/ave_kl_loss", ave_kl_loss, (epoch - 1) * n_samples + i)
                self.writer.add_scalar("Train/bow_loss", bow_loss.item(), (epoch - 1) * n_samples + i)
                self.writer.add_scalar("Train/ave_bow_loss", ave_bow_loss, (epoch - 1) * n_samples + i)
                self.writer.add_scalar("Train/lr", self.optimizer.param_groups[0]['lr'], (epoch - 1) * n_samples + i)

    def _eval_valid(self, epoch):
        self.model.eval()
        valid_dataloader = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False,
                                      collate_fn=self.valid_dataset.collate_func, num_workers=4)

        tqdm_data = tqdm(valid_dataloader, desc='Valid (epoch #{})'.format(epoch))

        ave_seq_loss = 0
        ave_kl_loss = 0
        ave_bow_loss = 0
        n_samples = len(tqdm_data)

        with torch.no_grad():
            for i, data in enumerate(tqdm_data):
                data = {key: data[key].to(self.device) for key in data}
                seq_logits, seq_loss, kl_loss, bow_loss = self.model(input_ids=data['input_ids'],
                                                                     type_ids=data['type_ids'],
                                                                     labels=data['labels'],
                                                                     context=data["context"],
                                                                     persona=data["persona"],
                                                                     response=data["response"],
                                                                     context_cls_position=data["context_cls_position"],
                                                                     persona_cls_position=data["persona_cls_position"],
                                                                     response_cls_position=data["response_cls_position"]
                                                                     )

                ave_seq_loss = (ave_seq_loss * i + seq_loss.item()) / (i + 1)
                ave_kl_loss = (ave_kl_loss * i + kl_loss.item()) / (i + 1)
                ave_bow_loss = (ave_bow_loss * i + bow_loss.item()) / (i + 1)

                tqdm_data.set_postfix({'seq_loss': seq_loss.item(),
                                       'ave_seq_loss': ave_seq_loss,
                                       'kl_loss': kl_loss.item(),
                                       'ave_kl_loss': ave_kl_loss,
                                       'bow_loss': bow_loss.item(),
                                       'ave_bow_loss': ave_bow_loss
                                       })

                if self.writer is not None:
                    self.writer.add_scalar("Valid/seq_loss", seq_loss.item(), (epoch - 1) * n_samples + i)
                    self.writer.add_scalar("Valid/ave_seq_loss", ave_seq_loss, (epoch - 1) * n_samples + i)
                    self.writer.add_scalar("Valid/kl_loss", kl_loss.item(), (epoch - 1) * n_samples + i)
                    self.writer.add_scalar("Valid/ave_kl_loss", ave_kl_loss, (epoch - 1) * n_samples + i)
                    self.writer.add_scalar("Valid/bow_loss", bow_loss.item(), (epoch - 1) * n_samples + i)
                    self.writer.add_scalar("Valid/ave_bow_loss", ave_bow_loss, (epoch - 1) * n_samples + i)

    def train(self, last_epoch=0):
        print('begin to train')
        for epoch_idx in range(last_epoch + 1, self.n_epochs + 1):
            self._eval_train(epoch_idx)
            self._eval_valid(epoch_idx)
            if epoch_idx % self.save_interval == 0:
                save_dir = os.path.join(self.save_model_dir, f"checkpoint{epoch_idx}.pt")
                torch.save(self.state_dict(), save_dir)

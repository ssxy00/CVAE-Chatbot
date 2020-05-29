# -*- coding: utf-8 -*-
# @Time        : 2020/5/24 22:02
# @Author      : ssxy00
# @File        : cvae_predictor.py
# @Description :

import jsonlines
import math
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from cvae_chatbot.predictors.predict_utils import beam_search, compute_f1, beam_search_for_compressed
from cvae_chatbot.models.model_utils import sample_z


class CVAEPredictor:
    def __init__(self, args, device, model, valid_dataset, vocab):
        self.args = args
        self.device = device
        self.model = model.to(device)
        # load checkpoint
        self.load_state_dict(torch.load(args.checkpoint_path, map_location=self.device))
        print('Weights loaded from {}'.format(args.checkpoint_path))

        self.vocab = vocab
        self.valid_dataset = valid_dataset
        self.batch_size = self.args.batch_size

    def state_dict(self):
        return {'model': self.model.state_dict()}

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model'], strict=False)

    def predict(self):
        ave_ppl = 0.
        ave_f1 = 0.  # average max f1 among candidates
        count = 0
        self.model.eval()
        valid_dataloader = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False,
                                      collate_fn=self.valid_dataset.collate_func, num_workers=4)

        with torch.no_grad():
            with jsonlines.open(self.args.output_path, mode='w') as writer:
                for i, data in enumerate(tqdm(valid_dataloader)):
                    data = {key: data[key].to(self.device) for key in data}
                    _, seq_loss, _, _ = self.model(input_ids=data['input_ids'],
                                                   type_ids=data['type_ids'],
                                                   labels=data['labels'],
                                                   context=data["context"],
                                                   persona=data["persona"],
                                                   response=data["response"],
                                                   context_cls_position=data["context_cls_position"],
                                                   persona_cls_position=data["persona_cls_position"],
                                                   response_cls_position=data["response_cls_position"]
                                                   )
                    ppl = math.exp(seq_loss.item())
                    ave_ppl = (ave_ppl * i + ppl) / (i + 1)

                    if self.model.cvae:
                        # get latent samples
                        samples = self.model.sample_z_for_inference(context=data['context'],
                                                                    persona=data['persona'],
                                                                    context_cls_position=data['context_cls_position'],
                                                                    persona_cls_position=data['persona_cls_position'],
                                                                    n_samples=self.args.n_outputs)
                        predict_ids = beam_search(
                            input_ids=data['input_ids'].squeeze(1).repeat(1, self.args.n_outputs, 1).view(-1, data[
                                'input_ids'].shape[-1]),
                            type_ids=data['type_ids'].squeeze(1).repeat(1, self.args.n_outputs, 1).view(-1, data[
                                'type_ids'].shape[-1]),
                            max_len=self.args.max_predict_len,
                            beam_size=1,
                            vocab=self.vocab,
                            model=self.model,
                            latent_sample=torch.cat(samples, dim=0)
                        )
                        predict_ids = predict_ids.view(data['input_ids'].shape[0], self.args.n_outputs, -1)
                    else:
                        predict_ids = beam_search(input_ids=data['input_ids'],
                                                  type_ids=data['type_ids'],
                                                  max_len=self.args.max_predict_len,
                                                  beam_size=self.args.n_outputs,
                                                  vocab=self.vocab,
                                                  model=self.model
                                                  )

                    for idx in range(data['input_ids'].shape[0]):

                        print(f"persona:")
                        persona_string = self.vocab.ids2string(data["persona"][idx, :], skip_special_tokens=True,
                                                               clean_up_tokenization_spaces=False)
                        print(persona_string)
                        print(f"context:")
                        context_string = self.vocab.ids2string(data["context"][idx, :], skip_special_tokens=True,
                                                               clean_up_tokenization_spaces=False)
                        print(context_string)
                        print(f"golden response:")
                        golden_response_string = self.vocab.ids2string(data["response"][idx, :].squeeze(0),
                                                                       skip_special_tokens=True,
                                                                       clean_up_tokenization_spaces=False)
                        print(golden_response_string)
                        result = {"persona": persona_string,
                                  "context": context_string,
                                  "golden_response": golden_response_string,
                                  "predict_responses": [],
                                  "predict_f1s": []}

                        f1 = 0
                        for predict_idx in range(predict_ids.shape[1]):
                            predict_response_string = self.vocab.ids2string(predict_ids[idx, predict_idx, :],
                                                                            skip_special_tokens=True,
                                                                            clean_up_tokenization_spaces=False)
                            print(
                                f"{predict_idx}: {predict_response_string}")
                            result["predict_responses"].append(predict_response_string)
                            candidate_f1 = compute_f1(gold_list=golden_response_string.split(),
                                                      predict_list=predict_response_string.split())
                            result["predict_f1s"].append(candidate_f1)
                            f1 = max(f1, candidate_f1)
                        writer.write(result)
                        ave_f1 = (ave_f1 * count + f1) / (count + 1)
                        count += 1

        print(f"average ppl: {ave_ppl}, average f1: {ave_f1}")

    def predict_compressed(self):
        ave_ppl = 0.
        ave_f1 = 0.
        count = 0
        self.model.eval()
        valid_dataloader = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False,
                                      collate_fn=self.valid_dataset.collate_func, num_workers=4)

        with torch.no_grad():
            with jsonlines.open(self.args.output_path, mode='w') as writer:
                for i, data in enumerate(tqdm(valid_dataloader)):
                    data = {key: data[key].to(self.device) for key in data}
                    _, seq_loss, _, _ = self.model(input_ids=data['input_ids'],
                                                   type_ids=data['type_ids'],
                                                   labels=data['labels'],
                                                   context=data["context"],
                                                   persona=data["persona"],
                                                   response=data["response"],
                                                   context_cls_position=data["context_cls_position"],
                                                   persona_cls_position=data["persona_cls_position"],
                                                   response_cls_position=data["response_cls_position"]
                                                   )
                    ppl = math.exp(seq_loss.item())
                    ave_ppl = (ave_ppl * i + ppl) / (i + 1)

                    # f1
                    persona_embedding = self.model.sent_embedding_module(input_ids=data['persona'],
                                                                         cls_positions=data['persona_cls_position'])
                    context_embedding = self.model.sent_embedding_module(input_ids=data['context'],
                                                                         cls_positions=data['context_cls_position'])

                    if self.model.cvae:
                        predict_ids = []
                        # get latent samples
                        prior_mu, prior_logvar = self.model.prior_module(persona_embedding=persona_embedding,
                                                                         context_embedding=context_embedding)
                        samples = []
                        for _ in range(self.args.n_outputs):
                            latent_sample = sample_z(mu=prior_mu, logvar=prior_logvar)
                            samples.append(latent_sample)

                        predict_ids = beam_search_for_compressed(
                            persona_embedding=persona_embedding.squeeze(1).repeat(1, self.args.n_outputs, 1).view(-1,
                                                                                                                  persona_embedding.shape[
                                                                                                                      -1]),
                            context_embedding=context_embedding.squeeze(1).repeat(1, self.args.n_outputs, 1).view(-1,
                                                                                                                  context_embedding.shape[
                                                                                                                      -1]),
                            max_len=self.args.max_predict_len,
                            beam_size=1,
                            vocab=self.vocab,
                            model=self.model,
                            latent_sample=torch.cat(samples, dim=0)
                        )
                        predict_ids = predict_ids.view(persona_embedding.shape[0], self.args.n_outputs, -1)
                    else:
                        predict_ids = beam_search_for_compressed(persona_embedding=persona_embedding,
                                                                 context_embedding=context_embedding,
                                                                 max_len=self.args.max_predict_len,
                                                                 beam_size=self.args.n_outputs,
                                                                 vocab=self.vocab,
                                                                 model=self.model
                                                                 )
                    for idx in range(data['input_ids'].shape[0]):
                        print(f"persona:")
                        persona_string = self.vocab.ids2string(data["persona"][idx, :], skip_special_tokens=True,
                                                               clean_up_tokenization_spaces=False)
                        print(persona_string)
                        print(f"context:")
                        context_string = self.vocab.ids2string(data["context"][idx, :], skip_special_tokens=True,
                                                               clean_up_tokenization_spaces=False)
                        print(context_string)
                        print(f"golden response:")
                        golden_response_string = self.vocab.ids2string(data["response"][idx, :].squeeze(0),
                                                                       skip_special_tokens=True,
                                                                       clean_up_tokenization_spaces=False)
                        print(golden_response_string)
                        result = {"persona": persona_string,
                                  "context": context_string,
                                  "golden_response": golden_response_string,
                                  "predict_responses": [],
                                  "predict_f1s": []}

                        f1 = 0
                        for predict_idx in range(predict_ids.shape[1]):
                            predict_response_string = self.vocab.ids2string(predict_ids[idx, predict_idx, :],
                                                                            skip_special_tokens=True,
                                                                            clean_up_tokenization_spaces=False)
                            print(
                                f"{predict_idx}: {predict_response_string}")
                            result["predict_responses"].append(predict_response_string)
                            candidate_f1 = compute_f1(gold_list=golden_response_string.split(),
                                                      predict_list=predict_response_string.split())
                            result["predict_f1s"].append(candidate_f1)
                            f1 = max(f1, candidate_f1)
                        writer.write(result)
                        ave_f1 = (ave_f1 * count + f1) / (count + 1)
                        count += 1

            print(f"average ppl: {ave_ppl}, average f1: {ave_f1}")

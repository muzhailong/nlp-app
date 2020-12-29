#!/bin/bash

import os
import pandas as pd
import numpy as np
from collections import *
import torch
import torch.functional as F
from transformers import *
from tqdm import tqdm
import argparse
from torch import cuda
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.cuda.amp import autocast, GradScaler
from torch.nn import CrossEntropyLoss
import json
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler
from copy import deepcopy
import torch.nn as nn
from transformers import GPT2LMHeadModel, BertTokenizer, GPT2Config
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import horovod.torch as hvd

args = None
optim = None
tb = None


class ConditionalDataset(Dataset):
    def __init__(self, df):
        super(ConditionalDataset, self).__init__()
        self.data = {
            "sentence1": df['sentence1'],
            "labels": df['labels'],
        }

    def __len__(self):
        return len(self.data['sentence1'])

    def __getitem__(self, index):
        text1 = self.data['sentence1'][index]
        labels = self.data['labels'][index]

        lt1 = args.tokenizer.encode(text1, add_special_tokens=False)
        lt2 = args.tokenizer.encode(labels, add_special_tokens=False)

        max_len1 = args.max_len - len(lt2) - 2
        lt1 = lt1[:max_len1] if len(lt1) > max_len1 else lt1 + [args.tokenizer.pad_token_id] * (max_len1 - len(lt1))
        ret = {
            "input_ids": torch.tensor([args.tokenizer.cls_token_id, ] + lt1 + [
                args.tokenizer.convert_tokens_to_ids("<INFER>")] + lt2),
            "labels": torch.tensor([args.tokenizer.pad_token_id] * (2 + len(lt1)) + lt2)
        }
        return ret


def get_dataLoader(df, is_predict=False):
    dt = ConditionalDataset(df)
    params = {
        "batch_size": args.batch_size,
        "pin_memory": True,
        "num_workers": 4,
    }
    sampler = torch.utils.data.distributed.DistributedSampler(
        dt, num_replicas=hvd.size(), rank=hvd.rank())
    params['sampler'] = sampler

    return DataLoader(dt, **params), sampler


def gradual_decay(epoch, steps, all_steps):
    if epoch < args.warmup_epochs:
        epoch += float(steps + 1) / all_steps
        lr_adj = 1. / hvd.size() * (epoch * (hvd.size() - 1) / args.warmup_epochs + 1)
    elif epoch < 30:
        lr_adj = 1.
    elif epoch < 60:
        lr_adj = 1e-1
    elif epoch < 80:
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3
    for param_group in optim.param_groups:
        param_group['lr'] = args.base_lr * hvd.size() * args.batches_per_allreduce * lr_adj


class ConditionalGenerationModel(nn.Module):
    def __init__(self):
        super(ConditionalGenerationModel, self).__init__()
        if args.base_model.endswith(".json"):
            # 使用config
            model_config = GPT2Config.from_json_file(args.base_model)
            self.base_model = GPT2LMHeadModel(config=model_config)
        else:
            # 载预训练模型
            self.base_model = GPT2LMHeadModel.from_pretrained(args.base_model)
        self.base_model.resize_token_embeddings(len(args.tokenizer))
        self.config = self.base_model.config

    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
    ):
        r"""
        labels 其中为title部分为token的id 其他部分为0 计算loss时可以使用ignore_index 忽略掉
        """
        output = self.base_model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        lm_logits = output.logits
        if labels is None:
            return lm_logits

        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss(reduction='sum', ignore_index=args.tokenizer.pad_token_id)  # [PAD]=0
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        num = shift_labels.ne(0).long().sum().item()
        loss = loss / num
        return loss


def train(model, loader, sampler):
    if os.path.isfile(args.model_save_path):
        model.load_state_dict(torch.load(args.model_save_path))
    if os.path.isfile(args.optim_save_path):
        optim.load_state_dict(torch.load(args.optim_save_path))

    model.train()
    device = args.device
    all_steps = 0
    for e in range(args.epochs):
        sampler.set_epoch(e)
        optim.zero_grad()

        gradual_decay(epoch=e, steps=0, all_steps=len(loader))
        for step, data in enumerate(loader, start=1):
            input_ids = data['input_ids'].to(device)
            labels = data['labels'].to(device)
            loss = model(
                input_ids=input_ids,
                labels=labels
            )
            loss.div_(args.batches_per_allreduce)
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            if step % args.batches_per_allreduce != 0:
                continue
            optim.step()
            optim.zero_grad()
            all_steps += 1
            gradual_decay(epoch=e, steps=all_steps, all_steps=len(loader))

            with torch.no_grad():
                if hvd.rank() == 0:
                    tb.add_scalar("Train/loss", loss.item(), global_step=all_steps)
                    print(f"epochs:{args.epochs}/{e},steps:{all_steps},loss:{loss.item()}", flush=True)
            if hvd.rank() == 0 and all_steps % args.save_steps == 0:
                torch.save(model.state_dict(), args.model_save_path)
                torch.save(optim.state_dict(), args.optim_save_path)


def generate(model, loader):
    model.load_state_dict(torch.load(args.model_save_path))
    model.eval()
    device = args.device


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=20, type=int)
    parser.add_argument('--base_lr', default=4e-4, type=float)

    parser.add_argument('--max_len', default=512)
    parser.add_argument('--do_train', default=1, type=int)
    parser.add_argument('--do_predict', default=1, type=int)

    parser.add_argument('--device', default='cuda')
    parser.add_argument("--train_df_path", type=str)
    parser.add_argument("--logdir", type=str, default="./log")
    parser.add_argument("--model_save_path", type=str)
    parser.add_argument("--optim_save_path", type=str)
    parser.add_argument("--predict_save_path", type=str, default='./predict_result.csv')

    parser.add_argument('--base_model', default='hfl/chinese-macbert-large')
    parser.add_argument('--tokenizer_model', default='bert-base-chinese', type=str)

    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--save_steps', default=100, type=int)

    parser.add_argument('--warmup-epochs', type=float, default=2,
                        help='number of warmup epochs')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--wd', type=float, default=0.00005,
                        help='weight decay')

    parser.add_argument('--compression_fp16', action='store_true', default=False,
                        help='use fp16 compression during allreduce')

    parser.add_argument('--batches-per-allreduce', type=int, default=1,
                        help='number of batches processed locally before '
                             'executing allreduce across workers; it multiplies '
                             'total batch size.')
    parser.add_argument('--use-adasum', action='store_true', default=False,
                        help='use adasum algorithm to do reduction')
    parser.add_argument('--seed', type=int, default=2020,
                        help='random seed')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='')

    args = parser.parse_args()
    if args.device == 'cuda':
        args.device = 'cuda' if cuda.is_available() else 'cpu'
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    hvd.init()
    if cuda.is_available():
        torch.cuda.set_device(hvd.local_rank())

    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_model)
    tokenizer.add_tokens("<INFER>")
    args.tokenizer = tokenizer
    args.rank_size = hvd.size()
    print(args)

    train_df = pd.read_csv(args.train_df_path)
    train_loader, train_sampler = get_dataLoader(train_df)

    model = ConditionalGenerationModel().to(args.device)
    #     model=torch.nn.DataParallel(model).to(args.device)#并行程序会卡住，不晓得为什么
    compression = hvd.Compression.fp16 if args.compression_fp16 else hvd.Compression.none
    optim = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr * hvd.local_size() * args.batches_per_allreduce,
        momentum=args.momentum,
        weight_decay=args.wd
    )

    optim = hvd.DistributedOptimizer(optim,
                                     named_parameters=model.named_parameters(),
                                     compression=compression,
                                     backward_passes_per_step=args.batches_per_allreduce,
                                     op=hvd.Adasum if args.use_adasum else hvd.Average)

    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    # Broadcast parameters from rank 0 to all other processes.

    if hvd.rank() == 0:
        tb = SummaryWriter(args.logdir)

    if args.do_train:
        train(model, train_loader, train_sampler)

    if hvd.rank() == 0:
        tb.close()

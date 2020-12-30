#!/usr/bin/env python
# coding: utf-8
'''
使用类bert模型直接fine turn

'''
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
from torch.nn import CrossEntropyLoss
import json
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler
from copy import deepcopy
import horovod.torch as hvd
from torch.utils.tensorboard import SummaryWriter
from models.classification import NLPClassification

args = None
optim = None
tb = None


class NLPDataset(torch.utils.data.Dataset):

    def __init__(self, df, tokenizer):
        super(NLPDataset, self).__init__()
        self.data = {
            "sentence1": df['sentence1'],
        }
        name_to_id = dict([(it, i) for i, it in enumerate(args.label_list)])
        self.data["labels"] = df['labels'].map(name_to_id)
        self.data["labels"].fillna(0, inplace=True)
        if 'sentence2' in df.columns:
            self.data['sentence2'] = df['sentence2']
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data['sentence1'])

    def __getitem__(self, index):
        text1 = str((self.data)['sentence1'][index])

        if 'sentence2' in self.data.keys():
            text2 = str((self.data)['sentence2'][index])

            inputs = self.tokenizer.encode_plus(
                text1,
                text2,
                add_special_tokens=True,
                max_length=args.max_len,
                padding="max_length",
                return_token_type_ids=True,
                truncation=True,
            )
        else:
            inputs = self.tokenizer.encode_plus(
                text1,
                None,
                add_special_tokens=True,
                max_length=args.max_len,
                padding="max_length",
                return_token_type_ids=True,
                truncation=True
            )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'labels': torch.tensor((self.data)['labels'][index], dtype=torch.long, )
        }


def get_dataloader(df, tokenizer, is_training=True):
    dt = NLPDataset(df, tokenizer)

    params = {
        "batch_size": args.batch_size,
        "pin_memory": True,
        "num_workers": 4,
    }
    if not is_training:
        return torch.utils.data.DataLoader(dt, **params), None

    sampler = torch.utils.data.distributed.DistributedSampler(
        dt, num_replicas=hvd.size(), rank=hvd.rank())
    params['sampler'] = sampler
    return torch.utils.data.DataLoader(dt, **params), sampler


def train(model, train_loader, train_sampler):
    device = args.device
    if os.path.isfile(args.model_save_path):
        model.load_state_dict(torch.load(args.model_save_path))
    if os.path.isfile(args.optim_save_path):
        optim.load_state_dict(torch.load(args.optim_save_path))
    model.train()
    all_steps = 0
    optim.zero_grad()
    for e in range(args.epochs):
        train_sampler.set_epoch(e)
        for step, data in enumerate(train_loader, start=1):
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            labels = data['labels'].to(device)
            loss = model(
                input_ids=ids,
                attention_mask=mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
            loss.div_(args.batchs_per_step)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if step % args.batchs_per_step != 0:
                continue
            optim.step()
            optim.zero_grad()
            all_steps += 1
            with torch.no_grad():
                if hvd.rank() == 0:
                    tb.add_scalar(f"{args.task_name}:Train/loss", loss.item(), global_step=all_steps)
                    print(f"epochs:{args.epochs}/{e},steps:{step},loss:{loss.item()}", flush=True)
        if hvd.rank() == 0:
            torch.save(model.state_dict(), args.model_save_path)
            torch.save(optim.state_dict(), args.optim_save_path)


def evaluate(model, eval_loader):
    if os.path.isfile(args.model_save_path):
        model.load_state_dict(torch.load(args.model_save_path))
    model.eval()
    device = args.device
    preds_all = []
    labels_all = []
    with torch.no_grad():
        for data in eval_loader:
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            labels = data['labels']

            outputs = model(
                input_ids=ids,
                attention_mask=mask,
                token_type_ids=token_type_ids)
            preds = torch.argmax(outputs, dim=1)
            preds_all.append(preds.cpu().numpy())
            labels_all.append(labels.cpu().numpy())

    labels = np.concatenate(labels_all, axis=-1)
    preds = np.concatenate(preds_all, axis=-1)
    reports = classification_report(labels, preds, target_names=args.label_list)
    ret = {"accuracy": (preds == labels).astype(np.float32).mean(), 'reports': reports}
    print(ret)
    return ret


def predict(model, pred_loader):
    if os.path.isfile(args.model_save_path):
        model.load_state_dict(torch.load(args.model_save_path))
    model.eval()
    device = args.device
    preds_all = []
    with torch.no_grad():
        for data in pred_loader:
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            token_type_ids = data['token_type_ids'].to(device)

            outputs = model(
                input_ids=ids,
                attention_mask=mask,
                token_type_ids=token_type_ids)
            preds = torch.argmax(outputs, dim=1)
            preds_all.append(preds.cpu().numpy())
    preds = np.concatenate(preds_all, axis=-1)
    preds = args.label_list[preds]
    df = pd.DataFrame()
    df['labels'] = preds
    df.to_csv(args.predict_save_path, index=False)
    return df


if __name__ == '__main__':
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument('--base_lr', default=4e-5, type=float)
    parser.add_argument('--max_len', default=128)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--do_predict', action='store_true')

    parser.add_argument('--device', default='cuda')
    parser.add_argument("--train_df_path", type=str)
    parser.add_argument("--eval_df_path", type=str)
    parser.add_argument("--predict_df_path", type=str)
    parser.add_argument("--logdir", type=str, default="./log")
    parser.add_argument("--task_name", type=str, default="tnews-classfication")

    parser.add_argument("--model_save_path", type=str)
    parser.add_argument("--optim_save_path", type=str)
    parser.add_argument("--predict_save_path", type=str, default='./predict_result.csv')
    parser.add_argument("--batchs_per_step", type=int, default=3)
    parser.add_argument("--compression_fp16", action='store_true')
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--wd', type=float, default=0.00005,
                        help='weight decay')

    parser.add_argument("--label_list", nargs='+', type=list,
                        default=['news_entertainment', 'news_military', 'news_finance', 'news_tech', 'news_travel',
                                 'news_culture', 'news_house', 'news_edu',
                                 'news_agriculture', 'news_car', 'news_sports', 'news_world', 'news_game', 'news_story',
                                 'news_stock'])

    parser.add_argument('--base_model', default='hfl/chinese-macbert-large')
    parser.add_argument('--tokenizer_model', default='bert-base-chinese', type=str)

    parser.add_argument('--epochs', default=10, type=int)

    args = parser.parse_args()
    if args.device == 'cuda':
        args.device = 'cuda' if cuda.is_available() else 'cpu'
    args.num_labels = len(args.label_list)
    args.label_list = np.array(args.label_list)

    print(args)

    model = NLPClassification(**args).to(args.device)
    # model = torch.nn.DataParallel(model).to(args.device)

    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_model)
    train_df = pd.read_csv(args.train_df_path)
    eval_df = pd.read_csv(args.eval_df_path)
    pred_df = pd.read_csv(args.predict_df_path)

    train_loader, train_sampler = get_dataloader(train_df, tokenizer, is_training=True)
    eval_loader, _ = get_dataloader(eval_df, tokenizer, is_training=False)
    pred_loader, _ = get_dataloader(pred_df, tokenizer, is_training=False)

    optim = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr * hvd.local_size() * args.batchs_per_step,
        momentum=args.momentum,
        weight_decay=args.wd
    )
    optim = hvd.DistributedOptimizer(optim,
                                     named_parameters=model.named_parameters(),
                                     backward_passes_per_step=args.batchs_per_step,
                                     compression=hvd.Compression.fp16 if args.compression_fp16 else hvd.Compression.none
                                     )
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    if hvd.rank() == 0:
        tb = SummaryWriter(args.logdir)

    if args.do_train:
        train(model, train_loader, train_sampler)

    if args.do_eval and hvd.rank() == 0:
        evaluate(model, eval_loader)

    if args.do_predict and hvd.rank() == 0:
        predict(model, pred_loader)

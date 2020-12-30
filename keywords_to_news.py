import os
import pandas as pd
import numpy as np
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
from utils import checkpointer
from models.generater import ConditionalGenerationModel

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
        keywords = [e for e in text1.split(",") if e.strip()]
        lt1 = []
        for key in keywords:
            lt1.extend(args.tokenizer.encode(key, add_special_tokens=False))
            lt1.append(args.tokenizer.sep_token_id)

        lt2 = args.tokenizer.encode(labels, add_special_tokens=False)
        max_len1 = args.max_len - len(lt1) - 2

        lt2 = lt2[:max_len1] if len(lt2) > max_len1 else lt2 + [args.tokenizer.pad_token_id] * (max_len1 - len(lt1))
        ret = {
            "input_ids": torch.tensor([args.tokenizer.cls_token_id, ] + lt1 + [
                args.tokenizer.convert_tokens_to_ids("[GENERATE_ARTICLE]")] + lt2),
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


def train(model, loader, sampler):
    device = args.device
    all_steps = 0
    epoch = -1
    if os.path.isfile(args.checkpoint_save_path):
        mp = checkpointer.load(args.checkpoint_save_path, return_dict=True)
        model.load_state_dict(mp['model_state'])
        optim.load_state_dict(mp['optim_state'])
        all_steps = mp['all_steps']
        epoch = mp['epoch']
    model.train()
    optim.zero_grad()
    scaler = GradScaler()
    for e in range(epoch + 1, args.epochs):
        sampler.set_epoch(e)
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

            if step % args.batches_per_allreduce != 0:
                continue
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optim.step()
            optim.zero_grad()
            all_steps += 1
            gradual_decay(epoch=e, steps=all_steps % len(loader), all_steps=len(loader))

            with torch.no_grad():
                if hvd.rank() == 0:
                    tb.add_scalar(f"{args.task_name}:", loss.item(), global_step=all_steps)
                    print(f"epochs:{args.epochs}/{e},steps:{all_steps},loss:{loss.item()}", flush=True)
            if hvd.rank() == 0 and all_steps % args.save_steps == 0:
                checkpointer.save(args.checkpoint_save_path, model.state_dict(), optim.state_dict(), e, all_steps)
    if hvd.rank() == 0:
        checkpointer.save(args.checkpoint_save_path, model.state_dict(), optim.state_dict(), e, all_steps)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=20, type=int)
    parser.add_argument('--base_lr', default=4e-4, type=float)

    parser.add_argument('--max_len', default=512)
    parser.add_argument('--do_train', action='store_true', default=False)
    parser.add_argument('--do_predict', action='store_true', default=False)

    parser.add_argument('--device', default='cuda')
    parser.add_argument("--train_df_path", type=str)
    parser.add_argument("--logdir", type=str, default="./log")
    parser.add_argument("--task_name", type=str, default="news_generate")
    parser.add_argument("--checkpoint_save_path", type=str, help='model optim epoch all_step save')

    parser.add_argument('--base_model', default='hfl/chinese-macbert-large')
    parser.add_argument('--tokenizer_model', default='bert-base-chinese', type=str)

    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--save_steps', default=1000, type=int)

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
    tokenizer.add_tokens("[GENERATE_ARTICLE]")
    args.tokenizer = tokenizer
    args.rank_size = hvd.size()
    print(args)

    train_df = pd.read_csv(args.train_df_path)
    train_loader, train_sampler = get_dataLoader(train_df)

    model = ConditionalGenerationModel(**args).to(args.device)
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

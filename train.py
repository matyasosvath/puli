from typing import Dict, List

import os
import time
import torch
import random
import tiktoken
import numpy as np
from tqdm import tqdm
from transformers import get_scheduler
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group

from . import config
from puli2.model import ModelArgs, GPTModel
from puli2.tokenizer import Tokenizer


TRAIN_PATH = "./input.txt"
TEST_PATH = "./input.txt"


def setup(
    rank: int, # a unique process ID
    world_size: int # total number of processes in the group
) -> None:

    os.environ['MASTER_ADDR'] = 'localhost' # assume all GPUs are on the same machine
    os.environ['MASTER_PORT'] = '12345' # any free port on the machine

    # nccl: NVIDIA Collective Communication Library
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


class GPTDataset(Dataset):

    def __init__(self, text: str, tokenizer, max_length: int, stride: int) -> None:

        self.tokenizer =tokenizer
        self.max_length = max_length
        self.stride = stride

        self.input_ids = []
        self.target_ids = []

        token_ids = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})

        # sliding window to chunk data into overlapping sequences of max_length
        for i in range(0, len(token_ids) - self.max_length, self.stride):

            input_chunk = token_ids[i : i + self.max_length]
            target_chunk = token_ids[i + 1 : i + self.max_length + 1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int):
        return self.input_ids[idx], self.target_ids[idx]


def train(
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    model: torch.nn.Module,
    model_args: ModelArgs,
    gpu
) -> Dict[str, List]:

    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "token_seen": []
    }

    model = model.to(gpu)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=config.PAD_IDX)
    optimizer = torch.optim.AdamW(model.parameters(), model_args.lr, model_args.betas, model_args.eps)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    for epoch in tqdm(range(config.EPOCHS)):

        epoch_start_time = time.time()

        train_loss, train_acc, token_seen = train_step(model, train_dataloader, loss_fn, optimizer, gpu=gpu)

        torch.cuda.empty_cache()

        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, gpu)

        scheduler.step()

        torch.save(model.state_dict(), f"puli_gpt2_{epoch}.pt")

        torch.cuda.empty_cache()

        print(
          f"Epoch: {epoch+1} | "
          f"Time: {time.time() - epoch_start_time} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Test Loss: {test_loss:.4f} | "
          f"Train Acc: {train_acc:.4f} | "
          f"Test Acc: {test_acc:.4f} | "
          f"Token seen: {token_seen:.4f} | "
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        results["token_seen"].append(token_seen)

    return results


def train_step(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    gpu: int,
):

    model.train()

    train_loss = 0
    train_acc = 0
    perplexity = 0
    token_seen = 0

    for batch, (src, tgt) in enumerate(dataloader):

        src = src.to(gpu)
        tgt = tgt.to(gpu)

        logits = model(src)

        optimizer.zero_grad()

        # logits := (batch_size, context_length, vocab_size) -> (batch_size x context_length, vocab_size)
        # tgt := (batch_size, context_length) -> (batch_size x context_length)
        loss = loss_fn(logits.view(-1, logits.size(-1)), tgt.view(-1))

        # preds := (batch_size, context_length)
        _, preds = torch.max(logits, dim=-1)
        compare = torch.eq(preds, tgt)
        acc = (compare.sum().float() / len(compare.view(-1))) * 100

        perplexity += torch.exp(loss)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)

        optimizer.step()

        train_loss += loss.item()
        train_acc += acc.item()
        token_seen += src.numel()

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    perplexity = perplexity / len(dataloader)

    return train_loss, train_acc, token_seen


def test_step(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn: torch.nn.Module,
    gpu: int
):

    model.eval()

    test_loss = 0
    test_acc = 0
    perplexity = 0

    with torch.inference_mode():
        for batch, (src, tgt) in enumerate(dataloader):

            src = src.to(gpu)
            tgt = tgt.to(gpu)

            logits = model(src)

            # (batch_size, context_length, vocab_size) -> (batch_size x context_length, vocab_size)
            logits = logits.view(-1, logits.size(-1))
            # (batch_size, context_length) -> (batch_size x context_length)
            tgt = tgt.view(-1)

            loss = loss_fn(logits, tgt)

            # preds := (batch_size, context_length)
            _, preds = torch.max(logits, dim=-1)
            compare = torch.eq(preds, tgt)
            acc = (compare.sum().float() / len(compare.view(-1))) * 100

            test_loss += loss.item()
            test_acc += acc.item()

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)

    return test_loss, test_acc


def main(rank: int, world_size: int, model_args: ModelArgs) -> None:

    setup(rank, world_size)  # initialize process groups

    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        train_data = f.read()
        print(f'Training data has {len(train_data)} characters.')

    with open(TEST_PATH, "r", encoding="utf-8") as f:
        test_data = f.read()
        print(f'Test data has {len(test_data)} characters.')


    tokenizer = tiktoken.get_encoding("gpt2")

    train_dataset = GPTDataset(train_data, tokenizer, model_args.context_length, model_args.context_length // 2)
    test_dataset = GPTDataset(test_data, tokenizer, model_args.context_length, model_args.context_length // 2)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    eval_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=model_args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=model_args.batch_size,
        sampler=eval_sampler,
        shuffle=(eval_sampler is None),
    )

    model = GPTModel(model_args)
    model.to(rank)
    model = DDP(model, device_ids=[rank])

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=config.PAD_IDX)
    optimizer = torch.optim.AdamW(model.parameters(), model_args.lr, model_args.betas, model_args.eps)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    pre_epoch = 0
    best_epoch = 0
    min_eval_loss = float('inf')

    for epoch in range(1+pre_epoch, config.EPOCHS+1):

        train_sampler.set_epoch(epoch)
        eval_sampler.set_epoch(epoch)

        epoch_start_time = time.time()

        train_loss, train_acc, token_seen = train_step(model, train_dataloader, loss_fn, optimizer, rank)

        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, rank)

        lr_scheduler.step()

        if rank==0:
            print(
            f"Epoch: {epoch} | "
            f"Time: {time.time() - epoch_start_time} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Test Acc: {test_acc:.4f} | "
            f"Token seen: {token_seen:.4f} | "
            )

        if test_loss < min_eval_loss:

            best_epoch = epoch
            min_eval_loss = test_loss
            checkpoint = {
                            'model': model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_sched': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'best_epoch': best_epoch,
                            'min_eval_loss': min_eval_loss
                            }
            torch.save(checkpoint, config.MODEL_PATH)

        if world_size > 1:
            dist.barrier()

    destroy_process_group()


if __name__ == "__main__":

    print("CUDA available:", torch.cuda.is_available())
    print("Number of GPUs available:", torch.cuda.device_count())

    # spawn new processes

    model_args = ModelArgs()

    world_size = torch.cuda.device_count()

    mp.spawn(
        main,
        args=(world_size, model_args), # spawn automatically passes rank param
        nprocs=world_size # world_size spawns one process per GPU
    )
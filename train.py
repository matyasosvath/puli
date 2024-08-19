from typing import Dict, List

import os
import time
import torch
import random
import tiktoken
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from puli_gpt2 import config
from puli_gpt2.model import ModelArgs, GPTModel
from puli_gpt2.tokenizer import Tokenizer


TRAIN_PATH = "./input.txt"
TEST_PATH = "./input.txt"


world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else 0
local_rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else 0

if world_size > 1:
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend='nccl') if world_size > 1 else None
else:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if config.DETERMINISTIC:
    seed = 0 + global_rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


model_args = ModelArgs()

model = GPTModel(model_args)
model = model.to(device)


tokenizer = tiktoken.get_encoding("gpt2")

if world_size > 1:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,  find_unused_parameters=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=model_args.lr)


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
    gpu: int
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



if __name__ == "__main__":

    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        train_data = f.read()
        print(f'Training data has {len(train_data)} characters.')

    with open(TEST_PATH, "r", encoding="utf-8") as f:
        test_data = f.read()
        print(f'Test data has {len(test_data)} characters.')

    train_dataset = GPTDataset(train_data, tokenizer, model_args.context_length, model_args.context_length // 2)
    test_dataset = GPTDataset(test_data, tokenizer, model_args.context_length, model_args.context_length // 2)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
    eval_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=local_rank)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=model_args.batch_size,
        pin_memory=True,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=model_args.batch_size,
        sampler=eval_sampler,
        shuffle=(eval_sampler is None),
    )

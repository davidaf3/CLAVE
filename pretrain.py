import math
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from tokenizer import SPECIAL_TOKENS, MASK_TOK, PADDING_TOK
from utils import save_checkpoint, load_latest_checkpoint, linear_with_warmup_scheduler
from model import PretrainingModel
from pretraining_data_loader import PretrainingDataset
from config import PRETRAINING_DATASET_PATH, TOKENIZER


def collate_fn(examples):
    return torch.stack(examples, dim=0)


def add_mask(batch):
    mask = torch.rand(batch.size(), device="cuda") < 0.15
    mask &= batch > SPECIAL_TOKENS
    to_mask = mask & (
        torch.rand(batch.size(), device="cuda") < 0.9
    )  # 90% to mask or set to random
    to_random = to_mask & (
        torch.rand(batch.size(), device="cuda") < 1 / 9
    )  # 10% to set to a random value

    masked = batch.detach().clone()
    masked[to_mask] = MASK_TOK
    masked[to_random] = 0
    randoms = torch.randint(
        SPECIAL_TOKENS, TOKENIZER.get_vocab_size(), batch.size(), device="cuda"
    )
    randoms[~to_random] = 0
    masked += randoms

    return masked, mask


def masked_accuracy(prediction, targets, mask):
    predicted_idxs = torch.argmax(prediction, dim=2)
    accuracy_matrix = predicted_idxs == targets
    accuracy_matrix[~mask] = False
    return torch.sum(accuracy_matrix) / torch.sum(mask)


def evaluate(
    model: nn.Module, eval_dataloader: DataLoader, criterion: nn.Module
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    batches = 0
    with torch.inference_mode():
        for batch in eval_dataloader:
            data = batch.cuda()
            inputs, mask = add_mask(data)

            attn_mask = inputs.ne(PADDING_TOK).logical_and(inputs.ne(MASK_TOK))
            fully_masked = torch.sum(attn_mask, dim=1).eq(0).sum()
            if fully_masked.item() != 0:
                continue

            targets = data
            targets[~mask] = PADDING_TOK

            output = model(inputs)
            output_flat = output.view(-1, ntokens)
            total_loss += criterion(output_flat, targets.view(-1)).item()
            total_acc += masked_accuracy(output, targets, mask).item()
            batches += 1

    return total_loss / batches, total_acc / batches


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
) -> None:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    skipped = 0
    log_interval = 10
    start_time = time.time()

    num_batches = math.ceil(len(train_data) / batch_size)
    for i, batch in enumerate(train_dataloader):
        data = batch.cuda()
        inputs, mask = add_mask(data)

        attn_mask = inputs.ne(PADDING_TOK).logical_and(inputs.ne(MASK_TOK))
        fully_masked = torch.sum(attn_mask, dim=1).eq(0).sum()
        if fully_masked.item() != 0:
            skipped += 1
        else:
            targets = data
            targets[~mask] = PADDING_TOK
            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(inputs)
                output_flat = output.view(-1, ntokens)
                loss = criterion(output_flat, targets.view(-1))
                total_acc += masked_accuracy(output, targets, mask)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            total_loss += loss.item()

        if i % log_interval == 0 and i > 0 and log_interval > skipped:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / (log_interval - skipped)
            cur_loss = total_loss / (log_interval - skipped)
            cur_acc = total_acc / (log_interval - skipped)
            print(
                f"| epoch {epoch:3d} | {i:5d}/{num_batches:5d} batches | "
                f"lr {lr:.8f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:.4f} | acc {cur_acc:.4f} ",
                end="\r",
            )
            total_loss = 0
            total_acc = 0
            skipped = 0
            start_time = time.time()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    train_data = PretrainingDataset("train", PRETRAINING_DATASET_PATH)
    val_data = PretrainingDataset("val", PRETRAINING_DATASET_PATH)

    batch_size = 32
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=2,
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=8,
    )

    ntokens = TOKENIZER.get_vocab_size()  # size of vocabulary
    emsize = 512  # embedding dimension
    d_hid = 2048  # dimension of the feedforward network
    nlayers = 6  # number of encoder layers
    nhead = 8  # number of attention heads
    dropout = 0.1  # dropout probability
    model = PretrainingModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to("cuda")
    # model = torch.compile(model)

    criterion = nn.CrossEntropyLoss(ignore_index=PADDING_TOK)
    lr = 1e-4  # learning rate
    warmup_steps = 100000  # warmup steps
    max_steps = 1500000  # max steps until lr is 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    sched_lambda = linear_with_warmup_scheduler(max_steps, warmup_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, sched_lambda)
    scaler = torch.cuda.amp.GradScaler()

    checkpoint_path = "pretraining_checkpoints"
    epoch = load_latest_checkpoint(checkpoint_path, model, optimizer, scaler, scheduler)

    while True:
        epoch_start_time = time.time()
        train(model, train_dataloader, criterion, optimizer, scheduler, scaler)
        val_loss, val_acc = evaluate(model, val_dataloader, criterion)
        elapsed = time.time() - epoch_start_time
        print("-" * 89)
        print(
            f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
            f"valid loss {val_loss:.4f} | valid acc {val_acc:.4f} "
        )
        print("-" * 89)

        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            scaler,
            scheduler,
            epoch,
            val_loss,
            val_acc,
        )
        epoch += 1

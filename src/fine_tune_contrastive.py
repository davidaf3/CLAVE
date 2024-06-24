import os
import math
import time
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from gcj_data_loader import GCJDataset, GCJValDataset
from utils import save_checkpoint, load_latest_checkpoint, linear_with_warmup_scheduler
from model import FineTunedModel
from config import FINE_TUNING_DATASET_PATH, TOKENIZER, PRETRAINING_CHECKPOINT
from fine_tune import triplet_accuracy


class CollateFn:
    def __init__(self, dim) -> None:
        self.dim = dim

    def __call__(self, triplets):
        anchors = [triplet[0] for triplet in triplets]
        positives = [triplet[1] for triplet in triplets]
        negatives = [triplet[2] for triplet in triplets]
        return tuple(
            torch.stack(examples, dim=self.dim)
            for examples in (anchors, positives, negatives)
        )


def contrastive_loss(
    anchor_out: torch.Tensor,
    positive_out: torch.Tensor,
    negative_out: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    x1 = torch.concat([anchor_out, anchor_out])
    x2 = torch.concat([positive_out, negative_out])
    y = torch.concat(
        [
            torch.zeros((anchor_out.size(0),), device=anchor_out.device),
            torch.ones((anchor_out.size(0),), device=anchor_out.device),
        ]
    )
    dist = F.pairwise_distance(x1, x2)
    loss = (1 - y) * torch.square(dist) + y * torch.square(
        torch.clamp(margin - dist, min=0.0)
    )
    return torch.mean(loss)


def evaluate(model: nn.Module, eval_dataloader: DataLoader) -> tuple[float, float]:
    model.eval()
    total_loss = torch.tensor(0.0, device="cuda")
    total_acc = torch.tensor(0.0, device="cuda")
    batches = 0
    with torch.inference_mode():
        for batch in eval_dataloader:
            batches += 1
            anchor, positive, negative = tuple(examples.cuda() for examples in batch)

            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)
            total_loss += contrastive_loss(anchor_out, positive_out, negative_out)
            total_acc += triplet_accuracy(anchor_out, positive_out, negative_out)

    return total_loss.item() / batches, total_acc.item() / batches


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
) -> None:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    log_interval = 10
    start_time = time.time()

    num_batches = math.ceil(len(train_data) / batch_size)
    for i, batch in enumerate(train_dataloader):
        anchor, positive, negative = tuple(examples.cuda() for examples in batch)
        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)
            loss = contrastive_loss(anchor_out, positive_out, negative_out)
            acc = triplet_accuracy(anchor_out, positive_out, negative_out)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()

        scheduler.step()

        total_loss += loss.item()
        total_acc += acc.item()

        if i % log_interval == 0 and i > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_acc = total_acc / log_interval
            print(
                f"| epoch {epoch:3d} | {i:5d}/{num_batches:5d} batches | "
                f"lr {lr:.8f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:.4f} | acc {cur_acc:.4f} ",
                end="\r",
            )
            total_loss = 0.0
            total_acc = 0.0
            start_time = time.time()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    batch_size = 32
    train_data = GCJDataset(
        "train", FINE_TUNING_DATASET_PATH, size=20000 * batch_size, random_fragment=True
    )
    val_data = GCJValDataset(FINE_TUNING_DATASET_PATH)

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        collate_fn=CollateFn(0),
        pin_memory=True,
        num_workers=2,
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=batch_size,
        collate_fn=CollateFn(0),
        pin_memory=True,
        num_workers=8,
    )

    pretraining_checkpoint = torch.load(PRETRAINING_CHECKPOINT)
    pretrained_weights = {
        k[8 if k.startswith("e") else 18:]: v
        for k, v in pretraining_checkpoint["model_state_dict"].items()
        if k.startswith("encoder.") or k.startswith("_orig_mod.encoder.")
    }

    ntokens = TOKENIZER.get_vocab_size()  # size of vocabulary
    emsize = 512  # embedding dimension
    d_hid = 2048  # dimension of the feedforward network
    d_out = 512  # output dimension
    nlayers = 6  # number of encoder layers
    nhead = 8  # number of attention heads
    dropout = 0.1  # dropout probability
    use_layer_norm = True  # use layer norm
    model = FineTunedModel(
        ntokens,
        emsize,
        d_out,
        nhead,
        d_hid,
        nlayers,
        pretrained_weights,
        dropout,
        use_layer_norm,
    ).to("cuda")
    # model = torch.compile(model)

    lr = 3e-5  # learning rate
    warmup_steps = 40000  # warmup steps
    max_steps = 500000  # max steps until lr is 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    sched_lambda = linear_with_warmup_scheduler(max_steps, warmup_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, sched_lambda)
    scaler = torch.cuda.amp.GradScaler()

    epochs = 100
    checkpoint_path = "fine_tuning_checkpoints"
    start_epoch = load_latest_checkpoint(
        checkpoint_path, model, optimizer, scaler, scheduler
    )

    for epoch in range(start_epoch, epochs + 1):
        epoch_start_time = time.time()
        train(model, train_dataloader, optimizer, scheduler, scaler)
        val_loss, val_acc = evaluate(model, val_dataloader)
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

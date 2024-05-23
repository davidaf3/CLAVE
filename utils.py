import os
import re
import torch
import numpy as np
from config import SEQ_LENGTH
from tokenizer import PADDING_TOK
from typing import Callable


def save_checkpoint(
    path, model, optimizer, scaler, scheduler, epoch, val_loss, val_acc
):
    path = os.path.join(
        path, f"checkpoint-{epoch}-val_loss-{val_loss:.4f}-val_acc-{val_acc:.4f}"
    )
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler is not None else None
            ),
        },
        path,
    )


def load_latest_checkpoint(path, model, optimizer, scaler, scheduler):
    if not os.path.exists(path):
        os.mkdir(path)

    checkpoints = os.listdir(path)
    if len(checkpoints) == 0:
        return 1

    regex = "checkpoint-([0-9]+)"
    checkpoints = [
        (checkpoint, int(re.match(regex, checkpoint).groups()[0]))
        for checkpoint in checkpoints
    ]
    checkpoints = sorted(checkpoints, key=lambda pair: pair[1], reverse=True)

    checkpoint = torch.load(os.path.join(path, checkpoints[0][0]))
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint["epoch"] + 1


def pad_and_split_tokens(tokens) -> list[np.ndarray]:
    tokens = np.array(tokens, dtype=np.int64)
    pad_width = 0 if (mod := len(tokens) % SEQ_LENGTH) == 0 else SEQ_LENGTH - mod
    tokens = np.pad(tokens, (0, pad_width), constant_values=PADDING_TOK)
    return np.split(tokens, len(tokens) / SEQ_LENGTH)


def linear_with_warmup_scheduler(
    max_steps: int, warmup_steps: int
) -> Callable[[int], float]:
    return lambda step: (
        step / warmup_steps
        if step <= warmup_steps
        else max(0.0, (max_steps - step) / (max_steps - warmup_steps))
    )

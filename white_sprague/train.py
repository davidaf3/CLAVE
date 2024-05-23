import os
import sys
import time
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from simclr import loss as nt_xent, accuracy, SimCLRDataset
from utils import save_checkpoint, load_latest_checkpoint
from config import FINE_TUNING_DATASET_PATH, TOKENIZER, SEQ_LENGTH


class WhiteSpragueModel(nn.Module):
    def __init__(self, ntoken: int):
        super().__init__()
        self.model_type = "WhiteSpragueModel"
        self.embedding = nn.Embedding(ntoken, 32)
        self.dropout = nn.Dropout(0.2)
        self.bilstm = nn.LSTM(
            32, 128, num_layers=2, batch_first=True, bidirectional=True
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(src)
        output, _ = self.bilstm(self.dropout(embeddings))
        return torch.squeeze(output[:, src.size(1) - 1, :])


def evaluate(model: nn.Module, eval_dataloader: DataLoader) -> tuple[float, float]:
    model.eval()
    total_loss = torch.tensor(0.0, device="cuda")
    total_acc = torch.tensor(0.0, device="cuda")
    batches = 0
    with torch.inference_mode():
        for batch in eval_dataloader:
            batches += 1
            batch = batch.cuda()
            output = model(batch)
            total_loss += nt_xent(output)
            total_acc += accuracy(output)

    return total_loss.item() / batches, total_acc.item() / batches


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
) -> None:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    log_interval = 10
    start_time = time.time()

    num_batches = len(train_data)
    for i, batch in enumerate(train_dataloader):
        batch = batch.cuda()
        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(batch)
            loss = nt_xent(output)
            acc = accuracy(output)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_acc += acc.item()

        if i % log_interval == 0 and i > 0:
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


def collate_fn(batch):
    return batch[0]


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    batch_size = 128
    train_data = SimCLRDataset(
        "train", os.path.join("..", FINE_TUNING_DATASET_PATH), batch_size, size=20000
    )
    val_data = SimCLRDataset(
        "val_no_triplets",
        os.path.join("..", FINE_TUNING_DATASET_PATH),
        batch_size,
        size=2000,
    )

    train_dataloader = DataLoader(
        train_data,
        batch_size=1,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=8,
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=1,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=8,
    )

    ntokens = TOKENIZER.get_vocab_size()  # size of vocabulary
    model = WhiteSpragueModel(ntokens).to("cuda")
    # model = torch.compile(model)

    lr = 0.001  # learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=2.5e-06)
    scaler = torch.cuda.amp.GradScaler()

    epochs = 20
    checkpoint_path = os.path.join("checkpoints", f"{SEQ_LENGTH}_4_layer")
    start_epoch = load_latest_checkpoint(
        checkpoint_path, model, optimizer, scaler, None
    )

    for epoch in range(start_epoch, epochs + 1):
        epoch_start_time = time.time()
        train(model, train_dataloader, optimizer, scaler)
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
            None,
            epoch,
            val_loss,
            val_acc,
        )

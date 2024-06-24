import os
import sys
import math
import time
import zarr
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils import save_checkpoint, load_latest_checkpoint


class FDRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_type = "FDRModel"
        self.dropout = nn.Dropout()
        self.linear1 = nn.Linear(8000, 500)
        self.activation1 = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(500)
        self.linear2 = nn.Linear(500, 160)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.linear1.bias.data.zero_()
        self.linear1.weight.data.uniform_(-initrange, initrange)
        self.linear2.bias.data.zero_()
        self.linear2.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor) -> Tensor:
        x = src.to(torch.float32)
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.batch_norm(x)
        x = self.linear2(x)
        return x


class FDRDataset(Dataset):
    def __init__(self, file, examples_per_class):
        self.data = zarr.open_array(file, mode="r")
        self.examples_per_class = examples_per_class

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        label = idx // self.examples_per_class
        return torch.tensor(self.data[idx]), torch.tensor(label)


def accuracy(prediction, targets):
    predicted_idxs = torch.argmax(prediction, dim=1)
    return torch.sum(predicted_idxs == targets) / targets.size(0)


def evaluate(
    model: nn.Module, eval_dataloader: DataLoader, criterion: nn.Module
) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.0
    total_acc = 0.0
    batches = 0
    with torch.no_grad():
        for batch in eval_dataloader:
            batches += 1
            inputs, targets = tuple(e.cuda() for e in batch)
            output = model(inputs)
            total_loss += criterion(output, targets).item()
            total_acc += accuracy(output, targets).item()
    return total_loss / batches, total_acc / batches


def collate_fn(examples):
    inputs = [e[0] for e in examples]
    targets = [e[1] for e in examples]
    return torch.stack(inputs), torch.stack(targets)


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
) -> None:
    model.train()  # turn on train mode
    total_loss = 0.0
    total_acc = 0.0
    log_interval = 10
    start_time = time.time()

    num_batches = math.ceil(len(train_data) / batch_size)
    for i, batch in enumerate(train_dataloader):
        inputs, targets = tuple(e.cuda() for e in batch)
        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(inputs)
            loss = criterion(output, targets)
            total_acc += accuracy(output, targets)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        if i % log_interval == 0 and i > 0:
            # lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_acc = total_acc / log_interval
            print(
                f"| epoch {epoch:3d} | {i:5d}/{num_batches:5d} batches | "
                f"lr {lr:.6f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:.4f} | acc {cur_acc:.4f} ",
                end="\r",
            )
            total_loss = 0
            total_acc = 0
            start_time = time.time()


if __name__ == "__main__":
    train_data = FDRDataset("train.zarr", 8)
    val_data = FDRDataset("val.zarr", 2)

    batch_size = 48
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=1,
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=1,
    )

    model = FDRModel().to("cuda")

    criterion = nn.CrossEntropyLoss()
    lr = 1e-4  # learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler()

    checkpoint_path = "checkpoints"
    epoch = load_latest_checkpoint(checkpoint_path, model, optimizer, scaler, None)
    best_val_loss = float("inf")
    patience = 5

    while True:
        epoch_start_time = time.time()
        train(model, train_dataloader, criterion, optimizer, scaler)
        val_loss, val_acc = evaluate(model, val_dataloader, criterion)
        elapsed = time.time() - epoch_start_time
        print("-" * 89)
        print(
            f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | "
            f"valid loss {val_loss:.4f} | valid acc {val_acc:.4f} "
        )
        print("-" * 89)

        save_checkpoint(
            checkpoint_path, model, optimizer, scaler, None, epoch, val_loss, val_acc
        )

        epoch += 1
        patience -= 1
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 5
        elif patience == 0:
            break

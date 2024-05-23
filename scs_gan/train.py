import os
import sys
import json
import math
import time
import zarr
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from gcj_data_loader import GCJDataset, GCJValDataset
from utils import save_checkpoint, load_latest_checkpoint
from tokenizer import PADDING_TOK
from config import FINE_TUNING_DATASET_PATH, TOKENIZER


class SCSGanDataset(GCJDataset):
    def __init__(self, part: str, base_dir: str, size=None):
        super().__init__(part, base_dir, size, random_fragment=False)
        self.codebert_embeddings = zarr.open_array(
            os.path.join(base_dir, f"codebert_embeddings_{part}.zarr"), mode="r"
        )

        metadata_file = os.path.join(
            base_dir, f"user_codebert_embeddings_idxs_{part}.json"
        )
        with open(metadata_file, "r", encoding="UTF-8") as f:
            self.user_codebert_embeddings_idxs = json.load(f)

    def get_codebert_embedding(self, user: str, idx: int) -> torch.Tensor:
        return torch.tensor(
            self.codebert_embeddings[self.user_codebert_embeddings_idxs[user][idx]]
        )

    def __getitem__(self, _):
        return tuple(
            (self.get_codebert_embedding(user, idx), tensor)
            for user, idx, tensor in self.get_triplet_tensors()
        )


class SCSGanValDataset(GCJValDataset):
    def __init__(self, base_dir: str):
        super().__init__(base_dir)
        self.codebert_embeddings = zarr.open_array(
            os.path.join(base_dir, "codebert_embeddings_val.zarr"), mode="r"
        )

    def __getitem__(self, idx):
        examples = super().__getitem__(idx)
        return tuple(
            (torch.tensor(self.codebert_embeddings[(3 * idx) + i]), examples[i])
            for i in range(3)
        )


class SCSGan(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int):
        super().__init__()
        self.d_model = d_model
        self.model_type = "SCSGan"
        self.embedding = nn.Embedding(ntoken, d_model)
        self.bilstm = nn.LSTM(d_model, d_model, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model * 2, d_model * 2)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(d_model * 2, 1)
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.linear1.bias.data.zero_()
        self.linear1.weight.data.uniform_(-initrange, initrange)
        self.linear2.bias.data.zero_()
        self.linear2.weight.data.uniform_(-initrange, initrange)

    def forward(
        self, srcs: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        if not isinstance(srcs, tuple):
            srcs = (srcs,)

        outputs = []
        for src in srcs:
            padding_mask = src.eq(PADDING_TOK)
            embeddings = self.embedding(src)
            output, _ = self.bilstm(embeddings)
            output = output[:, :, : self.d_model] + output[:, :, self.d_model :]
            output, _ = self.attention(
                output, output, output, key_padding_mask=padding_mask
            )
            output = torch.sum(output, dim=1)
            outputs.append(output)

        if len(outputs) == 1:
            return outputs[0]

        fc_output = self.linear1(torch.concat(outputs, dim=1))
        fc_output = self.activation(fc_output)
        fc_output = self.linear2(fc_output)
        return *outputs, fc_output.squeeze()


class CollateFn:
    def __init__(self, dim) -> None:
        self.dim = dim

    def __call__(self, triplets):
        anchors_embeddings = [triplet[0][0] for triplet in triplets]
        anchors = [triplet[0][1] for triplet in triplets]
        positives_embeddings = [triplet[1][0] for triplet in triplets]
        positives = [triplet[1][1] for triplet in triplets]
        negatives_embeddings = [triplet[2][0] for triplet in triplets]
        negatives = [triplet[2][1] for triplet in triplets]
        return tuple(
            torch.stack(examples, dim=self.dim)
            for examples in (
                anchors_embeddings,
                anchors,
                positives_embeddings,
                positives,
                negatives_embeddings,
                negatives,
            )
        )


def embedding_similarity(x1, x2):
    return torch.clamp((F.cosine_similarity(x1, x2) + 1) / 2, 0, 1)


def loss_fn(
    anchor_out,
    positive_out,
    negative_out,
    positive_fc_out,
    negative_fc_out,
    anchor_codebert_embedding,
    positive_codebert_embedding,
    negative_codebert_embedding,
):
    sim = torch.concat(
        [
            embedding_similarity(anchor_out, positive_out),
            embedding_similarity(anchor_out, negative_out),
        ]
    )
    y = torch.concat(
        [
            torch.ones((anchor_out.size(0),), device=anchor_out.device),
            torch.zeros((anchor_out.size(0),), device=anchor_out.device),
        ]
    )
    score_adv = torch.concat([positive_fc_out, negative_fc_out])
    y_adv = torch.concat(
        [
            embedding_similarity(
                anchor_codebert_embedding, positive_codebert_embedding
            ),
            embedding_similarity(
                anchor_codebert_embedding, negative_codebert_embedding
            ),
        ]
    )
    return 0.8 * F.binary_cross_entropy(
        sim, y
    ) + 0.2 * F.binary_cross_entropy_with_logits(score_adv, y_adv)


def triplet_accuracy(anchor_out, positive_out, negative_out):
    positive_sim = embedding_similarity(anchor_out, positive_out)
    negative_sim = embedding_similarity(anchor_out, negative_out)
    return torch.sum(positive_sim > negative_sim) / anchor_out.size(0)


def evaluate(model: nn.Module, eval_dataloader: DataLoader) -> float:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    batches = 0
    with torch.inference_mode():
        for batch in eval_dataloader:
            batches += 1
            (
                anchor_embeddings,
                anchor,
                positive_embeddings,
                positive,
                negative_embeddings,
                negative,
            ) = tuple(examples.cuda() for examples in batch)

            anchor_out, positive_out, positive_fc_out = model((anchor, positive))
            _, negative_out, negative_fc_out = model((anchor, negative))
            total_loss += loss_fn(
                anchor_out,
                positive_out,
                negative_out,
                positive_fc_out,
                negative_fc_out,
                anchor_embeddings,
                positive_embeddings,
                negative_embeddings,
            ).item()
            total_acc += triplet_accuracy(anchor_out, positive_out, negative_out).item()

    return total_loss / batches, total_acc / batches


def train(
    model: nn.Module, train_dataloader: DataLoader, optimizer: torch.optim.Optimizer
) -> None:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    log_interval = 10
    start_time = time.time()

    num_batches = math.ceil(len(train_data) / batch_size)
    for i, batch in enumerate(train_dataloader):
        (
            anchor_embeddings,
            anchor,
            positive_embeddings,
            positive,
            negative_embeddings,
            negative,
        ) = tuple(examples.cuda() for examples in batch)

        optimizer.zero_grad()

        anchor_out, positive_out, positive_fc_out = model((anchor, positive))
        _, negative_out, negative_fc_out = model((anchor, negative))
        loss = loss_fn(
            anchor_out,
            positive_out,
            negative_out,
            positive_fc_out,
            negative_fc_out,
            anchor_embeddings,
            positive_embeddings,
            negative_embeddings,
        )
        total_acc += triplet_accuracy(anchor_out, positive_out, negative_out).item()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()

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


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    batch_size = 32
    train_data = SCSGanDataset(
        "train", FINE_TUNING_DATASET_PATH, size=20000 * batch_size
    )
    val_data = SCSGanValDataset(FINE_TUNING_DATASET_PATH)

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        collate_fn=CollateFn(0),
        pin_memory=True,
        num_workers=4,
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=batch_size,
        collate_fn=CollateFn(0),
        pin_memory=True,
        num_workers=8,
    )

    ntokens = TOKENIZER.get_vocab_size()  # size of vocabulary
    emsize = 128  # embedding dimension
    nhead = 16  # number of attention heads
    model = SCSGan(ntokens, emsize, nhead).to("cuda")
    # model = torch.compile(model)

    lr = 0.001  # learning rate
    optimizer = torch.optim.NAdam(model.parameters(), lr=lr)

    epochs = 100
    checkpoint_path = "checkpoints"
    start_epoch = load_latest_checkpoint(checkpoint_path, model, optimizer, None, None)

    for epoch in range(start_epoch, epochs + 1):
        epoch_start_time = time.time()
        train(model, train_dataloader, optimizer)
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
            None,
            None,
            epoch,
            val_loss,
            val_acc,
        )

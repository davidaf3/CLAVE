import os
import random
import json
import zarr
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset


def _get_nt_xent_logits(
    x: torch.Tensor, temperature: float
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = x.size(0) // 2
    x = F.normalize(x, dim=-1)
    x1, x2 = torch.split(x, batch_size, 0)

    masks = F.one_hot(torch.arange(batch_size, device=x.device), batch_size)
    LARGE_NUM = 1e9

    logits_aa = x1 @ x1.t() / temperature
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = x2 @ x2.t() / temperature
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = x1 @ x2.t() / temperature
    logits_ba = x2 @ x1.t() / temperature
    return (
        torch.concat([logits_ab, logits_aa], 1),
        torch.concat([logits_ba, logits_bb], 1),
    )


def loss(x: torch.Tensor, temperature=0.005) -> torch.Tensor:
    batch_size = x.size(0) // 2
    labels = F.one_hot(torch.arange(batch_size, device=x.device), batch_size * 2).to(
        torch.float
    )
    logits_a, logits_b = _get_nt_xent_logits(x, temperature)
    loss_a = F.cross_entropy(logits_a, labels)
    loss_b = F.cross_entropy(logits_b, labels)
    return torch.mean(loss_a + loss_b)


def accuracy(x: torch.Tensor, temperature=0.005) -> torch.Tensor:
    batch_size = x.size(0) // 2
    labels = torch.arange(batch_size, device=x.device)
    logits_a, logits_b = _get_nt_xent_logits(x, temperature)
    acc_a = torch.mean((torch.argmax(logits_a, dim=1) == labels).to(torch.float))
    acc_b = torch.mean((torch.argmax(logits_b, dim=1) == labels).to(torch.float))
    return (acc_a + acc_b) / 2


class SimCLRDataset(Dataset):
    def __init__(
        self,
        part: str,
        base_dir: str,
        batch_size: int,
        random_fragment=False,
        size=None,
    ):
        self.data = zarr.open_array(os.path.join(base_dir, f"{part}.zarr"), mode="r")
        self.batch_size = batch_size
        self.random_fragment = random_fragment
        self.size = size

        metadata_file = os.path.join(base_dir, f"user_submission_idxs_{part}.json")
        with open(metadata_file, "r", encoding="UTF-8") as f:
            self.user_submission_idxs = json.load(f)

        self.candidates = [
            user
            for user, solutions in self.user_submission_idxs.items()
            if len(solutions) > 1
        ]

    def __len__(self):
        return self.size if self.size != None else self.data.shape[0]

    def __getitem__(self, _):
        batch = np.zeros((self.batch_size * 2, self.data.shape[1]), dtype=np.int64)
        authors = random.sample(self.candidates, self.batch_size)
        for i, author in enumerate(authors):
            [_, fragments1], [_, fragments2] = random.sample(
                self.user_submission_idxs[author], 2
            )
            batch[i] = self.data[
                random.choice(fragments1) if self.random_fragment else fragments1[0]
            ]
            batch[i + self.batch_size] = self.data[
                random.choice(fragments2) if self.random_fragment else fragments2[0]
            ]
        return torch.tensor(batch)

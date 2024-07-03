import os
import sqlite3
import json
import random
import numpy as np
import zarr
import torch
from collections import defaultdict
from config import SEQ_LENGTH, FINE_TUNING_DATASET_PATH, TOKENIZER
from torch.utils.data import Dataset
from typing import Literal
from utils import pad_and_split_tokens
from abc import ABCMeta, abstractmethod


DATASET_PATH = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "gcj")


def get_partition_data(part: str) -> list[tuple[str, str, str]]:
    with open(os.path.join(DATASET_PATH, "metadata.json"), "r", encoding="UTF-8") as f:
        by_user = json.load(f)

    data_flat = [
        (user, archive, solution)
        for user, by_archives in by_user.items()
        for archive, solutions in by_archives.items()
        for solution in solutions
    ]

    train_size = round(len(data_flat) * 0.8)
    # A user must not be in more than one partition
    while data_flat[train_size - 1][0] == data_flat[train_size][0]:
        train_size += 1

    val_size = round(len(data_flat) * 0.1)
    while (
        data_flat[train_size + val_size - 1][0] == data_flat[train_size + val_size][0]
    ):
        val_size += 1

    if part.startswith("train"):
        return data_flat[:train_size]
    if part.startswith("val"):
        return data_flat[train_size : train_size + val_size]
    return data_flat[train_size + val_size :]


class DatasetWriter(metaclass=ABCMeta):
    def __init__(
        self, part: str, base_dir: str, seq_length=SEQ_LENGTH, dtype=np.int64
    ) -> None:
        self.base_dir = base_dir
        self.part = part
        self.dataset = zarr.open_array(
            os.path.join(base_dir, f"{part}.zarr"),
            mode="w",
            shape=(0, seq_length),
            chunks=(1024, seq_length),
            dtype=dtype,
        )
        self.buffer = []

    @abstractmethod
    def get_solution(self, archive: str, solution: str) -> bytes:
        pass

    def write(self, sample: np.ndarray):
        self.buffer.append(sample)
        if len(self.buffer) == 102400:
            self.dataset.append(self.buffer)
            self.buffer = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if len(self.buffer) > 0:
            self.dataset.append(self.buffer)


class DbDatasetWriter(DatasetWriter):
    def __init__(
        self, part: str, base_dir: str, seq_length=SEQ_LENGTH, dtype=np.int64
    ) -> None:
        super().__init__(part, base_dir, seq_length, dtype)
        self.cons = {
            archive: sqlite3.connect(os.path.join(DATASET_PATH, "solutions", archive))
            for archive in os.listdir(os.path.join(DATASET_PATH, "solutions"))
        }

    def get_solution(self, archive: str, solution: str) -> bytes:
        con = self.cons[archive]
        cur = con.cursor()
        res = cur.execute(f"SELECT data FROM sqlar WHERE name = '{solution}' LIMIT 1")
        program = res.fetchone()[0]
        cur.close()
        return program

    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        for con in self.cons.values():
            con.close()


class FileDatasetWriter(DatasetWriter):
    def get_solution(self, archive: str, solution: str) -> bytes:
        with open(os.path.join(self.base_dir, archive, solution), mode="rb") as f:
            return f.read()


def write_dataset(part: str, writer: DatasetWriter):
    tokenizer = TOKENIZER()
    partition_data = get_partition_data(part)
    user_submission_idxs = defaultdict(list)
    i = 0

    with writer:
        for user, archive, solution in partition_data:
            program = writer.get_solution(archive, solution)
            try:
                tokens = tokenizer.tokenizes(program.decode())
                fragment_idxs = []
                for sample in pad_and_split_tokens(tokens):
                    writer.write(sample)
                    fragment_idxs.append(i)
                    i += 1

                user_submission_idxs[user].append((archive, fragment_idxs))
            except Exception:
                pass

    metadata_file = os.path.join(writer.base_dir, f"user_submission_idxs_{part}.json")
    with open(metadata_file, "w", encoding="UTF-8") as f:
        json.dump(user_submission_idxs, f)


def write_pre_selected_dataset(pairs_or_triplets, writer: DatasetWriter):
    tokenizer = TOKENIZER()
    are_pairs = len(pairs_or_triplets[0]) == 2
    not_skipped = []
    with writer:
        for i in range(0, len(pairs_or_triplets), 2 if are_pairs else 1):
            tokens = []
            two_pairs_or_triplet = (
                pairs_or_triplets[i] + pairs_or_triplets[i + 1]
                if are_pairs
                else pairs_or_triplets[i]
            )
            try:
                for archive, solution in two_pairs_or_triplet:
                    program = writer.get_solution(archive, solution)
                    program_tokens = tokenizer.tokenizes(program.decode())
                    tokens.append(pad_and_split_tokens(program_tokens)[0])
                for program_tokens in tokens:
                    writer.write(program_tokens)

                not_skipped.append(pairs_or_triplets[i])
                if are_pairs:
                    not_skipped.append(pairs_or_triplets[i + 1])
            except Exception:
                pass

    mode = "pairs" if are_pairs else "triplets"
    metadata_file = os.path.join(writer.base_dir, f"{writer.part}_{mode}.json")
    with open(metadata_file, "w", encoding="UTF-8") as f:
        json.dump(not_skipped, f)


def select_samples(
    part: str, base_dir: str, mode: Literal["triplets", "pairs"], n: int
):
    metadata_file = os.path.join(base_dir, f"{part}_{mode}.json")
    if os.path.isfile(metadata_file):
        with open(metadata_file, "r", encoding="UTF-8") as f:
            return json.load(f)

    flat_data = get_partition_data(part)
    data = defaultdict(list)
    for user, archive, solution in flat_data:
        data[user].append((archive, solution))

    selected_samples = []
    visited_triplets = set()

    anchor_candidates = [user for user, solutions in data.items() if len(solutions) > 1]
    all_candidates = list(data.keys())

    while len(selected_samples) < n:
        anchor_user = random.choice(anchor_candidates)
        negative_user = random.choice(all_candidates)
        while negative_user == anchor_user:
            negative_user = random.choice(all_candidates)

        anchor, positive = random.sample(data[anchor_user], 2)
        negative = random.choice(data[negative_user])

        triplet_key = ";".join(sorted(anchor + positive + negative))
        if triplet_key in visited_triplets:
            continue

        visited_triplets.add(triplet_key)
        if mode == "pairs":
            selected_samples.append((anchor, positive))
            selected_samples.append((anchor, negative))
        else:
            selected_samples.append((anchor, positive, negative))

    with open(metadata_file, "w", encoding="UTF-8") as f:
        json.dump(selected_samples, f)

    return selected_samples


class GCJDataset(Dataset):
    def __init__(self, part: str, base_dir: str, size=None, random_fragment=True):
        self.data = zarr.open_array(os.path.join(base_dir, f"{part}.zarr"), mode="r")
        self.size = size
        self.random_fragment = random_fragment

        metadata_file = os.path.join(base_dir, f"user_submission_idxs_{part}.json")
        with open(metadata_file, "r", encoding="UTF-8") as f:
            self.user_submission_idxs = json.load(f)

        self.all_candidates = list(self.user_submission_idxs.keys())
        self.anchor_candidates = [
            user
            for user, solutions in self.user_submission_idxs.items()
            if len(solutions) > 1
        ]

        self.by_archive = defaultdict(list)
        for user, solutions in self.user_submission_idxs.items():
            for archive, solution in solutions:
                self.by_archive[archive].append((user, solution))

    def get_same_archive_different_author(self, archive, author):
        tries = 0
        user, solution = random.choice(self.by_archive[archive])
        while user == author and tries < 10:
            user, solution = random.choice(self.by_archive[archive])
            tries += 1
        return solution if user != author else None

    def get_triplet(
        self,
    ) -> tuple[tuple[str, int, int], tuple[str, int, int], tuple[str, int, int]]:
        anchor_user = random.choice(self.anchor_candidates)
        negative_user = random.choice(self.all_candidates)
        while negative_user == anchor_user:
            negative_user = random.choice(self.all_candidates)

        anchor_solution_idx, positive_solution_idx = random.sample(
            range(len(self.user_submission_idxs[anchor_user])), 2
        )
        _, anchor_solution = self.user_submission_idxs[anchor_user][anchor_solution_idx]
        _, positive_solution = self.user_submission_idxs[anchor_user][
            positive_solution_idx
        ]

        negative_solution_idx = random.choice(
            range(len(self.user_submission_idxs[negative_user]))
        )
        _, negative_solution = self.user_submission_idxs[negative_user][
            negative_solution_idx
        ]

        triplet = (
            (
                random.choice(anchor_solution),
                random.choice(positive_solution),
                random.choice(negative_solution),
            )
            if self.random_fragment
            else (anchor_solution[0], positive_solution[0], negative_solution[0])
        )

        return (
            (anchor_user, anchor_solution_idx, triplet[0]),
            (anchor_user, positive_solution_idx, triplet[1]),
            (negative_user, negative_solution_idx, triplet[2]),
        )

    def get_triplet_tensors(
        self,
    ) -> tuple[
        tuple[str, int, torch.Tensor],
        tuple[str, int, torch.Tensor],
        tuple[str, int, torch.Tensor],
    ]:
        (
            (anchor_user, anchor_idx, anchor),
            (positive_user, positive_idx, positive),
            (negative_user, negative_idx, negative),
        ) = self.get_triplet()
        anchor_tokens = self.data[anchor]
        positive_tokens = self.data[positive]
        negative_tokens = self.data[negative]
        while all(anchor_tokens == positive_tokens):
            (
                (anchor_user, anchor_idx, anchor),
                (positive_user, positive_idx, positive),
                (negative_user, negative_idx, negative),
            ) = self.get_triplet()
            anchor_tokens = self.data[anchor]
            positive_tokens = self.data[positive]
            negative_tokens = self.data[negative]
        return (
            (anchor_user, anchor_idx, torch.tensor(anchor_tokens)),
            (positive_user, positive_idx, torch.tensor(positive_tokens)),
            (negative_user, negative_idx, torch.tensor(negative_tokens)),
        )

    def __len__(self):
        return self.size if self.size != None else self.data.shape[0]

    def __getitem__(self, _):
        return tuple(tensor for _, _, tensor in self.get_triplet_tensors())


class GCJValDataset(Dataset):
    def __init__(self, base_dir: str):
        self.data = zarr.open_array(os.path.join(base_dir, "val.zarr"), mode="r")

    def __len__(self):
        return self.data.shape[0] // 3

    def __getitem__(self, idx):
        return tuple(torch.tensor(self.data[(3 * idx) + i]) for i in range(3))


if __name__ == "__main__":
    if not os.path.exists(FINE_TUNING_DATASET_PATH):
        os.mkdir(FINE_TUNING_DATASET_PATH)

    write_dataset("train", DbDatasetWriter("train", FINE_TUNING_DATASET_PATH))
    write_dataset(
        "val_no_triplets", DbDatasetWriter("val_no_triplets", FINE_TUNING_DATASET_PATH)
    )
    write_pre_selected_dataset(
        select_samples("val", FINE_TUNING_DATASET_PATH, "triplets", 50000),
        DbDatasetWriter("val", FINE_TUNING_DATASET_PATH),
    )
    write_pre_selected_dataset(
        select_samples("test", FINE_TUNING_DATASET_PATH, "pairs", 100000),
        DbDatasetWriter("test", FINE_TUNING_DATASET_PATH),
    )

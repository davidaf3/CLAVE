import os
import numpy as np
import zarr
import random
import torch
from torch.utils.data import Dataset
from config import SEQ_LENGTH, PRETRAINING_DATASET_PATH, TOKENIZER
from utils import pad_and_split_tokens


DATASET_PATH = os.path.join(os.pardir, os.pardir, "Crawler", "python_dataset")


def dataset_generator(files, part):
    tokenizer = TOKENIZER()
    train_size = round(len(files) * 0.9)
    files = files[:train_size] if part == "train" else files[train_size:]
    for filename in files:
        try:
            filepaht = os.path.join(DATASET_PATH, filename)
            if os.stat(filepaht).st_size == 0:
                continue
            for sample in pad_and_split_tokens(tokenizer.tokenize(filepaht)):
                yield sample
        except Exception:
            pass


def write_dataset(files: str, part: str, base_dir: str):
    dataset = zarr.open_array(
        os.path.join(base_dir, f"{part}.zarr"),
        mode="w",
        shape=(0, SEQ_LENGTH),
        chunks=(1024, SEQ_LENGTH),
        dtype=np.int64,
    )

    buffer = []
    for sample in dataset_generator(files, part):
        buffer.append(sample)
        if len(buffer) == 102400:
            dataset.append(buffer)
            buffer = []

    if len(buffer) > 0:
        dataset.append(buffer)


class PretrainingDataset(Dataset):
    def __init__(self, part: str, base_dir: str):
        self.data = zarr.open_array(os.path.join(base_dir, f"{part}.zarr"), mode="r")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx])


if __name__ == "__main__":
    if not os.path.exists(PRETRAINING_DATASET_PATH):
        os.mkdir(PRETRAINING_DATASET_PATH)

    files = os.listdir(DATASET_PATH)
    random.shuffle(files)
    write_dataset(files, "train", PRETRAINING_DATASET_PATH)
    write_dataset(files, "val", PRETRAINING_DATASET_PATH)

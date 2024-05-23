import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from transformers import RobertaTokenizer, RobertaModel
from gcj_data_loader import DbDatasetWriter, get_partition_data, select_samples
from config import FINE_TUNING_DATASET_PATH


class CodeBERT:
    def __init__(self) -> None:
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.model.cuda()
        self.model.eval()

    def __call__(self, batch: list[str]) -> torch.Tensor:
        with torch.inference_mode():
            tokenized = self.tokenizer(batch, padding=True)
            tokens = torch.tensor(
                [prog_tokens[:512] for prog_tokens in tokenized["input_ids"]],
                device="cuda",
            )
            mask = torch.tensor(
                [prog_mask[:512] for prog_mask in tokenized["attention_mask"]],
                device="cuda",
            )
            embeddings = self.model(tokens, attention_mask=mask).last_hidden_state
            embeddings = torch.movedim(embeddings, (1, 2), (2, 1))
            return F.avg_pool1d(embeddings, kernel_size=embeddings.size(2)).squeeze(2)


def write_dataset(part: str):
    codebert = CodeBERT()
    partition_data = get_partition_data(part)
    user_embeddings_idxs = defaultdict(list)
    batch = []
    batch_users = []
    i = 0
    j = 0

    with DbDatasetWriter(
        f"codebert_embeddings_{part}", FINE_TUNING_DATASET_PATH, 768, np.float32
    ) as writer:
        for user, archive, solution in partition_data:
            i += 1
            program = writer.get_solution(archive, solution)
            batch.append(program.decode())
            batch_users.append(user)

            if len(batch) == 128 or i == len(partition_data):
                for user, embedding in zip(batch_users, codebert(batch)):
                    writer.write(embedding.cpu().numpy())
                    user_embeddings_idxs[user].append(j)
                    j += 1

                batch = []
                batch_users = []
                print(f"{i} / {len(partition_data)}", end="\r")

    metadata_file = os.path.join(
        writer.base_dir, f"user_codebert_embeddings_idxs_{part}.json"
    )
    with open(metadata_file, "w", encoding="UTF-8") as f:
        json.dump(user_embeddings_idxs, f)


def write_pre_selected_dataset(pairs_or_triplets, part: str):
    codebert = CodeBERT()
    are_pairs = len(pairs_or_triplets[0]) == 2
    total_count = len(pairs_or_triplets) * (2 if are_pairs else 3)
    batch = []
    i = 0

    with DbDatasetWriter(
        f"codebert_embeddings_{part}", FINE_TUNING_DATASET_PATH, 768, np.float32
    ) as writer:
        for pair_or_triplet in pairs_or_triplets:
            for archive, solution in pair_or_triplet:
                i += 1
                program = writer.get_solution(archive, solution)
                batch.append(program.decode())

                if len(batch) == 128 or i == total_count:
                    for embedding in codebert(batch):
                        writer.write(embedding.cpu().numpy())

                    batch = []
                    print(f"{i} / {total_count}", end="\r")


if __name__ == "__main__":
    write_dataset("train")
    write_pre_selected_dataset(
        select_samples("val", FINE_TUNING_DATASET_PATH, "triplets", 50000), "val"
    )
    write_pre_selected_dataset(
        select_samples("test", FINE_TUNING_DATASET_PATH, "pairs", 100000), "test"
    )

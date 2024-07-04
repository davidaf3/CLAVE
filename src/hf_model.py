import os
import json
import torch
import itertools
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
from bert import tokenization
from tensor2tensor.data_generators import text_encoder
from cubert import python_tokenizer, unified_tokenizer, code_to_subtokenized_sentences
from gcj_data_loader import DbDatasetWriter, get_partition_data, select_samples
from config import FINE_TUNING_DATASET_PATH


class HFModel:
    def __init__(self, hf_model_name: str, tokenizer=None) -> None:
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                hf_model_name, trust_remote_code=True, add_eos_token=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer
        self.model = AutoModel.from_pretrained(hf_model_name, trust_remote_code=True)
        self.model.cuda()
        self.model.eval()

    def __call__(self, batch: list[str]) -> torch.Tensor:
        with torch.inference_mode():
            tokenized = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            embeddings = self.model(
                tokenized["input_ids"].cuda(),
                attention_mask=tokenized["attention_mask"].cuda(),
            ).last_hidden_state
            embeddings = torch.movedim(embeddings, (1, 2), (2, 1))
            return F.avg_pool1d(embeddings, kernel_size=embeddings.size(2)).squeeze(2)


class CuBertTokenizer:
    def __init__(self):
        self.code_tokenizer = python_tokenizer.PythonTokenizer()
        self.code_tokenizer.replace_reserved_keywords(
            ("__HOLE__", "unknown_token_default")
        )
        self.code_tokenizer.update_types_to_skip(
            (
                unified_tokenizer.TokenKind.COMMENT,
                unified_tokenizer.TokenKind.WHITESPACE,
            )
        )
        self.subwork_tokenizer = text_encoder.SubwordTextEncoder(
            os.path.join(
                "cubert_data",
                "20210711_Python_github_python_minus_ethpy150open_deduplicated_vocabulary.txt",
            )
        )

    def __call__(self, batch, **kwargs):
        batch_token_ids = np.full(
            (len(batch), 512), text_encoder.PAD_ID, dtype=np.int64
        )
        for i, text in enumerate(batch):
            token_ids = self.convert_tokens_to_ids(self.tokenize(text))[:512]
            batch_token_ids[i, 0 : len(token_ids)] = token_ids

        tensor_ids = torch.tensor(batch_token_ids, dtype=torch.int64)
        return {
            "input_ids": tensor_ids,
            "attention_mask": tensor_ids.not_equal(text_encoder.PAD_ID).to(torch.int64),
        }

    def tokenize(self, text):
        subtokenized_sentences = (
            code_to_subtokenized_sentences.code_to_cubert_sentences(
                code=text,
                initial_tokenizer=self.code_tokenizer,
                subword_tokenizer=self.subwork_tokenizer,
            )
        )
        return list(itertools.chain(*subtokenized_sentences))

    def convert_tokens_to_ids(self, tokens):
        return tokenization.convert_by_vocab(
            self.subwork_tokenizer._subtoken_string_to_id,  # pylint: disable = protected-access
            tokens,
        )


BATCH_SIZE = 64


def write_dataset(part: str, model_name: str, model: HFModel, emsize: int):
    partition_data = get_partition_data(part)
    user_embeddings_idxs = defaultdict(list)
    batch = []
    batch_users = []
    i = 0
    j = 0

    with DbDatasetWriter(
        f"{model_name}_embeddings_{part}", FINE_TUNING_DATASET_PATH, emsize, np.float32
    ) as writer:
        for user, archive, solution in partition_data:
            i += 1
            program = writer.get_solution(archive, solution)
            batch.append(program.decode())
            batch_users.append(user)

            if len(batch) == BATCH_SIZE or i == len(partition_data):
                for user, embedding in zip(batch_users, model(batch)):
                    writer.write(embedding.cpu().numpy())
                    user_embeddings_idxs[user].append(j)
                    j += 1

                batch = []
                batch_users = []
                print(f"{i} / {len(partition_data)}", end="\r")

    metadata_file = os.path.join(
        writer.base_dir, f"user_{model_name}_embeddings_idxs_{part}.json"
    )
    with open(metadata_file, "w", encoding="UTF-8") as f:
        json.dump(user_embeddings_idxs, f)


def write_pre_selected_dataset(
    pairs_or_triplets, part: str, model_name: str, model: HFModel, emsize: int
):
    are_pairs = len(pairs_or_triplets[0]) == 2
    total_count = len(pairs_or_triplets) * (2 if are_pairs else 3)
    batch = []
    i = 0

    with DbDatasetWriter(
        f"{model_name}_embeddings_{part}", FINE_TUNING_DATASET_PATH, emsize, np.float32
    ) as writer:
        for pair_or_triplet in pairs_or_triplets:
            for archive, solution in pair_or_triplet:
                i += 1
                program = writer.get_solution(archive, solution)
                batch.append(program.decode())

                if len(batch) == BATCH_SIZE or i == total_count:
                    for embedding in model(batch):
                        writer.write(embedding.cpu().numpy())

                    batch = []
                    print(f"{i} / {total_count}", end="\r")


if __name__ == "__main__":
    """write_dataset("train", "codebert", HFModel("microsoft/codebert-base"), 768)
    write_pre_selected_dataset(
        select_samples("val", FINE_TUNING_DATASET_PATH, "triplets", 50000),
        "val",
        "codebert",
        HFModel("microsoft/codebert-base"),
        768,
    )"""
    write_pre_selected_dataset(
        select_samples("test", FINE_TUNING_DATASET_PATH, "pairs", 100000),
        "test",
        "cubert",
        HFModel("claudios/cubert-20210711-Python-512", CuBertTokenizer()),
        1024,
    )

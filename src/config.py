import os


SEQ_LENGTH = 512
MAX_SEQ_LENGTH = 2048
PRETRAINING_DATASET_PATH = os.path.join("data", "pretraining_dataset")
FINE_TUNING_DATASET_PATH = os.path.join("data", "gcj_dataset_512")
AD_HOC_USE_SP = True
PRETRAINING_CHECKPOINT = os.path.join(
    "best_checkpoints",
    "pretraining",
    "sp_512_emb",
    "512",
    "checkpoint-43-val_loss-0.2770-val_acc-0.9380",
)

from tokenizer import SpTokenizer

TOKENIZER = SpTokenizer

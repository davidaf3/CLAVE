import os


SEQ_LENGTH = 512
MAX_SEQ_LENGTH = 2048
PRETRAINING_DATASET_PATH = os.path.join("data", "pretraining_dataset")
FINE_TUNING_DATASET_PATH = os.path.join("data", "gcj_dataset_512")
AD_HOC_USE_SP = True
PRETRAINING_CHECKPOINT = os.path.join(
    "best_checkpoints",
    "pretraining",
    "sp",
    "512",
    "checkpoint-25-val_loss-0.3707-val_acc-0.9200",
)

from tokenizer import SpTokenizer

TOKENIZER = SpTokenizer

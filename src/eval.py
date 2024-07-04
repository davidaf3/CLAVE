import os
import zarr
import torch
import torch.nn.functional as F
import numpy as np
import scipy.stats.distributions as distributions
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
    f1_score,
)
from sklearn.utils import resample
from fdr.train import FDRModel
from fine_tune import FineTunedModel
from config import FINE_TUNING_DATASET_PATH
from scs_gan.train import SCSGan, embedding_similarity
from white_sprague.train import WhiteSpragueModel
from tokenizer import AdHocTokenizer, SpTokenizer


class EvalDataset(Dataset):
    def __init__(self, file):
        self.data = zarr.open_array(
            file, mode="r", cache_metadata=True, cache_attrs=True
        )

    def __len__(self):
        return self.data.shape[0] // 2

    def __getitem__(self, idx):
        label = 1 - (idx % 2)
        return (
            torch.tensor(self.data[2 * idx]),
            torch.tensor(self.data[(2 * idx) + 1]),
            torch.tensor(label),
        )


class CollateFnPairs:
    def __init__(self, dim) -> None:
        self.dim = dim

    def __call__(self, examples):
        xs = [e[0] for e in examples]
        ys = [e[1] for e in examples]
        labels = [e[2] for e in examples]
        return (
            torch.stack(xs, dim=self.dim),
            torch.stack(ys, dim=self.dim),
            torch.stack(labels),
        )


def get_distances(model, data, distance_fn, collate_fn):
    batch_size = 128
    dataloader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=8,
    )

    distances = []
    for batch in dataloader:
        xs, ys, labels = tuple(e.cuda() for e in batch)
        xs_out = model(xs)
        ys_out = model(ys)

        dists = distance_fn(xs_out, ys_out)

        for i in range(dists.size(0)):
            distances.append((dists[i].cpu().numpy(), labels[i].item()))

        print(f"{len(distances)} / {len(data)}", end="\r")

    print("Computed distances")
    return distances


def get_disctances_hf_model(distance_fn, model_name: str):
    embeddings = zarr.open_array(
        os.path.join(FINE_TUNING_DATASET_PATH, f"{model_name}_embeddings_test.zarr"),
        mode="r",
        cache_metadata=True,
        cache_attrs=True,
    )

    distances = []
    for i in range(0, len(embeddings) // 2):
        x, y = embeddings[i * 2 : (i * 2) + 2]
        distance = distance_fn(torch.tensor([x]), torch.tensor([y]))
        distances.append((distance.numpy(), 1 - (i % 2)))
        print(f"{len(distances)} / {len(embeddings) // 2}", end="\r")

    print("Computed distances")
    return distances


def compute_threshold(distances, start, end, step):
    best = 0
    best_f1 = 0
    best_acc = 0
    for t in np.arange(start, end, step):
        y_true, y_pred, _ = eval_threshold(distances, t)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="binary")
        print(f"f1 = {f1:.4f}, acc = {acc:.4f}, t = {t:.4f}")
        if f1 > best_f1 or (f1 == best_f1 and acc > best_acc):
            best = t
            best_f1 = f1
            best_acc = acc

    return best


def eval_threshold(distances, t):
    y_true = []
    y_pred = []
    y_score = []
    for distance, is_positive in distances:
        y_true.append(1 - is_positive)
        y_score.append(distance)
        y_pred.append(1 - int(distance <= t))

    return y_true, y_pred, y_score


def get_metrics(y_true, y_pred, y_score):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_score)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    return acc, auc, prec, rec, f1


def print_metrics(t, acc, auc, prec, rec, f1):
    print(f"acc = {acc:.4f}, auc = {auc:.4f}", end=", ")
    print(f"prec = {prec:.4f}, rec = {rec:.4f}, f1 = {f1:.4f}", end=", ")
    print(f"t = {t:.4f}")


def print_metrics_with_ci(t, acc, auc, prec, rec, f1):
    print(f"acc = {acc[0]:.4f} ({acc[1][0]:.4f}, {acc[1][1]:.4f})", end=", ")
    print(f"auc = {auc[0]:.4f} ({auc[1][0]:.4f}, {auc[1][1]:.4f})", end=", ")
    print(f"prec = {prec[0]:.4f} ({prec[1][0]:.4f}, {prec[1][1]:.4f})", end=", ")
    print(f"rec = {rec[0]:.4f} ({rec[1][0]:.4f}, {rec[1][1]:.4f})", end=", ")
    print(f"f1 = {f1[0]:.4f} ({f1[1][0]:.4f}, {f1[1][1]:.4f})", end=", ")
    print(f"t = {t:.4f}")


def confidence_interval(samples, confidence_level=0.95):
    mean = np.mean(samples)
    stdev = np.std(samples)
    n = len(samples)
    sem = stdev / np.sqrt(n)
    alpha = 1 - confidence_level
    z_value = distributions.norm.ppf(1 - alpha / 2)
    return mean - z_value * sem, mean + z_value * sem


def eval_model(distances, start, end, step1, step2, bootstraping=True):
    to_compute_t = len(distances) // 10
    if to_compute_t % 2 == 1:
        to_compute_t -= 1
    t = compute_threshold(distances[:to_compute_t], start, end, step1)
    t = compute_threshold(distances[:to_compute_t], t - step1, t + step1, step2)
    y_true, y_pred, y_score = eval_threshold(distances[to_compute_t:], t)

    if not bootstraping:
        print_metrics(t, *get_metrics(y_true, y_pred, y_score))
        return

    metrics = tuple([] for _ in range(5))
    for i in range(1000):
        y_true_b, y_pred_b, y_score_b = resample(
            y_true, y_pred, y_score, stratify=y_true
        )
        for j, metric in enumerate(get_metrics(y_true_b, y_pred_b, y_score_b)):
            metrics[j].append(metric)
        if i % 100 == 0:
            print(f"{i} / 1000", end="\r")

    metrics_with_ci = (
        (np.mean(metric), confidence_interval(metric)) for metric in metrics
    )
    print_metrics_with_ci(t, *metrics_with_ci)


if __name__ == "__main__":
    ntokens = SpTokenizer.get_vocab_size()
    emsize = 512
    d_hid = 2048
    d_out = 512
    nlayers = 6
    nhead = 8
    dropout = 0.1
    use_layer_norm = True
    model = FineTunedModel(
        ntokens,
        emsize,
        d_out,
        nhead,
        d_hid,
        nlayers,
        dropout=dropout,
        use_layer_norm=use_layer_norm,
    ).cuda()
    model_checkpoint = torch.load(
        "best_checkpoints/fine_tuning/sp_512_emb/simclr/512/checkpoint-25-val_loss-1.3580-val_acc-0.8659"
    )
    weights = {
        k[10:] if k.startswith("_orig_mod") else k: v
        for k, v in model_checkpoint["model_state_dict"].items()
    }
    model.load_state_dict(weights)
    model.eval()
    model_data = EvalDataset(os.path.join("data", "gcj_dataset_512", "test.zarr"))

    fdr = FDRModel().cuda()
    fdr_checkpoint = torch.load(
        "fdr/checkpoints/checkpoint-75-val_loss-1.3245-val_acc-0.7336"
    )
    fdr.load_state_dict(fdr_checkpoint["model_state_dict"])
    fdr.eval()
    fdr_data = EvalDataset("fdr/test.zarr")

    scs_gan = SCSGan(ntokens, 128, 16).cuda()
    scs_gan_checkpoint = torch.load(
        "scs_gan/checkpoints/sp_no_custom/checkpoint-8-val_loss-0.3820-val_acc-0.9590"
    )
    scs_gan.load_state_dict(scs_gan_checkpoint["model_state_dict"])
    scs_gan.eval()
    rnn_data = EvalDataset(
        os.path.join("data", "gcj_dataset_sp_no_custom", "test.zarr")
    )

    white_sprague = WhiteSpragueModel(ntokens).cuda()
    white_sprague_checkpoint = torch.load(
        "white_sprague/checkpoints/512_no_custom/checkpoint-5-val_loss-3.1827-val_acc-0.6956"
    )
    white_sprague.load_state_dict(white_sprague_checkpoint["model_state_dict"])
    white_sprague.eval()

    with torch.inference_mode():
        eval_model(
            get_distances(
                model,
                model_data,
                lambda x, y: 1 - F.cosine_similarity(x, y),
                CollateFnPairs(0),
            ),
            0,
            0.2,
            0.01,
            0.0005,
        )

        """"
        eval_model(
            get_distances(model, model_data, F.pairwise_distance, CollateFnPairs(0)),
            0,
            15,
            0.25,
            0.05,
        )
        """

        """
        eval_model(
            get_distances(
                fdr,
                fdr_data,
                lambda x, y: 1 - F.cosine_similarity(x, y),
                CollateFnPairs(0),
            ),
            0,
            2,
            0.025,
            0.005,
        )
        """

        """
        eval_model(
            get_distances(
                scs_gan,
                rnn_data,
                lambda x, y: 1 - embedding_similarity(x, y),
                CollateFnPairs(0),
            ),
            0,
            1,
            0.25,
            0.005,
        )
        """

        """
        eval_model(
            get_distances(
                white_sprague,
                rnn_data,
                lambda x, y: 1 - F.cosine_similarity(x, y),
                CollateFnPairs(0),
            ),
            0,
            0.2,
            0.01,
            0.0005,
        )
        """

        """
        eval_model(
            get_disctances_hf_model(
                lambda x, y: 1 - F.cosine_similarity(x, y), "codebert"
            ),
            0,
            0.2,
            0.01,
            0.0005,
        )
        """

        """
        eval_model(
            get_disctances_hf_model(
                lambda x, y: 1 - F.cosine_similarity(x, y), "graphcodebert"
            ),
            0,
            0.2,
            0.01,
            0.0005,
        )
        """

        """
        eval_model(
            get_disctances_hf_model(
                lambda x, y: 1 - F.cosine_similarity(x, y), "codesage"
            ),
            0,
            2,
            0.01,
            0.0005,
        )
        """

        """
        eval_model(
            get_disctances_hf_model(
                lambda x, y: 1 - F.cosine_similarity(x, y), "starencoder"
            ),
            0,
            2,
            0.01,
            0.0005,
        )
        """

        """
        eval_model(
            get_disctances_hf_model(
                lambda x, y: 1 - F.cosine_similarity(x, y), "cubert"
            ),
            0,
            2,
            0.01,
            0.0005,
        )
        """

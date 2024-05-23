import os
import sys
import sqlite3
import json
import random
import numpy as np
import zarr
from collections import defaultdict
from make_vocab import make_vocab
from nested_bigrams import get_feature_vector, get_bigrams

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from gcj_data_loader import DbDatasetWriter


def main():
    train_data, _, _ = select_data()
    print("Making train and val...")
    vocab = make_train(train_data)
    print("Making test...")
    make_test(vocab)


def query_program(archive, solution, cons):
    con = cons[archive]
    cur = con.cursor()
    res = cur.execute(f"SELECT data FROM sqlar WHERE name = '{solution}' LIMIT 1")
    return res.fetchone()[0]


def select_data():
    gcj_path = os.path.join(os.pardir, os.pardir, "gcj")
    with open(os.path.join(gcj_path, "metadata.json"), "r", encoding="UTF-8") as f:
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

    train_data = data_flat[:train_size]
    val_data = data_flat[train_size : train_size + val_size]
    test_data = data_flat[train_size + val_size :]
    return unflatten(train_data), unflatten(val_data), unflatten(test_data)


def unflatten(data):
    by_user = defaultdict(list)
    for user, archive, solution in data:
        by_user[user].append((archive, solution))
    return by_user


def make_train(data):
    solutions_path = os.path.join(os.pardir, os.pardir, "gcj", "solutions")
    cons = {
        archive: sqlite3.connect(os.path.join(solutions_path, archive))
        for archive in os.listdir(solutions_path)
    }

    train_ds = zarr.open_array(
        f"train.zarr", mode="w", shape=(0, 8000), chunks=(64, 8000), dtype=np.int64
    )
    train_buffer = []
    val_ds = zarr.open_array(
        f"val.zarr", mode="w", shape=(0, 8000), chunks=(64, 8000), dtype=np.int64
    )
    val_buffer = []

    selected_users_train = {}
    selected_users_val = {}
    for user, solutions in data.items():
        if len(solutions) >= 10:
            invalid_user = False
            for archive, solution in solutions:
                program = query_program(archive, solution, cons)
                try:
                    get_bigrams(program)
                except (SyntaxError, ValueError):
                    invalid_user = True
                    break

            if not invalid_user:
                selected_users_train[user] = solutions[:8]
                selected_users_val[user] = solutions[8:10]
                if len(selected_users_train) >= 160:
                    break

    print(f"Selected {len(selected_users_train)} users")

    vocab = make_vocab(selected_users_train)

    for user, solutions in selected_users_train.items():
        for archive, solution in solutions:
            program = query_program(archive, solution, cons)
            train_buffer.append(get_feature_vector(program, vocab))

            if len(train_buffer) == 102400:
                train_ds.append(train_buffer)
                train_buffer = []

    if len(train_buffer) > 0:
        train_ds.append(train_buffer)

    for user, solutions in selected_users_val.items():
        for archive, solution in solutions:
            program = query_program(archive, solution, cons)
            val_buffer.append(get_feature_vector(program, vocab))

            if len(val_buffer) == 102400:
                val_ds.append(val_buffer)
                val_buffer = []

    if len(val_buffer) > 0:
        val_ds.append(val_buffer)

    for con in cons.values():
        con.close()

    return vocab


def make_test(vocab):
    metadata_file = os.path.join(
        os.pardir, "data", "gcj_dataset_ad_hoc", f"test_pairs.json"
    )
    with open(metadata_file, "r", encoding="UTF-8") as f:
        pairs = json.load(f)

    not_skipped = []
    with DbDatasetWriter("test", ".", 8000) as writer:
        for i in range(0, len(pairs), 2):
            two_pairs = pairs[i] + pairs[i + 1]
            feature_vectors = []
            try:
                for archive, solution in two_pairs:
                    program = writer.get_solution(archive, solution)
                    feature_vectors.append(get_feature_vector(program.decode(), vocab))

                not_skipped.append(pairs[i])
                not_skipped.append(pairs[i + 1])
                for feature_vector in feature_vectors:
                    writer.write(feature_vector)
            except Exception:
                pass

    with open(metadata_file, "w", encoding="UTF-8") as f:
        json.dump(not_skipped, f)


if __name__ == "__main__":
    main()

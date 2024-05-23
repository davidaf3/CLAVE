import sqlite3
import os
import json
from nested_bigrams import get_bigrams
from collections import defaultdict


def make_vocab(train_data):
    solutions_path = os.path.join(os.pardir, os.pardir, "gcj", "solutions")
    cons = {
        archive: sqlite3.connect(os.path.join(solutions_path, archive))
        for archive in os.listdir(solutions_path)
    }

    bigrams_users_freq = defaultdict(lambda: (0, 0))
    for _, solutions in train_data.items():
        user_freq = defaultdict(int)

        for archive, solution in solutions:
            con = cons[archive]
            cur = con.cursor()
            res = cur.execute(
                f"SELECT data FROM sqlar WHERE name = '{solution}' LIMIT 1"
            )
            program = res.fetchone()[0]
            for bigram, freq in get_bigrams(program).items():
                user_freq[bigram] += freq

        for bigram, freq in user_freq.items():
            users, total_freq = bigrams_users_freq[bigram]
            bigrams_users_freq[bigram] = (users + 1, total_freq + freq)

    for con in cons.values():
        con.close()

    common_bigrams = [
        (bigram, freq)
        for bigram, (users, freq) in bigrams_users_freq.items()
        if users > 1
    ]
    top_bigrams = sorted(common_bigrams, key=lambda e: e[1], reverse=True)[:8000]
    top_bigrams = list(map(lambda e: str(e[0]), top_bigrams))
    while len(top_bigrams) < 8000:
        top_bigrams.append("_")

    with open("vocab.json", "w", encoding="UTF-8") as f:
        json.dump(top_bigrams, f)

    return top_bigrams

# -*- coding: utf-8 -*-
# @Time        : 2020/5/27 15:20
# @Author      : ssxy00
# @File        : evaluate_diversity.py
# @Description : this file is modified from https://github.com/rekriz11/DeDiv/blob/dfafd46f57b3b4b6184bee30a7190e75f34ae81a/analyze_diversity.py

import argparse
import jsonlines
import collections
import numpy as np
from tqdm import tqdm

def eval_distinct_k(candidates, k):
    """The total number of k-grams divided by the total number of tokens
         over all the candidates.
      """
    kgrams = set()
    total = 0
    for cand in candidates:
        if len(cand) < k:
            continue
        for i in range(0, len(cand) - k + 1):
            kgrams.add(tuple(cand[i:i + k]))
        total += len(cand)
    if total == 0:
        return 0
    else:
        return len(kgrams) / total


def eval_entropy_k(candidates, k):
    """Entropy method which takes into account word frequency."""
    kgram_counter = collections.Counter()
    for cand in candidates:
        for i in range(0, len(cand) - k + 1):
            kgram_counter.update([tuple(cand[i:i + k])])

    counts = kgram_counter.values()
    s = sum(counts)
    if s == 0:
        # all of the candidates are shorter than k
        return 0
    return (-1.0 / s) * sum(f * np.log(f / s) for f in counts)


def main(args):
    print(args.eval_file)
    average_distinct_1 = 0.
    average_distinct_2 = 0.
    average_entropy_4 = 0.
    with jsonlines.open(args.eval_file) as reader:
        for idx, row in enumerate(tqdm(reader)):
            average_distinct_1 = (average_distinct_1 * idx + eval_distinct_k(row["predict_responses"], 1)) / (idx + 1)
            average_distinct_2 = (average_distinct_2 * idx + eval_distinct_k(row["predict_responses"], 2)) / (idx + 1)
            average_entropy_4 = (average_entropy_4 * idx + eval_entropy_k(row["predict_responses"], 4)) / (idx + 1)
    print(f"distinct 1: {average_distinct_1}")
    print(f"distinct 2: {average_distinct_2}")
    print(f"entropy 4: {average_entropy_4}")


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", default="./result.jsonl")
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()

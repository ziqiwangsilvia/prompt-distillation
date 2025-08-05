import argparse
import asyncio
from collections import Counter
import csv
from functools import partial
import glob
import httpx
import logging
import os
import random
import re
import sys
import time  # noqa
from typing import Any, Dict, List

from question_metrics import read_file_match


def main(
    input_path: str = "",
    dataset_family: str = "",
) -> None:
    files = glob.glob(input_path, recursive=True)
    correctness_dict: Dict[str, float] = {}
    for csv_path in files:
        correctness = read_file_match(csv_path, hotpot="hotpot" in dataset_family)
        correctness_dict[csv_path] = correctness

    # Sort the dictionary by correctness percentage in descending order
    sorted_correctness = sorted(correctness_dict.items(), key=lambda item: item[1], reverse=True)
    print("Sorted Evaluation Correctness:")
    for index, (grading_path, correctness) in enumerate(sorted_correctness, start=1):
        print(f"{index}. {grading_path}: {correctness:.2f}%", flush=True)


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)


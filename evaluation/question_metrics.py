import ast
import collections
import csv
import html
import re
import string
from pathlib import Path
from typing import List

ARTICLES_REGEX = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def _normalize(text: str) -> str:
    """Lower‑case, strip articles/punctuation, squeeze whitespace."""
    def remove_articles(t: str) -> str:
        return ARTICLES_REGEX.sub(" ", t)

    def remove_punc(t: str) -> str:
        return "".join(ch for ch in t if ch not in string.punctuation)

    def white_space_fix(t: str) -> str:
        return " ".join(t.split())

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def _answer_match(prediction: str, gold_answers: List[str]) -> bool:
    """True if *any* gold answer is contained in the normalised prediction."""
    pred_norm = _normalize(prediction)
    return any(_normalize(gold) in pred_norm for gold in gold_answers)


def read_file_match(csv_path: str | Path, *, hotpot: bool = False) -> float:
    """
    Read an output CSV produced by the evaluation pipeline and return the
    **percentage of questions whose prediction contains (match‑by‑substring)
    at least one gold answer**.

    Parameters
    ----------
    csv_path : str | Path
        Path to the `output_*.csv`.
    hotpot : bool, default False
        HotpotQA files store a *single* gold answer in column 2 instead of a
        list‐of‐strings repr.

    Returns
    -------
    float
        Accuracy in **percentage** (`0 – 100`).
    """
    csv_path = Path(csv_path)
    print(f"Scoring answer file: {csv_path}")

    matches, total = 0, 0
    with csv_path.open(newline="", encoding="utf-8", errors="ignore") as fh:
        reader = csv.reader(fh, delimiter=";")

        for row in reader:
            try:
                prediction = row[2]                       # generated answer
                if hotpot:
                    golds = [row[1]]
                else:
                    # Column 2 is a list-repr in SQuADShifts
                    golds = set(ast.literal_eval(row[1]))
            except Exception:
                try:
                    # Get rid of escaping
                    l = row[1].replace('\\"', '"').replace("\\'", "'")
                    golds = set(ast.literal_eval(l))
                except:
                    # Last try to clean
                    html_unescaped = html.unescape(row[1])
                    cleaned = html_unescaped.replace('\\', '')
                    try:
                        golds = set(ast.literal_eval(cleaned))
                    except:
                        # Malformed ground-truth answer
                        continue

            if _answer_match(prediction, golds):
                matches += 1
            total += 1

    if total == 0:
        raise ValueError(f"No valid rows found in {csv_path}")

    return (matches / total) * 100.0

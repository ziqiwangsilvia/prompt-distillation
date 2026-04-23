import json
from typing import List

from datasets import Dataset


_SQUADSHIFTS_URLS = {
    "new_wiki": "https://raw.githubusercontent.com/modestyachts/squadshifts-website/master/datasets/new_wiki_v1.0.json",
    "nyt": "https://raw.githubusercontent.com/modestyachts/squadshifts-website/master/datasets/nyt_v1.0.json",
    "reddit": "https://raw.githubusercontent.com/modestyachts/squadshifts-website/master/datasets/reddit_v1.0.json",
    "amazon": "https://raw.githubusercontent.com/modestyachts/squadshifts-website/master/datasets/amazon_reviews_v1.0.json",
}


def load_squadshifts(subset: str):
    """Load a SquadShifts subset directly from source JSON, bypassing the deprecated HF loading script."""
    import urllib.request

    url = _SQUADSHIFTS_URLS[subset]
    with urllib.request.urlopen(url) as resp:
        squad = json.loads(resp.read().decode("utf-8"))

    rows = []
    for article in squad["data"]:
        title = article.get("title", "").strip()
        for paragraph in article["paragraphs"]:
            context = paragraph["context"].strip()
            for qa in paragraph["qas"]:
                rows.append({
                    "id": qa["id"],
                    "title": title,
                    "context": context,
                    "question": qa["question"].strip(),
                    "answers": {
                        "answer_start": [a["answer_start"] for a in qa["answers"]],
                        "text": [a["text"].strip() for a in qa["answers"]],
                    },
                })
    return Dataset.from_list(rows)

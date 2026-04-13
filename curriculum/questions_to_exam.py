import csv
import json
import os
from typing import Optional

from datasets import load_dataset

from data.naming import generate_exam_filename, generate_exam_name, generate_question_path
from data.loading import load_squadshifts
from evaluation.utils import get_prompt_context


def create_lessons(
    dataset_family: str,
    dataset: str,
    max_items: int,
    variant: str = "default",
) -> dict:
    """Create a lessons dict from a HuggingFace dataset."""
    lessons = []

    if dataset_family == "squadshifts":
        hf_dataset = load_squadshifts(dataset)
    elif dataset_family == "hotpotqa":
        hf_dataset = load_dataset("hotpotqa/hotpot_qa", dataset, trust_remote_code=True)["validation"]
    else:
        raise ValueError(f"Unknown dataset_family: {dataset_family}")

    for i, item in enumerate(hf_dataset):
        if i >= max_items > 0:
            break

        exercise = item['question']
        context = get_prompt_context(item, dataset_family)

        if variant == "cot":
            material = f"{context}\n\nPlease answer the following question. Reason step by step.\n"
        elif variant == "default":
            material = context
        else:
            raise ValueError(f"Unknown format: {variant}")

        lessons.append({
            "id": generate_exam_name(dataset_family, dataset, variant, max_items, i),
            "material": material,
            "exercises": [{"exercise": exercise}],
        })

    return {"lessons": lessons}


def create_lessons_from_csv(
    csv_path: str,
    dataset_family: str,
    dataset: str,
    max_items: int,
    variant: str = "default",
) -> dict:
    """Create a lessons dict from a generated eval CSV."""
    lessons = []

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';', quoting=csv.QUOTE_NONE)
        for i, row in enumerate(reader):
            if max_items and i >= max_items:
                break
            exercise = row[0]
            context = ';'.join(row[1:])
            if not exercise:
                continue

            lessons.append({
                "id": generate_exam_name(dataset_family, dataset, variant, max_items, i),
                "material": context,
                "exercises": [{"exercise": exercise}],
            })

    return {"lessons": lessons}


def main(
    dataset_family: str = "financial",
    dataset: str = "tool_calling",
    max_items: int = 20,
    variant: str = "default",
    eval_csv_path: str = "",
    base: str = "llama3-8b-instruct",
    train_questions: int = 200,
    temperature: float = 1.5,
    max_train_items: int = 1,
) -> None:
    """Create a JSON file for exam questions."""
    output_dir = generate_exam_filename(dataset_family, dataset, variant, max_items)
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    if os.path.exists(output_dir):
        print(f"{output_dir} already exists — skipping.")
        return

    if dataset_family in ("squadshifts", "hotpotqa"):
        data = create_lessons(dataset_family, dataset, max_items, variant)
    else:
        if not eval_csv_path:
            train_path = generate_question_path(
                dataset_family, dataset, base, train_questions, temperature, max_train_items,
            )
            eval_csv_path = str(os.path.join(os.path.dirname(train_path), f"eval_{os.path.basename(train_path)}"))
        if not os.path.exists(eval_csv_path):
            raise FileNotFoundError(f"Eval CSV not found: {eval_csv_path}. Run sample_tool_questions.py first.")
        print(f"Reading eval questions from {eval_csv_path}")
        data = create_lessons_from_csv(eval_csv_path, dataset_family, dataset, max_items, variant)

    print(f"Processing {dataset_family=}, {dataset=}, {variant=}, {max_items=}")
    with open(output_dir, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"JSON written to {output_dir}")


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)

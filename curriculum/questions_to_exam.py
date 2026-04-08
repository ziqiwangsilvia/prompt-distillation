import csv
import os
from xml.etree.ElementTree import Element, SubElement, tostring
from typing import Literal, Optional

from datasets import load_dataset

from data.naming import generate_exam_filename, generate_exam_name, generate_question_path
from data.loading import load_squadshifts
from curriculum.csv_to_lesson import prettify
from evaluation.utils import get_prompt_context


def create_xml(
    dataset_family: str,
    dataset: str,
    max_items: int,
    variant: str = "default",
) -> str:
    """Create an XML exam file from a HuggingFace dataset, supporting squadshifts/hotpotqa and default/cot formats."""
    lessons = Element('lessons')

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

        lesson = SubElement(
            lessons, 'lesson',
            id=generate_exam_name(dataset_family, dataset, variant, max_items, i)
        )
        material = SubElement(lesson, 'material')
        if variant == "cot":
            material.text = f"{context}\n\nPlease answer the following question. Reason step by step.\n"
        elif variant == "default":
            material.text = context
        else:
            raise ValueError(f"Unknown format: {variant}")

        ex_element = SubElement(lesson, 'exercise')
        ex_element.text = exercise

    try:
        return prettify(lessons)
    except Exception:
        print(f"Failure to prettify output for {dataset_family=}, {dataset=}")
        return tostring(lessons, 'unicode')


def create_xml_from_csv(
    csv_path: str,
    dataset_family: str,
    dataset: str,
    max_items: int,
    variant: str = "default",
) -> str:
    """Create an XML exam file from a generated eval CSV."""
    lessons = Element('lessons')

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';', quoting=csv.QUOTE_NONE)
        for i, row in enumerate(reader):
            if max_items and i >= max_items:
                break
            exercise = row[0]
            context = ';'.join(row[1:])
            if not exercise:
                continue

            lesson = SubElement(
                lessons, 'lesson',
                id=generate_exam_name(dataset_family, dataset, variant, max_items, i)
            )
            material = SubElement(lesson, 'material')
            material.text = context
            ex_element = SubElement(lesson, 'exercise')
            ex_element.text = exercise

    try:
        return prettify(lessons)
    except Exception:
        print(f"Failure to prettify output for {csv_path}")
        return tostring(lessons, 'unicode')


def main(
    dataset_family: str = "financial",
    dataset: str = "tool_calling",
    max_items: int = 20,
    variant: str = "default",
    eval_csv_path: str = "",
    # These are only needed to locate the eval CSV when eval_csv_path is not set
    base: str = "llama3-8b-instruct",
    train_questions: int = 200,
    temperature: float = 1.5,
    max_train_items: int = 1,
) -> None:
    """Create an XML file for exam questions."""
    output_dir = generate_exam_filename(dataset_family, dataset, variant, max_items)
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    if os.path.exists(output_dir):
        print(f"{output_dir} already exists — skipping.")
        return

    if dataset_family in ("squadshifts", "hotpotqa"):
        xml_output = create_xml(dataset_family, dataset, max_items, variant)
    else:
        # Find eval CSV
        if not eval_csv_path:
            train_path = generate_question_path(
                dataset_family, dataset, base, train_questions, temperature, max_train_items,
            )
            eval_csv_path = str(os.path.join(os.path.dirname(train_path), f"eval_{os.path.basename(train_path)}"))
        if not os.path.exists(eval_csv_path):
            raise FileNotFoundError(f"Eval CSV not found: {eval_csv_path}. Run sample_tool_questions.py first.")
        print(f"Reading eval questions from {eval_csv_path}")
        xml_output = create_xml_from_csv(eval_csv_path, dataset_family, dataset, max_items, variant)

    print(f"Processing {dataset_family=}, {dataset=}, {variant=}, {max_items=}")
    with open(output_dir, 'w', encoding='utf-8') as f:
        f.write(xml_output)
    print(f"XML written to {output_dir}")


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)

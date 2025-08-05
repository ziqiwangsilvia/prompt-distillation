from xml.etree.ElementTree import Element, SubElement, tostring
from typing import Literal, Optional

from datasets import load_dataset

from core.file_naming import generate_exam_filename, generate_exam_name
from curriculum.csv_to_lesson import prettify
from evaluation.utils import get_prompt_context

def create_xml(
    dataset_family: Literal["squadshifts", "hotpotqa"],
    dataset: str,
    max_items: int,
    variant: Literal["default", "cot"] = "default",
) -> str:
    """Create an XML exam file from a HuggingFace dataset, supporting squadshifts/hotpotqa and default/cot formats."""
    lessons = Element('lessons')

    # Dataset loading logic
    if dataset_family == "squadshifts":
        hf_dataset = load_dataset(dataset_family, dataset, trust_remote_code=True)["test"]
    elif dataset_family == "hotpotqa":
        hf_dataset = load_dataset("hotpotqa/hotpot_qa", dataset, trust_remote_code=True)["validation"]
    else:
        raise ValueError(f"Unknown dataset_family: {dataset_family}")

    # Build lessons
    for i, item in enumerate(hf_dataset):
        if i >= max_items > 0:
            break

        # Support both squadshifts and hotpotqa fields
        exercise = item['question']
        context = get_prompt_context(item, dataset_family)

        # Format-specific lesson id and material
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

    # Return formatted XML
    try:
        return prettify(lessons)
    except Exception:
        print(f"Failure to prettify output for {dataset_family=}, {dataset=}")
        return tostring(lessons, 'unicode')


def main(
    dataset_family: Literal["squadshifts", "hotpotqa"] = "squadshifts",
    dataset: str = "nyt",
    max_items: int = 1000,
    variant: Literal["default", "cot"] = "default"
) -> None:
    """Create an XML file for exam questions from a dataset, supporting family and format."""

    # Build a unified output_dir string using variable interpolation
    output_dir = generate_exam_filename(dataset_family, dataset, variant, max_items) 
    print(f"Processing {dataset_family=}, {dataset=}, {variant=}, {max_items=}")
    xml_output = create_xml(dataset_family, dataset, max_items, variant)
    with open(output_dir, 'w', encoding='utf-8') as f:
        f.write(xml_output)
    print(f"XML written to {output_dir}")

if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)

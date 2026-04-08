import argparse
import csv
import itertools
import os
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom

from data.naming import generate_question_path, generate_lesson_name, generate_lesson_filename

def prettify(elem: Element) -> str:
    """Return a pretty-printed XML string for the Element."""
    rough_string = tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="")

def remove_nul_characters(filename: str) -> str:
    """Remove NUL characters from file, write to a temp file, and return its path."""
    with open(filename, 'rb') as file:
        data = file.read().decode('utf-8').replace('\r', '')
    clean_data = data.replace('\0', '')
    temp_filename = filename + '.tmp'
    with open(temp_filename, 'w', encoding='utf-8') as file:
        file.write(clean_data)
    return temp_filename


def create_xml(
    dataset_family: str,
    dataset: str,
    input_path: str,
    model: str,
    questions: int,
    temperature: float,
    max_items: int,
    variant: str
) -> str:
    """Read CSV, create XML lessons, and return XML as string."""
    lessons = Element('lessons')
    cleaned_input_path = remove_nul_characters(input_path)
    print(cleaned_input_path)
    with open(cleaned_input_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quoting=csv.QUOTE_NONE)
        for i, row in enumerate(reader):
            exercise = row[0]
            context = ';'.join(row[1:])
            if not exercise:
                continue
            lesson = SubElement(
                lessons, 'lesson',
                id=generate_lesson_name(dataset_family, dataset, variant, model, questions, temperature, max_items, i),
            )
            material = SubElement(lesson, 'material')
            if variant == "cot":
                material.text = f"{context}\n\nPlease answer the following question. Reason step by step.\n"
            elif variant == "default":
                material.text = f"{context}"
            ex_element = SubElement(lesson, 'exercise')
            ex_element.text = exercise
    try:
        return prettify(lessons)
    except Exception:
        print(f"Failure to prettify output for {input_path}")
        return tostring(lessons, 'unicode')


def main(
    dataset: str = "nyt",
    dataset_family: str = "squadshifts",
    temperature: float = 1.5,
    model: str = "llama3-8b-instruct",
    train_questions: int = 30,
    max_items: int = 1000,
    variant: str = "default"
) -> None:
    """Convert CSV to pretty-printed XML lessons for curriculum generation."""
    if variant not in ["default", "cot"]:
        raise ValueError(f"Unknown variant: {variant}")

    input_path = generate_question_path(dataset_family, dataset, model, train_questions, temperature, max_items)
    output_path = generate_lesson_filename(dataset_family, dataset, variant, model, train_questions, temperature, max_items)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        print(f"{output_path} already exists — skipping.")
        return

    print(f"Processing {input_path}")
    xml_output = create_xml(dataset_family, dataset, input_path, model, train_questions, temperature, max_items, variant)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(xml_output)
    print(f"XML written to {output_path}")

if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)

from pathlib import Path
from typing import Dict, Optional

def generate_lesson_name(
    dataset_family: str,
    dataset: str,
    variant: str,
    model: str,
    questions: int,
    temperature: float,
    max_items: int,
    idx: Optional[int] = None,
) -> str:
    name = f"{dataset_family}_{dataset}_{variant}_{model}_{questions}_{temperature}_{max_items}_train"
    if idx is not None:
        name += f"_{idx}"
    return name

def generate_exam_name(
    dataset_family: str,
    dataset: str,
    variant: str,
    max_items: int,
    idx: Optional[int] = None, 
) -> str:
    name = f"{dataset_family}_{dataset}_{variant}_{max_items}_test"
    if idx is not None:
        name += f"_{idx}"
    return name

def generate_lesson_filename(*args, **kwargs) -> str:
    lesson_name = generate_lesson_name(*args, **kwargs)
    return f"output/lessons/lesson_{lesson_name}.xml"

def generate_exam_filename(*args, **kwargs) -> str:
    exam_name = generate_exam_name(*args, **kwargs)
    return f"output/exams/exam_{exam_name}.xml"

def generate_question_path(
    dataset_family: str,
    dataset: str,
    model: str,
    questions: int,
    temperature: float,
    max_items: int
) -> str:
    if max_items:
        return f"output/questions/{dataset_family}_{dataset}/{model}/questions_{questions}_{temperature}_{max_items}.csv"
    else:
        return f"output/questions/{dataset_family}_{dataset}/{model}/questions_{questions}_{temperature}.csv"

def generate_augmented_filename(
    lesson_filename: str,
    n_choices: int = 1,
    temperature: float = 1,
    model_flags: Optional[Dict[str, bool]] = None,
    partition_idx: Optional[int] = None,
    partition_type: Optional[str] = None,
    suffix: str = "xml",
) -> Path:
    """
    Create an augmented filename for training/validation/output XML files.
    - lesson_filename: base name or lesson ID (without extension)
    - model_flags: dictionary of {flag_name: bool}
    - n_choices, temperature: flags/suffixes
    - partition_idx/partition_type: additional info for chunked datasets
    """
    fname = lesson_filename
    if n_choices != 1:
        fname += f"_x{n_choices}"
    if temperature != 1:
        fname += f"_t{temperature}"
    if model_flags:
        for flag_name, is_active in model_flags.items():
            if is_active:
                fname += f"_{flag_name.replace('_', '-')}"
    if partition_idx is not None:
        fname += f"_chunk{partition_idx}"
    if partition_type:
        fname += f"_{partition_type}"
    fname += f".{suffix}"
    return Path(fname)

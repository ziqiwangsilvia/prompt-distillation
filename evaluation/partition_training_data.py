import math
from pathlib import Path
import xml.etree.ElementTree as ET
from functools import partial
from typing import List, Literal

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator

from core.llm import LLM
from core.messages import Message, Role
from core.model_configs import MODEL_CONFIGS, create_model_flags
from core.file_naming import (
    generate_augmented_filename,
    generate_lesson_name,
)
from curriculum.exercise_with_answers import xml_dump
from training.student_teacher_dataset import (
    StudentTeacherDataset,
    IGNORE_INDEX,
    read_exercises,
)


def chunk_list(seq: list, n_parts: int):
    """
    Yield *n_parts* contiguous slices of *seq* (last one may be shorter).
    """
    if n_parts <= 0:
        raise ValueError("n_parts must be > 0")
    sz = math.ceil(len(seq) / n_parts)
    for i in range(0, len(seq), sz):
        yield seq[i : i + sz]


def main(
    # student model that will *score* the data
    base: str = "llama3-8b-instruct",
    # model that produced the XML lesson files
    lesson_model: str = "llama3-8b-instruct",
    # dataset spec like "nyt_default"  or "amazon_cot"
    dataset: str = "nyt",
    variant: str = "default",
    dataset_family: str = "squadshifts",
    # partitioning
    partitions: int = 5,
    partition_type: Literal["kld", "entropy"] = "kld",
    # flags to reproduce the lesson filename
    lesson_temp: float = 1.5,
    lesson_num_choices: int = 1,
    # question‑generator
    question_model: str = "llama3-8b-instruct",
    train_questions: int = 30,
    question_temperature: float = 1.5,
    max_items_train: int = 1000,
    # misc
    datapath: Path = Path("data"),
    batch_size: int = 4,
    overwrite_file: bool = False,
) -> None:


    if partitions is None or partitions <= 0:
        raise ValueError("`partitions` must be a positive integer")
    if partition_type not in {"kld", "entropy"}:
        raise ValueError("`partition_type` must be either 'kld' or 'entropy'")

    base_lesson_id = generate_lesson_name(
        dataset_family=dataset_family,
        dataset=dataset,
        variant=variant,
        model=question_model,
        questions=train_questions,
        temperature=question_temperature,
        max_items=max_items_train
    )

    lesson_model_flags = create_model_flags(lesson_model)
    xml_filename = generate_augmented_filename(
        lesson_filename=base_lesson_id,
        n_choices=lesson_num_choices,
        temperature=lesson_temp,
        model_flags=lesson_model_flags,
        suffix="xml",
    )

    xml_path = datapath / xml_filename
    if not xml_path.exists():
        raise FileNotFoundError(f"Lesson XML not found: {xml_path}")

    system_prompt = MODEL_CONFIGS[base].system_message
    base_llm = LLM(base, opening_message=Message(Role.SYSTEM, system_prompt))

    accelerator = Accelerator(mixed_precision="bf16")
    model = accelerator.prepare(base_llm.load_model(training=False).to(torch.bfloat16)).eval()

    dataset = StudentTeacherDataset(
        base_llm, [xml_filename.name], datapath=datapath, verbose=False
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(dataset.collate_fn, padding_value=0, llm=base_llm),
    )
    loader = accelerator.prepare(loader)

    klds = []
    entropies = []
    for batch in loader:
        with torch.no_grad():
            student_inputs = batch['student_seqs'][..., :-1]  # (batch_size, seq_length)
            student_labels = batch['student_labels'][..., 1:]  # (batch_size, seq_length)
            student_masks = student_labels != IGNORE_INDEX
            batch_size, seq_length = student_inputs.shape

            teacher_inputs = batch['teacher_seqs'][..., :-1]
            teacher_masks = batch['teacher_masks'][..., 1:]
            teacher_output = model.forward(teacher_inputs)
            t_logits = teacher_output.logits[teacher_masks].detach()
            t_log_probs = F.log_softmax(t_logits, dim=-1).detach()
            t_probs = torch.exp(t_log_probs)
            t_entropy = -(t_probs * t_log_probs).sum(-1)  # Average entropy
            t_entropy_mx = torch.zeros(batch_size, teacher_inputs.shape[-1], device=teacher_inputs.device)
            t_entropy_mx[teacher_masks] = t_entropy
            t_entropy = t_entropy_mx.sum(-1) / teacher_masks.sum(-1) # Mask question entropy

            student_output = model.forward(student_inputs)
            student_logits = student_output.logits
            s_logits = student_logits[student_masks]
            s_log_probs = F.log_softmax(s_logits, dim=-1)
            logit_loss = F.kl_div(
                s_log_probs, t_log_probs, log_target=True,
                reduction="none",
            )
            logit_loss_t = logit_loss.sum(-1)  # (n_tokens,)

            logit_loss_mx = torch.zeros(batch_size, seq_length, device=student_inputs.device, dtype=logit_loss_t.dtype)
            logit_loss_mx[student_masks] = logit_loss_t
            logit_loss = logit_loss_mx.sum(-1) / student_masks.sum(-1)

            klds += logit_loss.tolist()
            entropies += t_entropy.tolist()

    print("Average entropy", sum(entropies) / len(entropies))
    print("Average KL-divergence", sum(klds) / len(klds))

    # ---- Read the original exercises ----
    print(f"Reading lessons from {xml_path}")
    exercises = read_exercises(xml_path)
    print(f"Number of exercises read: {len(exercises)}")
    
    assert len(klds) == len(entropies) == len(exercises)

    if partition_type == "kld":
        # Pair up (kld, exercise), sort ascending
        paired_data = sorted(zip(klds, exercises), key=lambda x: x[0])
    elif partition_type == "entropy":
        # Pair up (entropy, exercise), sort ascending
        paired_data = sorted(zip(entropies, exercises), key=lambda x: x[0])

    # After sorting, keep only the exercises in the new order
    sorted_exercises = [p[1] for p in paired_data]

    # Chunk them into N (partitions) contiguous slices
    exercise_chunks = list(chunk_list(sorted_exercises, partitions))
    print(f"Split into {partitions} chunks. Starting to write.")

    # ---- Write out each chunk as an XML file ----
    for chunk_idx, chunk in enumerate(exercise_chunks, start=1):
        root = ET.Element("exercises_with_answers")
        ET.SubElement(root, "temperature", value=str(lesson_temp))

        for ex in chunk:
            ex.to_xml(root)

        # Prepare the new filename for each chunk
        output_filename = str(xml_filename).replace(".xml", f"_chunk{chunk_idx}_{partition_type}.xml")
        output_path = datapath / output_filename

        if not overwrite_file and output_path.exists():
            raise FileExistsError(
                f"The file '{output_path}' already exists. Exiting to avoid overwriting."
            )

        with open(output_path, "w") as output_file:
            xml_dump(root, output_file)

        print(f"Saved chunk {chunk_idx} to {output_path}")


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)

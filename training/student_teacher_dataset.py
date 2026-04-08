import numpy as np
import os
from pathlib import Path
import re
import torch
from torch.nn.utils.rnn import pad_sequence
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape
from typing import List, Dict, Any, Optional

from data.paths import DATA_PATH
from models.messages import Role, QUESTION_PLACEHOLDER
from curriculum.exercise_with_answers import ExerciseWithAnswers
from training import IGNORE_INDEX
from training.utils import tokenize, read_exercises, ensure_path_exists, extract_question, extract_material_and_question

DISTRACTOR_PROB = 0.6


def prepare_answer_tokens(llm, content: str, max_length: int, truncated: bool) -> torch.Tensor:
    """Tokenize answer content, handle EOS and max length."""
    tokens = llm.tokenize(content)
    if not truncated:
        tokens = llm.add_eos(tokens)
    if max_length:
        tokens = tokens[:, :max_length]
    return tokens


class StudentTeacherDataset(torch.utils.data.Dataset):
    """
    Dataset for training with exercises and teacher answers.
    Each sample includes both student and teacher prompts and answers.
    """
    def __init__(
        self,
        llm,
        filenames: List[str],
        verbose: bool = False,
        datapath: Path = DATA_PATH,
        max_length: int = 0,
        debug: bool = False,
    ):
        assert isinstance(filenames, list), "filenames should be a list"
        self.samples: List[Dict[str, Any]] = []
        self.lesson_names: List[str] = []
        if verbose:
            print("==== StudentTeacherDataset ====", flush=True)

        for filename in filenames:
            filepath = Path(datapath) / filename
            ensure_path_exists(filepath)
            lesson_name = os.path.splitext(os.path.basename(filename))[0]
            self.lesson_names.append(lesson_name)
            lesson_ix = len(self.lesson_names) - 1
            exercises = read_exercises(filepath)
            training_pairs = 0
            for exercise in exercises:
                question = extract_question(exercise.messages[-1])
                prompt_with_tips = llm.messages_to_prompt(exercise.messages)
                student_prompt_tokens, teacher_prompt_tokens = tokenize(prompt_with_tips, llm)

                for choice in exercise.answer_choices:
                    answer_tokens = prepare_answer_tokens(llm, choice.content, max_length, choice.truncated)
                    sample = {
                        "student_prompt_tokens": student_prompt_tokens,
                        "teacher_prompt_tokens": teacher_prompt_tokens,
                        "answer_tokens": answer_tokens,
                        "teacher_answer": choice.content,
                        "lesson_ix": lesson_ix,
                        "question": question,
                    }
                    self.samples.append(sample)
                training_pairs += len(exercise.answer_choices)
            if verbose:
                print(f"{lesson_name}: {training_pairs} exercises with generated choices", flush=True)
        if verbose:
            print("==== /StudentTeacherDataset ====", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    @staticmethod
    def collate_fn(samples, padding_value, llm):
        """Collate batch for student-teacher training."""
        student_seqs, student_labels, teacher_seqs, teacher_masks = [], [], [], []
        for sample in samples:
            student_prompt_tokens = sample["student_prompt_tokens"]
            seq = torch.cat([student_prompt_tokens, sample["answer_tokens"]], dim=1)
            
            # The response is the target 
            labels = seq.clone()
            student_prompt_len = student_prompt_tokens.size(1)
            labels[0, :student_prompt_len] = IGNORE_INDEX
            student_seqs.append(seq[0])
            student_labels.append(labels[0])

            seq = torch.cat([sample["teacher_prompt_tokens"], sample["answer_tokens"]], dim=1)
            teacher_mask = torch.ones_like(seq[0], dtype=torch.bool)
            teacher_prompt_len = sample["teacher_prompt_tokens"].size(1)
            teacher_mask[:teacher_prompt_len] = 0
            teacher_seqs.append(seq[0])
            teacher_masks.append(teacher_mask)

        return {
            'student_seqs': pad_sequence(student_seqs, batch_first=True, padding_value=padding_value),
            'student_labels': pad_sequence(student_labels, batch_first=True, padding_value=IGNORE_INDEX).long(),
            'teacher_seqs': pad_sequence(teacher_seqs, batch_first=True, padding_value=padding_value),
            'teacher_masks': pad_sequence(teacher_masks, batch_first=True, padding_value=0).bool(),
            'lesson_ixs': torch.tensor([sample["lesson_ix"] for sample in samples]),
        }


class TeacherDataset(torch.utils.data.Dataset):
    """
    Dataset for teacher-only training (token loss).
    Each sample is a prompt and single answer, optionally with distractor context.
    """
    def __init__(
        self,
        llm,
        filenames: List[str],
        verbose: bool = False,
        datapath: Path = DATA_PATH,
        max_length: int = 0,
        distractor_dataset: str = "",
    ):
        assert isinstance(filenames, list), "filenames should be a list"
        self.samples: List[Dict[str, Any]] = []
        self.lesson_names: List[str] = []

        self.distractor_dataset = build_distractor_dataset(distractor_dataset) if distractor_dataset else None

        for filename in filenames:
            filepath = Path(datapath) / filename
            ensure_path_exists(filepath)
            lesson_name = os.path.splitext(os.path.basename(filename))[0]
            self.lesson_names.append(lesson_name)
            lesson_ix = len(self.lesson_names) - 1

            exercises = read_exercises(filepath)

            for exercise in exercises:
                if len(exercise.answer_choices) != 1:
                    raise NotImplementedError("Multiple choices per answer are not currently supported in token loss training")
               
                answer_choice = exercise.answer_choices[0]
                answer_tokens = prepare_answer_tokens(llm, answer_choice.content, max_length, answer_choice.truncated)
                prompt_with_tips = llm.messages_to_prompt(exercise.messages)
                student_prompt_tokens, teacher_prompt_tokens = tokenize(prompt_with_tips, llm)

                sample = {
                    "prompt_tokens": teacher_prompt_tokens,
                    "student_prompt_tokens": student_prompt_tokens,
                    "answer_tokens": answer_tokens,
                    "lesson_ix": lesson_ix,
                }

                if self.distractor_dataset:
                    material, question = extract_material_and_question(exercise.messages[-1])
                    prompt_placeholder = llm.messages_to_prompt(exercise.messages, placeholder=True)
                    sample.update({
                        "question": question,
                        "material": material,
                        "prompt_placeholder": prompt_placeholder,
                    })

                self.samples.append(sample)

            if verbose:
                print(f"{lesson_name}: {len(self.samples)} exercises with answers", flush=True)
        if verbose:
            print("==== /TeacherDataset ====", flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def collate_fn(self, samples, padding_value, llm, max_total_length=0):
        open_book_seqs = []
        open_book_labels = []
        closed_book_seqs = []
        closed_book_labels = []
        for sample in samples:
            if self.distractor_dataset:
                # Compose material (with distractors) + question
                material = sample["material"]
                question = sample["question"]
                distractors = self.distractor_dataset.sample()
                if np.random.rand() < DISTRACTOR_PROB:
                    # Insert context among distractors at a random position
                    idx = np.random.randint(0, len(distractors) + 1)
                    context_list = distractors[:idx] + [material] + distractors[idx:]
                    mixed_material = "\n\n".join(context_list)
                else:
                    mixed_material = "\n\n".join(distractors)
                full_question = (mixed_material + "\n\n" + question.strip()).strip()
                # Replace the placeholder in the template
                prompt = sample["prompt_placeholder"].replace(QUESTION_PLACEHOLDER, full_question)
                prompt_tokens = llm.tokenize(prompt)
                open_book_seq = torch.cat([prompt_tokens, sample["answer_tokens"]], dim=1)
                prompt_len = prompt_tokens.size(1)
            else:
                # use original tokens
                open_book_seq = torch.cat([sample["prompt_tokens"], sample["answer_tokens"]], dim=1)
                prompt_len = sample["prompt_tokens"].size(1)

            # Open-book targets: prompt + response, mask out prompt
            open_book_target_labels = open_book_seq.clone()
            open_book_target_labels[0, :prompt_len] = IGNORE_INDEX
            if max_total_length:
                open_book_seq = open_book_seq[:, -max_total_length:]
                open_book_target_labels = open_book_target_labels[:, -max_total_length:]

            # Closed-book side (always plain, no distractors)
            closed_book_seq = torch.cat([sample["student_prompt_tokens"], sample["answer_tokens"]], dim=1)
            closed_book_target_labels = closed_book_seq.clone()
            closed_book_prompt_len = sample["student_prompt_tokens"].size(1)
            closed_book_target_labels[0, :closed_book_prompt_len] = IGNORE_INDEX
            if max_total_length:
                closed_book_seq = closed_book_seq[:, -max_total_length:]
                closed_book_target_labels = closed_book_target_labels[:, -max_total_length:]

            open_book_seqs.append(open_book_seq[0])
            open_book_labels.append(open_book_target_labels[0])
            closed_book_seqs.append(closed_book_seq[0])
            closed_book_labels.append(closed_book_target_labels[0])

        open_book_seqs = pad_sequence(open_book_seqs, batch_first=True, padding_value=padding_value)
        open_book_labels = pad_sequence(open_book_labels, batch_first=True, padding_value=IGNORE_INDEX).long()
        lesson_ixs = torch.tensor([sample["lesson_ix"] for sample in samples])
        closed_book_seqs = pad_sequence(closed_book_seqs, batch_first=True, padding_value=padding_value)
        closed_book_labels = pad_sequence(closed_book_labels, batch_first=True, padding_value=IGNORE_INDEX).long()

        return {
            'open_book_seqs': open_book_seqs,
            'open_book_labels': open_book_labels,
            'lesson_ixs': lesson_ixs,
            'closed_book_seqs': closed_book_seqs,
            'closed_book_labels': closed_book_labels,
        }

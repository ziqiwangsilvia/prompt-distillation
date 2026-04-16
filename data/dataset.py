import numpy as np
import os
from pathlib import Path
import re
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Any, Optional

from data.paths import DATA_PATH
from data.tool_call_format import to_native_format, normalize_tool_call
from models.messages import Role, QUESTION_PLACEHOLDER
from curriculum.exercise_with_answers import ExerciseWithAnswers
from training import IGNORE_INDEX
from training.utils import tokenize_teacher_student, read_exercises, ensure_path_exists, extract_question, extract_material_and_question

DISTRACTOR_PROB = 0.6


def prepare_answer_tokens(llm, content: str, max_length: int, truncated: bool, use_tool_token: bool = False) -> torch.Tensor:
    """Tokenize answer content and add EOS."""
    is_tool = use_tool_token and llm.model_family == "llama" and content.lstrip().startswith('{"name"')
    if is_tool:
        content = "<|python_tag|>" + content
    tokens = llm.tokenize(content)
    if is_tool:
        eom_id = llm.tokenizer.convert_tokens_to_ids("<|eom_id|>")
        tokens = torch.cat([tokens, torch.tensor([[eom_id]])], dim=1)
    else:
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
        teacher_llm=None,
        tools: list = None,
        use_tool_token: bool = False,
        max_samples: int = 0,
    ):
        assert isinstance(filenames, list), "filenames should be a list"
        self.samples: List[Dict[str, Any]] = []
        self.lesson_names: List[str] = []
        t_llm = teacher_llm or llm
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
            for ex_i, exercise in enumerate(exercises):
                if max_samples and len(self.samples) >= max_samples:
                    break
                question = extract_question(exercise)
                material, _ = extract_material_and_question(exercise)
                student_closed_tokens, student_open_tokens, teacher_tokens = tokenize_teacher_student(material, question, llm, teacher_llm=teacher_llm, tools=tools, student_tools=tools if use_tool_token else None)

                if verbose and ex_i == 0:
                    print(f"  [Example] Teacher sees: {t_llm.decode(teacher_tokens)[:200]}...")
                    print(f"  [Example] Student (open) sees: {llm.decode(student_open_tokens)[:200]}...")
                    print(f"  [Example] Student (closed) sees: {llm.decode(student_closed_tokens)[:200]}...")

                for choice in exercise.answer_choices:
                    student_content = choice.content
                    teacher_content = to_native_format(choice.content, t_llm.model_family) if use_tool_token and teacher_llm else choice.content
                    alignment_text = normalize_tool_call(choice.content) if use_tool_token and teacher_llm else choice.content
                    answer_tokens = prepare_answer_tokens(llm, student_content, max_length, choice.truncated, use_tool_token=use_tool_token)
                    teacher_answer_tokens = prepare_answer_tokens(t_llm, teacher_content, max_length, choice.truncated) if teacher_llm else answer_tokens
                    sample = {
                        "student_prompt_tokens": student_closed_tokens,
                        "student_open_prompt_tokens": student_open_tokens,
                        "teacher_prompt_tokens": teacher_tokens,
                        "answer_tokens": answer_tokens,
                        "teacher_answer_tokens": teacher_answer_tokens,
                        "teacher_answer": alignment_text,
                        "student_answer_text": student_content,
                        "teacher_answer_text": teacher_content,
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
        student_open_seqs, student_open_labels = [], []
        student_closed_seqs, student_closed_labels = [], []
        teacher_seqs, teacher_masks = [], []
        for sample in samples:
            answer = sample["answer_tokens"]

            # Student open-book (with material)
            open_prompt = sample["student_open_prompt_tokens"]
            seq = torch.cat([open_prompt, answer], dim=1)
            labels = seq.clone()
            labels[0, :open_prompt.size(1)] = IGNORE_INDEX
            student_open_seqs.append(seq[0])
            student_open_labels.append(labels[0])

            # Student closed-book (no material)
            closed_prompt = sample["student_prompt_tokens"]
            seq = torch.cat([closed_prompt, answer], dim=1)
            labels = seq.clone()
            labels[0, :closed_prompt.size(1)] = IGNORE_INDEX
            student_closed_seqs.append(seq[0])
            student_closed_labels.append(labels[0])

            # Teacher open-book (with material, teacher tokenizer)
            teacher_prompt = sample["teacher_prompt_tokens"]
            teacher_answer = sample["teacher_answer_tokens"]
            seq = torch.cat([teacher_prompt, teacher_answer], dim=1)
            teacher_mask = torch.ones_like(seq[0], dtype=torch.bool)
            teacher_mask[:teacher_prompt.size(1)] = 0
            teacher_seqs.append(seq[0])
            teacher_masks.append(teacher_mask)

        return {
            'student_open_seqs': pad_sequence(student_open_seqs, batch_first=True, padding_value=padding_value),
            'student_open_labels': pad_sequence(student_open_labels, batch_first=True, padding_value=IGNORE_INDEX).long(),
            'student_closed_seqs': pad_sequence(student_closed_seqs, batch_first=True, padding_value=padding_value),
            'student_closed_labels': pad_sequence(student_closed_labels, batch_first=True, padding_value=IGNORE_INDEX).long(),
            'teacher_seqs': pad_sequence(teacher_seqs, batch_first=True, padding_value=padding_value),
            'teacher_masks': pad_sequence(teacher_masks, batch_first=True, padding_value=0).bool(),
            'teacher_answers': [sample["teacher_answer"] for sample in samples],
            'student_answer_texts': [sample.get("student_answer_text", sample["teacher_answer"]) for sample in samples],
            'teacher_answer_texts': [sample.get("teacher_answer_text", sample["teacher_answer"]) for sample in samples],
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
        tools: list = None,
        use_tool_token: bool = False,
        max_samples: int = 0,
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

            for ex_i, exercise in enumerate(exercises):
                if max_samples and len(self.samples) >= max_samples:
                    break
                if len(exercise.answer_choices) != 1:
                    raise NotImplementedError("Multiple choices per answer are not currently supported in token loss training")

                answer_choice = exercise.answer_choices[0]
                answer_tokens = prepare_answer_tokens(llm, answer_choice.content, max_length, answer_choice.truncated, use_tool_token=use_tool_token)
                material, question = extract_material_and_question(exercise)
                student_closed_tokens, student_open_tokens, _ = tokenize_teacher_student(material, question, llm, tools=tools, student_tools=tools if use_tool_token else None)

                if verbose and ex_i == 0:
                    print(f"  [Example] Open-book sees: {llm.decode(student_open_tokens)[:200]}...")
                    print(f"  [Example] Closed-book sees: {llm.decode(student_closed_tokens)[:200]}...")

                sample = {
                    "prompt_tokens": student_open_tokens,
                    "student_prompt_tokens": student_closed_tokens,
                    "answer_tokens": answer_tokens,
                    "lesson_ix": lesson_ix,
                }

                if self.distractor_dataset:
                    mat, q = extract_material_and_question(exercise)
                    prompt_placeholder = llm.messages_to_prompt(exercise.messages, placeholder=True)
                    sample.update({
                        "question": q,
                        "material": mat,
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
                material = sample["material"]
                question = sample["question"]
                distractors = self.distractor_dataset.sample()
                if np.random.rand() < DISTRACTOR_PROB:
                    idx = np.random.randint(0, len(distractors) + 1)
                    context_list = distractors[:idx] + [material] + distractors[idx:]
                    mixed_material = "\n\n".join(context_list)
                else:
                    mixed_material = "\n\n".join(distractors)
                full_question = (mixed_material + "\n\n" + question.strip()).strip()
                prompt = sample["prompt_placeholder"].replace(QUESTION_PLACEHOLDER, full_question)
                prompt_tokens = llm.tokenize(prompt)
                open_book_seq = torch.cat([prompt_tokens, sample["answer_tokens"]], dim=1)
                prompt_len = prompt_tokens.size(1)
            else:
                open_book_seq = torch.cat([sample["prompt_tokens"], sample["answer_tokens"]], dim=1)
                prompt_len = sample["prompt_tokens"].size(1)

            open_book_target_labels = open_book_seq.clone()
            open_book_target_labels[0, :prompt_len] = IGNORE_INDEX
            if max_total_length:
                open_book_seq = open_book_seq[:, -max_total_length:]
                open_book_target_labels = open_book_target_labels[:, -max_total_length:]

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

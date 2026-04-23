import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Any

from paths import DATA_PATH
from models.messages import QUESTION_PLACEHOLDER
from training import IGNORE_INDEX
from data.samples import load_samples


DISTRACTOR_PROB = 0.6


class StudentTeacherDataset(torch.utils.data.Dataset):
    """Dataset for distillation training (logit loss / DPO).
    Each sample includes student (open/closed-book) and teacher prompts and answers.
    """
    def __init__(self, llm, filenames, verbose=False, datapath=DATA_PATH,
                 max_length=0, debug=False, teacher_llm=None, tools=None,
                 use_tool_token=False, max_samples=0, multi_turn=False):
        if verbose:
            print("==== StudentTeacherDataset ====", flush=True)
        self.samples, self.lesson_names = load_samples(
            llm, filenames, datapath, max_samples, multi_turn,
            tools, use_tool_token, max_length, teacher_llm=teacher_llm)
        if verbose:
            print(f"  {len(self.samples)} training pairs", flush=True)
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


def build_distractor_dataset(distractor_dataset):
    """Build distractor dataset for mixing with training data."""
    from datasets import load_dataset
    ds = load_dataset(distractor_dataset, split="train")
    class DistractorSampler:
        def __init__(self, texts, k=5):
            self.texts = texts
            self.k = k
        def sample(self):
            idxs = np.random.choice(len(self.texts), self.k, replace=False)
            return [self.texts[i] for i in idxs]
    return DistractorSampler([row["context"] for row in ds])


class StudentDataset(torch.utils.data.Dataset):
    """Dataset for SFT-only training (token loss, no teacher).
    Each sample is a prompt and single answer, optionally with distractor context.
    """
    def __init__(self, llm, filenames, verbose=False, datapath=DATA_PATH,
                 max_length=0, distractor_dataset="", tools=None,
                 use_tool_token=False, max_samples=0, multi_turn=False):
        self.distractor_dataset = build_distractor_dataset(distractor_dataset) if distractor_dataset else None
        if verbose:
            print("==== StudentDataset ====", flush=True)
        self.samples, self.lesson_names = load_samples(
            llm, filenames, datapath, max_samples, multi_turn,
            tools, use_tool_token, max_length,
            include_distractor_fields=bool(self.distractor_dataset))
        for s in self.samples:
            s["prompt_tokens"] = s["student_open_prompt_tokens"]
        if verbose:
            print(f"  {len(self.samples)} training pairs", flush=True)
            print("==== /StudentDataset ====", flush=True)

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

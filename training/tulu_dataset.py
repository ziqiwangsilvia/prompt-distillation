import torch
from torch.nn.utils.rnn import pad_sequence

from core.messages import Message, Role
from training import IGNORE_INDEX
from training.utils import tokenize

def tulu_collate_fn(elems, padding_value, llm, max_length, system_msg, lesson_ix, logit_collate_fn, use_batch_size):
    samples = []
    for elem in elems:
        if elem['messages'][0]['role'] != 'user' or elem['messages'][1]['role'] != 'assistant':
            # Misformatted training example, continuing
            continue
        messages = [
            Message(content=elem['messages'][0]['content'], role=Role.USER),
        ]
        prompt_with_tips = llm.messages_to_prompt(messages, no_template=True)
        prompt_with_tips += "\nAnswer: "
        student_prompt_tokens, teacher_prompt_tokens = tokenize(prompt_with_tips, llm)

        content = elem['messages'][1]['content']
        answer_tokens = llm.tokenize(content)
        answer_tokens = llm.add_eos(answer_tokens)

        if max_length:
            max_answer_len = max_length - len(student_prompt_tokens[0])
            if max_answer_len <= 1:
                # Too long example
                continue
            answer_tokens = answer_tokens[:, :max_answer_len]
        sample = {
            "student_prompt_tokens": student_prompt_tokens,
            "teacher_prompt_tokens": teacher_prompt_tokens,
            "answer_tokens": answer_tokens,
            "teacher_answer": content,
            "lesson_ix": lesson_ix,
            "question": "",
        }
        samples.append(sample)

        # We cheat by sampling more instances we can use per batch. The tulu dataset is very large, we can afford to do that.
        # If the example is malformatted or too long, we skip it a few lines above
        if len(samples) == use_batch_size:
            break

    if len(samples):
        return logit_collate_fn(samples)
    return []

def merge_with_tulu_batch(batch1, batch2, padding_value):
    """
    This function merges the tulu regularization batch with the default batch
    """
    if not batch2:
        return batch1
    merged_student_seqs = pad_sequence(
        [*batch1["student_seqs"], *batch2["student_seqs"]], batch_first=True, padding_value=padding_value
    )
    merged_student_labels = pad_sequence(
        [*batch1["student_labels"], *batch2["student_labels"]], batch_first=True, padding_value=IGNORE_INDEX
    ).long()
    merged_teacher_seqs = pad_sequence(
        [*batch1["teacher_seqs"], *batch2["teacher_seqs"]], batch_first=True, padding_value=padding_value
    )
    merged_teacher_masks = pad_sequence(
        [*batch1["teacher_masks"], *batch2["teacher_masks"]], batch_first=True, padding_value=0
    ).bool()
    merged_lesson_ixs = torch.cat([batch1["lesson_ixs"], batch2["lesson_ixs"]])
    return {
        'student_seqs': merged_student_seqs,
        'student_labels': merged_student_labels,
        'teacher_seqs': merged_teacher_seqs,
        'teacher_masks': merged_teacher_masks,
        'lesson_ixs': merged_lesson_ixs,
    }


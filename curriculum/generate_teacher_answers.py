# %%
import os
import re
import time
import warnings
import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict

import torch
from tqdm import tqdm
from openai import OpenAI
from vllm import LLM as vLLM

from data.paths import DATA_PATH
from data.naming import generate_augmented_filename, generate_lesson_filename, generate_exam_filename
from models.llm import LLM, get_model_family
from models.messages import Message, Role, merge_messages
from models.configs import create_model_flags, MODEL_CONFIGS, get_model_config
from models.tool_call_format import convert_tool_call_format
from models.utils import generate_sampling_params
from curriculum.lesson import read_lessons, Lesson, Exercise
from curriculum.exercise_with_answers import ExerciseWithAnswers, Choice, xml_dump
from training.utils import clean_xml_content


def generate_prompt(
    llm: LLM,
    lesson: Lesson,
    max_total_tokens: int,
    max_new_tokens: int,
) -> Tuple[List[Tuple[str, int]], List[Exercise]]:
    """Generate prompts and exercises from a lesson."""
    prompts = []
    exercises = []
    lesson.create_exercise_prompts(verbose=False)

    for ex in lesson.exercises:
        n_tokens_prompt = len(llm.tokenize(ex.student_prompt))
        max_tokens_to_generate = max_total_tokens - n_tokens_prompt

        if max_new_tokens > 0:
            max_tokens_to_generate = min(max_new_tokens, max_tokens_to_generate)

        if max_tokens_to_generate <= 10:
            warnings.warn(f"Too few tokens left for the answer: {max_tokens_to_generate}.", stacklevel=2)
        elif max_tokens_to_generate <= 0:
            raise ValueError(f"Too many tokens in the prompt: {n_tokens_prompt}, while the limit is {max_total_tokens}.")

        prompt = ex.teacher_prompt
        messages = [Message(Role.USER, prompt)]
        messages = merge_messages(messages)
        prompt = llm.messages_to_prompt(messages)
        prompt = prompt.replace("&lt;", "<").replace("&gt;", ">")

        prompts.append((prompt, max_tokens_to_generate))
        exercises.append(ex)

    return prompts, exercises


def _is_mixed_response(text: str) -> bool:
    """Detect responses that mix text with a tool call."""
    tc = re.search(r'<tool_call>\s*\{.*\}\s*(</tool_call>)?', text, re.DOTALL)
    if tc is None:
        return False  # pure text, no tool call
    before = text[:tc.start()].strip()
    after = text[tc.end():].strip()
    return bool(before or after)


def process_answers(llm: LLM, exercise: Exercise, answers: List[str],
                    source_family: str = "", target_family: str = "") -> ExerciseWithAnswers:
    """Process answers for an exercise, optionally converting tool-call format."""
    answer_choices = []

    for answer in answers:
        if not isinstance(answer, str):
            answer = answer.text

        if _is_mixed_response(answer):
            print(f"  Skipping mixed response: {answer[:80]}...")
            continue

        # Strip stop token if present, drop truncated answers
        raw_tokens = llm.tokenize(answer)
        terminators = llm.get_terminators()
        if raw_tokens.numel() > 0 and raw_tokens[0, -1] in terminators:
            answer = llm.decode(raw_tokens[:, :-1])
        else:
            print(f"  Skipping truncated answer: {answer[:80]}...")
            continue

        if source_family and target_family and source_family != target_family:
            answer = convert_tool_call_format(answer, source_family, target_family)

        choice = Choice(answer)
        answer_choices.append(choice)

    messages = [Message(Role.USER, exercise.teacher_prompt_with_tips_tags)]
    return ExerciseWithAnswers(
        messages, 
        answer_choices, 
        model_answer=exercise.model_answer, 
        grading_str=exercise.grading_str
    )


def save_to_xml(lesson_id: str, exercises_with_answers: List[ExerciseWithAnswers],
                temperature: float, n_choices: int, model_flags: Dict[str, bool]):
    """Save exercises with answers to XML file."""
    root = ET.Element("exercises_with_answers")
    ET.SubElement(root, "temperature", value=str(temperature))

    for ex in exercises_with_answers:
        ex.to_xml(root)

    fname = generate_augmented_filename(lesson_id, n_choices, temperature, model_flags)
    path = DATA_PATH / fname
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as file:
        xml_dump(root, file)

    print(f"Saved to {path}")


def setup_models(base: str, vllm_hostname: str = "") -> Tuple[LLM, object]:
    """Setup LLM and vLLM models."""
    if base not in MODEL_CONFIGS:
        raise ValueError(f"Model '{base}' not supported. Available models: {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[base]
    opening_message = Message(Role.SYSTEM, config.system_message)

    llm = LLM(base, opening_message=opening_message)

    if vllm_hostname:
        vllm_client = OpenAI(base_url=f"http://{vllm_hostname}:8000/v1", api_key="token-abc123")
        vllm_client._model_name = config.vllm_model
    else:
        vllm_client = vLLM(config.vllm_model, tensor_parallel_size=torch.cuda.device_count())

    return llm, vllm_client


def main(
    base: str = "llama3-8b-instruct",
    student_base: str = "",
    generate_lesson: bool = False,
    generate_exam: bool = False,
    lesson_num_choices: int = 1,
    exam_num_choices: int = 1,
    lesson_temp: float = 1.5,
    exam_temp: float = 0.25,
    max_total_tokens: int = 1024,
    max_new_tokens: int = 500,
    dataset_family: str = "squadshifts",
    dataset: str = "nyt",
    variant: str = "default",
    question_model: str = "llama3-8b-instruct",
    max_items: int = 1000,
    train_questions: int = 30,
    question_temperature: float = 1.5,
    chunk_size: int = 10000,
    verbose: bool = False,
    vllm_hostname: str = "",
):
    assert not (generate_lesson and generate_exam), "The code doesn't support generating lesson and exam simultaneously"
    # Setup models
    llm, vllm_client = setup_models(base, vllm_hostname)
    model_flags = create_model_flags(base)

    # Setup processing modes
    if generate_lesson:
        xml_name = generate_lesson_filename(
            dataset_family, dataset, variant, question_model, train_questions, question_temperature, max_items
        )
    if generate_exam:
        xml_name = generate_exam_filename(
            dataset_family, dataset, variant, max_items
        )

    temperature = lesson_temp if generate_lesson else exam_temp
    num_choices = lesson_num_choices if generate_lesson else exam_num_choices

    # Setup sampling parameters
    sampling_params = generate_sampling_params(max_total_tokens, temperature)
    print(f"Processing {xml_name}", flush=True)

    # Read lessons
    try:
        lessons = read_lessons(xml_name)
    except ET.ParseError:
        cleaned_xml_filename = clean_xml_content(xml_name)
        lessons = read_lessons(cleaned_xml_filename)

    # Check if output already exists before expensive generation
    first_lesson_id = next(iter(lessons))
    output_fname = generate_augmented_filename(
        first_lesson_id.rsplit('_', 1)[0], num_choices, temperature, model_flags
    )
    output_path = DATA_PATH / output_fname
    if output_path.exists():
        print(f"{output_path} already exists — skipping.", flush=True)
        return

    # Generate prompts and exercises
    prompts = []
    exercises = []
    print(f"Number of lessons: {len(lessons)}", flush=True)

    for lesson_id, lesson in lessons.items():
        p, e = generate_prompt(llm, lesson, max_total_tokens, max_new_tokens)
        prompts += p
        exercises += e

    assert len(prompts) == len(exercises)
    print(f"Number of prompts: {len(prompts)}", flush=True)

    # Generate answers
    start_time = time.time()
    prompts_only = [p for p, _ in prompts]
    answers = []

    if isinstance(vllm_client, OpenAI):
        for prompt in tqdm(prompts_only):
            resp = vllm_client.completions.create(
                model=vllm_client._model_name,
                prompt=prompt,
                n=num_choices,
                max_tokens=max_total_tokens,
                temperature=temperature,
                top_p=1.0,
                extra_body={"top_k": 50, "skip_special_tokens": False},
            )
            answers.append([c.text for c in resp.choices])
    else:
        for i in tqdm(range(0, len(prompts_only), chunk_size)):
            chunk = prompts_only[i:i + chunk_size]
            outputs = vllm_client.generate(chunk, sampling_params)
            for output in outputs:
                answers.append(output.outputs)

    end_time = time.time()
    print(f"Generation time: {end_time - start_time:.4f} s", flush=True)

    assert len(prompts) == len(exercises) == len(answers)
    assert len(answers[0]) == num_choices

    # Process answers
    teacher_family = get_model_family(get_model_config(base).vllm_model)
    student_family = get_model_family(get_model_config(student_base).vllm_model) if student_base else teacher_family
    exercises_with_answers = []
    for ex, ans in zip(exercises, answers):
        ewa = process_answers(llm, ex, ans, teacher_family, student_family)
        if ewa.answer_choices:
            exercises_with_answers.append(ewa)
        else:
            print(f"  Dropped exercise with 0 valid answers")

    # Save results
    save_to_xml(
        lesson_id.rsplit('_', 1)[0], 
        exercises_with_answers, 
        temperature, 
        num_choices,
        model_flags,
    )


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)

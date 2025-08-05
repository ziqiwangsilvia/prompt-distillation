import argparse
import asyncio
from collections import Counter
import csv
from functools import partial
import glob
import httpx
import logging
import os
import random
import re
import sys
import time  # noqa
from typing import Any, Dict, List

from openai import AsyncOpenAI
from transformers import AutoTokenizer  # noqa

from core.llm import LLM
from core.messages import Message, Role
from core.model_configs import get_model_config
from core.utils import generate_extra_body
from evaluation.utils import async_wrapper

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def extract_grade(text: str) -> Dict[str, Any]:
    """
    Returns ``{"valid_grade": bool, "grade": Optional[bool]}``
    """
    m = re.search(r"<grade>(.*?)</grade>", text, flags=re.I)
    content = m.group(1) if m else text
    lowered = content.lower()

    if "true" in lowered:
        return {"valid_grade": True, "grade": True}
    if "false" in lowered:
        return {"valid_grade": True, "grade": False}
    return {"valid_grade": False}


def create_prompt(
    q: str,
    gt: str | List[str],
    a: str,
    dataset_family: str,
    llm: LLM,
) -> str:
    """Return a single string prompt."""
    if dataset_family == "squadshifts":
        prompt = f"""
Here is a question, the list of accepted ground-truth answers and the proposed answer. Please evaluate if the answer is true or false and return the reasoning and grade as xml. If the answer matches any of the ground-truth answers, the grade should be true. Example:

Example
Question: What was the punishment for Mattingly for not getting a haircut?
Ground-truth answer: ['fined and benched', 'benched', 'fined and benched']
Proposed answer: Mattingly was benched for 20 games as punishment for not getting a haircut.                                                                                                                                                                            Output:                                                                                                                             <reasoning>The ground-truth answer is that Mattingly was fined and benched for not getting a haircut. However, simply answering that Mattingly was benched is also a valid answer. The proposed answer correctly mentions that Mattingly was benched for not getting a haircut</reasoning>
<grade>true</grad>

Example:
Question: For what was Mattingly once fined and benched for?
Ground-truth answer: ['not getting a haircut']
Proposed Answer: Mattingly was fined and benched for not playing in a game, which was likely a form of protest against the team's management or the way he was being treated.
Output:
<reasoning>According to the ground-truth answer, Mattingly was fined and benched for not getting a haircut. The proposed answer does not answer the question accurately, and mentions a protest as a reason. However, this is clearly different from the ground-truth answer.</reasoning>
<grade>false</grade>

Example:
Question: Which baseball teams was Mattingly apart of?
Ground-truth answer: ['Yankee', 'Yankees']
Proposed answer: Mattingly was a part of the New York Yankees and the Los Angeles Dodgers baseball teams.

Output:
<reasoning>The ground-truth answer indicates that Mattingly was a part of the "Yankees" team. The proposed answer states that he was a part of the New York Yankees, which matches one of the ground-truth answers. However, it also mentions the Los Angeles Dodgers, which is not included in the ground-truth answers. Since the additional information about the Dodgers does not directly contradict the truth of him being on the Yankees, the primary question's focus remains valid.</reasoning>
<grade>true</grade>

Please grade the following question using the xml-template given:
Question: {q}
Ground-truth answer: {gt}
Proposed answer: {a}
"""
    elif dataset_family == "hotpotqa":
        prompt = f"""
Here is a question, the ground-truth answer and the proposed answer. Please evaluate if the answer is true or false and return the reasoning and grade as xml. Example:

Example
Question: What was the punishment for Mattingly for not getting a haircut?
Ground-truth answer: benched
Proposed answer: Mattingly was benched for 20 games as punishment for not getting a haircut.

Output:
<reasoning>The ground-truth answer is that Mattingly was benched for not getting a haircut. The proposed answer correctly mentions that Mattingly was benched for not getting a haircut</reasoning>
<grade>true</grad>

Example:
Question: For what was Mattingly once fined and benched for?
Ground-truth answer: not getting a haircut
Proposed Answer: Mattingly was fined and benched for not playing in a game, which was likely a form of protest against the team's management or the way he was being treated.

Output:
<reasoning>According to the ground-truth answer, Mattingly was fined and benched for not getting a haircut. The proposed answer does not answer the question accurately, and mentions a protest as a reason. However, this is clearly different from the ground-truth answer.</reasoning>
<grade>false</grade>

Example:
Question: Which baseball teams was Mattingly apart of?
Ground-truth answer: Yankees
Proposed answer: Mattingly was a part of the New York Yankees and the Los Angeles Dodgers baseball teams.

Output:
<reasoning>The ground-truth answer indicates that Mattingly was a part of the "Yankees" team. The proposed answer states that he was a part of the New York Yankees, which matches one of the ground-truth answers. However, it also mentions the Los Angeles Dodgers, which is not included in the ground-truth answer. Since the additional information about the Dodgers does not directly contradict the truth of him being on the Yankees, the primary question's focus remains valid.</reasoning>
<grade>true</grade>
                                                                                                                                    Please grade the following question using the xml-template given:
Question: {q}
Ground-truth answer: {gt}
Proposed answer: {a}
"""
    messages = [Message(Role.USER, prompt)]
    return llm.messages_to_prompt(messages)


def main(
    base: str = "llama3-8b-instruct",
    dataset_family: str = "squadshifts",
    input_path: str = "",
    max_items: int = 100000,
    max_tokens: int = 512,
    grading_temperature: float = 0.0,
    vllm_hostname: str = "",
) -> None:

    try:
        cfg = get_model_config(base)
    except ValueError as e:  # fallback for legacy names etc.
        raise SystemExit(str(e)) from None

    opening_message = Message(Role.SYSTEM, cfg.system_message)
    model_id: str = cfg.vllm_model  # e.g. "meta-llama/Meta-Llama-3-8B-Instruct"
    model_suffix = f"{cfg.flag_name}"

    # Note: initialize LLM but do not load the model
    # This is for prompt creation
    llm = LLM(base, opening_message=opening_message)
    api_base = f"http://{vllm_hostname}:8000/v1"
    vllm_client = AsyncOpenAI(base_url=api_base, api_key="token-abc123")
    extra_body = generate_extra_body(base)

    files = glob.glob(input_path, recursive=True)
    random.shuffle(files)  # allow parallel runs with less contention

    correctness_dict: Dict[str, float] = {}

    for csv_path in files:
        # Derive grading filename
        stem = os.path.basename(csv_path)
        grading_path = os.path.join(
            os.path.dirname(csv_path),
            f"grading_{model_suffix}_{stem}",
        )
 
        n_questions = 0
        n_correct = 0
 
        print("Starting to grade the answers in:", csv_path)
        print("Writing the grading to:", grading_path)
 
        if os.path.exists(grading_path):
            print(f"Grading file {grading_path} exists already, reading previous grading")
            with open(grading_path, mode='r', newline='') as outfile:
                reader = csv.reader(outfile, delimiter=';')
                for line in reader:
                    grade = line[0]
                    if grade == 'True':
                        n_questions += 1
                        n_correct += 1
                    elif grade == 'False':
                        n_questions += 1
        else:
            with open(csv_path, mode='r', newline='', encoding='utf-8', errors='ignore') as infile:
                reader = csv.reader(infile, delimiter=';')
  
                prompts = []
                questions = []
                gt_answers = []
                generated_answers = []
                # Read each row from the output file, process it, and write to the grading file
                for i, row in enumerate(reader):
                    if max_items is not None and i >= max_items > 0:
                        break
                    # Assign the parts to variables
                    question = row[0]  # The question
                    if dataset_family == "squadshifts":
                        try:
                            ground_truth_answer = eval(row[1])  # The ground truth answers, converting string list to actual list
                        except:
                            print(f"Faulty instance", flush=True)
                            continue
                    else:
                        ground_truth_answer = row[1]
                    generated_answer = row[2]  # The generated answer
  
                    prompt = create_prompt(question, ground_truth_answer, generated_answer, dataset_family, llm)
                    prompts.append(prompt)
                    questions.append(question)
                    gt_answers.append(ground_truth_answer)
                    generated_answers.append(generated_answer)
  
                print(f"Number of prompts: {len(prompts)}", flush=True)
                start_time = time.time()
                grading_outputs = asyncio.run(async_wrapper(vllm_client, model_id, prompts, extra_body, grading_temperature, max_tokens, temp_file=grading_path + ".temp"))
                end_time = time.time()
                print(f"Generation time: {end_time - start_time:.4f} s", flush=True)

            # If we run multiple gradings concurrently, make sure that the path does indeed not exist
            if os.path.exists(grading_path):
                print(f"Grading file {grading_path} has been created concurrently, skipping writing")
                continue
            elif len(grading_outputs) != len(prompts):
                print(f"Failure in grading file {csv_path}")
                print(f"Number of questions: {len(questions)}")
                print(f"Number of grades: {len(grading_outputs)}")
                continue

            with open(grading_path, mode='w', newline='') as outfile:
                writer = csv.writer(outfile, delimiter=';', escapechar='\\', quoting=csv.QUOTE_NONE)

                for (prompt, result, question, gt_answer, gen_answer) in zip(prompts,
                                                                             grading_outputs,
                                                                             questions,
                                                                             gt_answers,
                                                                             generated_answers):
                    grade_dict = extract_grade(result)
  
                    result = result.replace('\n', ' ')
  
                    if grade_dict['valid_grade']:
                        grade = grade_dict['grade']
                        n_questions += 1
                        n_correct += grade
                    else:
                        grade = "Invalid"
  
                    writer.writerow([grade, question, gt_answer, gen_answer, result])
                    outfile.flush()

        if n_questions > 0:
            correctness = n_correct/n_questions*100
            correctness_dict[grading_path] = correctness
            print(f"Number of graded questions {n_questions}, correct {n_correct}, correctness {correctness:.2f} %")
            print("Files have been processed and grading files created.")
        else:
            print(f"No answers found in {csv_path}", flush=True)

    # Sort the dictionary by correctness percentage in descending order
    sorted_correctness = sorted(correctness_dict.items(), key=lambda item: item[1], reverse=True)
    print("Sorted Evaluation Correctness:")
    for index, (grading_path, correctness) in enumerate(sorted_correctness, start=1):
        print(f"{index}. {grading_path}: {correctness:.2f}%", flush=True)


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)


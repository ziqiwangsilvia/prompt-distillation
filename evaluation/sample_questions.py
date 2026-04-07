import asyncio
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import csv
import httpx
from datasets import load_dataset
from openai import AsyncOpenAI
from core.file_naming import generate_question_path
from core.llm import LLM
from core.messages import Message, Role
from core.model_configs import MODEL_CONFIGS, get_model_config
from core.utils import generate_extra_body, load_squadshifts
from evaluation.utils import async_wrapper, get_gt_answer, get_rag_context

MAX_FAILURES = 20


def _extract_questions(text: str) -> List[str]:
    questions = re.findall(r'<question>(.*?)<\/question>', text, re.DOTALL)
    return questions


def _generate_prompt_async(context: str, llm: LLM) -> str:
    prompt = f"""Here is a paragraph of text:
{context}

Please generate challenging five trivia questions based on this paragraph. Do not make the questions multiple-choice. Do not assume that the person answering the questions has access to the paragraph. The questions must be understandable without access to the text. Do not output anything except the questions and format your output as in the followimg example:
<question>What is the capital of Japan?</question>
<question>How many months are there in a year?</question>
<question>What was the first name of Reagan?</question>
<question>How many goals did Messi score during the calendar year 2012</question>
<question>Where is the Santa Monica pier located?</question>"""
    
    messages = [Message(Role.USER, prompt)]
    return llm.messages_to_prompt(messages)


async def _sample_questions(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    extra_body: Dict,
    temperature: float,
    max_tokens: int,
    needed_calls: int,
    **kwargs: Dict,
) -> List[str]:
    """Generate question sets until needed_calls successful generations."""
    qs: List[str] = []
    failures = 0
    done = 0

    while done < needed_calls:
        rsp = await client.completions.create(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            extra_body=extra_body,
        )
        text = rsp.choices[0].text.strip()
        extracted = _extract_questions(text)
        if extracted and extracted[0].strip().lower() != "q1":
            qs.extend(extracted)
            done += 1
        else:
            failures += 1
            if failures >= MAX_FAILURES:
                print("[WARN] too many failed generations for a prompt – skipping")
                break
    return qs


def main(
    base: str = "llama3-8b-instruct",
    dataset_family: str = "squadshifts",
    dataset: str = "nyt",
    max_items: int = 1000,
    max_tokens: int = 512,
    train_questions: int = 30,
    temperature: float = 1.5,
    vllm_hostname: str = "",
) -> None:
    cfg = get_model_config(base)
    llm = LLM(base, opening_message=Message(Role.SYSTEM, cfg.system_message))

    client = AsyncOpenAI(base_url=f"http://{vllm_hostname}:8000/v1", api_key="token-abc123")
    extra_body = generate_extra_body(base)
    output_directory_name = f"{dataset_family}_{dataset}"

    if dataset_family == "squadshifts":
        ds = load_squadshifts(dataset)
    elif dataset_family == "hotpotqa":
        ds = load_dataset("hotpotqa/hotpot_qa", dataset, trust_remote_code=True)["validation"]
    else:
        raise NotImplementedError(f"Unknown dataset family '{dataset_family}'")
    
    output_file = Path(
        generate_question_path(
            dataset_family=dataset_family,
            dataset=dataset,
            model=base,
            questions=train_questions,
            temperature=temperature,
            max_items=max_items,
        )
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if output_file.exists():
        print(f"{output_file} already exists — skipping.", flush=True)
        return

    print(f"Writing questions to {output_file}", flush=True)
    
    questions_written = 0
    print(f"Output file for questions: {output_file}", flush=True)
    print(f"Starting to generate questions to be written into {output_file}", flush=True)
    contexts = []
    prompts = []
    for i, item in enumerate(ds):
        if max_items and i >= max_items > 0:
            break
        contexts += get_rag_context(item, dataset_family=dataset_family)

    for context in contexts:
        prompts.append(_generate_prompt_async(context, llm))

    start_time = time.time()
    questions_per_paragraph = asyncio.run(
        async_wrapper(
            client,
            cfg.vllm_model,
            prompts,
            extra_body,
            temperature,
            max_tokens,
            custom_fnc=_sample_questions,
            custom_fnc_extra_kwargs={"needed_calls": max(1, train_questions // 5)},
        )
    )
    end_time = time.time()
    print(f"Generation time: {end_time - start_time:.4f} s", flush=True)
    assert len(contexts) == len(questions_per_paragraph)

    written = 0
    with open(output_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=";")
        for context, q_list in zip(contexts, questions_per_paragraph):
            for q in q_list:
                writer.writerow([q.replace("\n", " "), context.replace("\n", " ")])
                written += 1
                file.flush()
    print(f"{written} questions generated and written into {output_file}, exiting", flush=True)


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)

import asyncio
import csv
import os
import re
import time
from pathlib import Path
from typing import List

from datasets import load_dataset
from openai import AsyncOpenAI
from core.llm import LLM
from core.messages import Message, Role, merge_messages
from core.model_configs import MODEL_CONFIGS, get_model_config
from core.utils import load_squadshifts
from evaluation.utils import async_wrapper, get_prompt_context, get_gt_answer


def main(
    base: str = "llama3-70b-instruct",
    vllm_hostname: str = "",
    temperature: float = 0.1,
    dataset: str = "nyt",
    n_questions: int = 1000,
) -> None:
    cfg = get_model_config(base)
    opening_msg = Message(Role.SYSTEM, cfg.system_message)
    llm = LLM(base, opening_message=opening_msg)          # not loading weights here

    base_url = f"http://{vllm_hostname}:8000/v1"
    api_key = "token-abc123"
    vllm_client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    extra_body = {
        "top_k": 50,
        "include_stop_str_in_output": True,
        "skip_special_tokens": False,
    }
    if "llama" in base.lower():
        extra_body["stop_token_ids"] = [128009]

    ds = load_squadshifts(dataset)
    out_file = Path(f"./datasets/{dataset}_filtered.csv")
    out_file.parent.mkdir(exist_ok=True, parents=True)
    if out_file.exists():
        print(f"{out_file} already exists – aborting.")
        return

    prompts = []
    original_questions = []
    for i, item in enumerate(ds):
        if i >= n_questions > 0:
            break
        question = item['question']
        context = get_prompt_context(item, "squadshifts")
        gt_answer = str(get_gt_answer(item, "squadshifts"))
        original_questions.append(question)

        prompt = f"""Here is a piece of text:
{context}

Here is a question related to the text:
{question}

Here is a list of valid ground-truth answers:
{gt_answer}

Please re-write the question such that it can be fully understood and it makes sense without access to the text. Output the new question inside <question> xml tags, like this:

<question>Rewritten question</question>"""

        messages = [Message(Role.USER, prompt)]
        messages = merge_messages(messages)
        prompt = llm.messages_to_prompt(messages)
        prompts.append(prompt)

    print(f"Number of prompts: {len(prompts)}", flush=True)
    start_time = time.time()
    answers = asyncio.run(async_wrapper(vllm_client, cfg.vllm_model, prompts, extra_body, temperature, max_tokens=500))
    end_time = time.time()
    print(f"Generation time: {end_time - start_time:.4f} s", flush=True)
    assert len(prompts) == len(answers)
    
    with open(out_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter=";")
        for orig_q, answer in zip(original_questions, answers):
            # Regex pattern to match content between <question> and </question>
            pattern = r'<question>(.*?)</question>'

            # Find all matches
            matches = re.findall(pattern, answer)
            if len(matches) == 1 and len(matches[0]):
                answer = matches[0]
            else:
                print(f"Answer {answer} invalid. Resorting to original")
                answer = orig_q 

            writer.writerow([answer.replace("\n", " ")])
    print(f"Saved rewritten questions in {out_file}")


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)

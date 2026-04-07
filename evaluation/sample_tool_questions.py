"""
Generate user queries for financial tool-calling + NLP distillation.

Outputs train and eval CSVs in the same format as sample_questions.py
so the rest of the pipeline works unchanged.

Usage:
    python3 evaluation/sample_tool_questions.py \
        --system_prompt_path context/financial_system_prompt.txt \
        --vllm_hostname localhost
"""
import asyncio
import csv
import re
import time
from pathlib import Path
from typing import Dict, List

from openai import AsyncOpenAI

from core.file_naming import generate_question_path
from core.llm import LLM
from core.messages import Message, Role
from core.model_configs import get_model_config
from core.utils import generate_extra_body
from evaluation.utils import async_wrapper

MAX_FAILURES = 20

TOOL_PROMPT_TEMPLATE = """You are given a system prompt that describes an AI assistant with access to specific tools for a financial app.

<system_prompt>
{system_prompt}
</system_prompt>

Generate 5 diverse, realistic user messages that would require the assistant to call one of its tools. The messages should:
- Be natural and conversational (how a real customer would talk)
- Cover different tools and argument combinations
- Vary in complexity (simple lookups vs multi-filter requests)
- Never reference tool names, schemas, or JSON directly

Format each as:
<question>user message here</question>"""

NLP_PROMPT_TEMPLATE = """You are given a system prompt that describes an AI assistant for a financial app.

<system_prompt>
{system_prompt}
</system_prompt>

Generate 5 diverse, realistic user messages that the assistant should answer with plain conversational advice — NOT by calling any tool. These are questions about budgeting strategy, financial planning, spending habits, saving tips, or general money guidance. The messages should:
- Be natural and conversational
- Cover different financial topics (budgeting, saving, debt, spending priorities, cash flow)
- Not require looking up any account data — just general or personalised advice
- Never reference tool names or schemas

Format each as:
<question>user message here</question>"""


def _extract_questions(text: str) -> List[str]:
    return re.findall(r'<question>(.*?)</question>', text, re.DOTALL)


def _build_prompt(system_prompt: str, llm: LLM, nlp: bool) -> str:
    template = NLP_PROMPT_TEMPLATE if nlp else TOOL_PROMPT_TEMPLATE
    content = template.format(system_prompt=system_prompt)
    messages = [Message(Role.USER, content)]
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
        if extracted and len(extracted) >= 2:
            qs.extend(extracted)
            done += 1
        else:
            failures += 1
            if failures >= MAX_FAILURES:
                print("[WARN] too many failed generations – skipping")
                break
    return qs


def _generate_and_write(label, output_file, t_batches, n_batches, needed_calls,
                        system_prompt, llm, client, cfg, extra_body, temperature, max_tokens):
    if output_file.exists():
        print(f"{output_file} already exists — skipping {label}.", flush=True)
        return

    prompts = (
        [_build_prompt(system_prompt, llm, nlp=False) for _ in range(t_batches)]
        + [_build_prompt(system_prompt, llm, nlp=True) for _ in range(n_batches)]
    )

    print(f"Generating {label} questions: {t_batches} tool + {n_batches} NLP batches", flush=True)

    start_time = time.time()
    questions_per_batch = asyncio.run(
        async_wrapper(
            client, cfg.vllm_model, prompts, extra_body,
            temperature, max_tokens,
            custom_fnc=_sample_questions,
            custom_fnc_extra_kwargs={"needed_calls": needed_calls},
        )
    )
    print(f"Generation time: {time.time() - start_time:.2f}s", flush=True)

    seen = set()
    unique = []
    for q_list in questions_per_batch:
        for q in q_list:
            q_clean = q.strip()
            if q_clean and q_clean not in seen:
                seen.add(q_clean)
                unique.append(q_clean)

    if not unique:
        print(f"WARNING: No {label} questions generated — not writing empty file.", flush=True)
        return

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        for q in unique:
            writer.writerow([q.replace("\n", " "), system_prompt.replace("\n", " ")])

    print(f"{len(unique)} unique {label} questions written to {output_file}", flush=True)


def main(
    system_prompt_path: str = "",
    base: str = "llama3-8b-instruct",
    dataset_family: str = "financial",
    dataset: str = "tool_calling",
    train_questions: int = 200,
    max_items: int = 1,
    max_tokens: int = 1024,
    temperature: float = 1.5,
    vllm_hostname: str = "",
    tool_batches: int = 10,
    nlp_batches: int = 5,
    eval_tool_batches: int = 3,
    eval_nlp_batches: int = 2,
) -> None:
    system_prompt = Path(system_prompt_path).read_text().strip()
    cfg = get_model_config(base)
    llm = LLM(base, opening_message=Message(Role.SYSTEM, cfg.system_message))

    client = AsyncOpenAI(base_url=f"http://{vllm_hostname}:8000/v1", api_key="token-abc123")
    extra_body = generate_extra_body(base)

    train_file = Path(
        generate_question_path(
            dataset_family=dataset_family, dataset=dataset, model=base,
            questions=train_questions, temperature=temperature, max_items=max_items,
        )
    )
    train_file.parent.mkdir(parents=True, exist_ok=True)
    eval_file = train_file.parent / f"eval_{train_file.name}"

    common = dict(system_prompt=system_prompt, llm=llm, client=client, cfg=cfg,
                  extra_body=extra_body, temperature=temperature, max_tokens=max_tokens)

    _generate_and_write(
        "train", train_file, tool_batches, nlp_batches,
        max(1, train_questions // (5 * (tool_batches + nlp_batches))),
        **common,
    )
    _generate_and_write(
        "eval", eval_file, eval_tool_batches, eval_nlp_batches, 1,
        **common,
    )


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)

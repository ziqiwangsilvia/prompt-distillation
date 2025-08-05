import csv
import os
import string
import warnings
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset
from nltk.corpus import stopwords
from openai import OpenAI
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from core.llm import LLM
from core.messages import Message, Role
from core.model_configs import get_model_config
from evaluation.utils import (
    get_rag_context,
    get_prompt_context,
    get_gt_answer,
)

STOP_WORDS = set(stopwords.words("english"))


def _clean_tokenize(paragraph: str) -> List[str]:
    """Lower‑case, strip punctuation, drop stop‑words."""
    translator = str.maketrans("", "", string.punctuation)
    toks = paragraph.lower().translate(translator).split()
    return [t for t in toks if t not in STOP_WORDS]


def _cosine_similarity(
    a: np.ndarray, B: np.ndarray
) -> np.ndarray:  # (emb,) vs (N,emb)
    a_norm = np.linalg.norm(a)
    B_norms = np.linalg.norm(B, axis=1)
    return (B @ a) / (B_norms * a_norm + 1e-9)


def _sample_answer(
    llm: LLM,
    prompt: str,
    temperature: float,
    max_total_tokens: int,
    max_new_tokens: int,
) -> str:
    """Generate an answer, automatically shrinking the budget if needed."""
    n_prompt = llm.tokenize(prompt).shape[1]
    max_gen = min(max_new_tokens, max_total_tokens - n_prompt)
    if max_gen <= 0:
        raise ValueError(
            f"Prompt too long ({n_prompt} tokens); budget={max_total_tokens}"
        )
    if max_gen <= 10:
        warnings.warn(f"Only {max_gen} tokens budget for generation.", stacklevel=2)

    answer, _ = llm.call(
        [Message(Role.USER, prompt)],
        temperature=temperature,
        max_new_tokens=max_gen,
    )
    return answer.replace("&lt;", "<").replace("&gt;", ">")


def _get_openai_emb(text: str) -> np.ndarray:
    client = OpenAI()
    resp = client.embeddings.create(input=text, model="text-embedding-3-small")
    return np.array(resp.data[0].embedding)


def main(
    *,
    base: str = "llama3-8b-instruct",
    adapter_id: str = "",
    temperature: float = 0.0,
    dataset_family: str = "squadshifts",
    dataset: str = "nyt",
    output_filename: str = "output.csv",
    n_questions: int = 1000,
    bm25: bool = False,
    openai_rag: bool = False,
    n_questions_rag: int = 1000,
    n_documents_rag: int = 7,
    oracle: bool = False,
    rewritten_questions: bool = True,
) -> None:

    system_msg = get_model_config(base).system_message
    opening_message = Message(Role.SYSTEM, system_msg)

    assert not (bm25 and oracle), "`bm25` and `oracle` are mutually exclusive"

    output_dir_name = f"{dataset_family}_{dataset}"
    if dataset_family == "squadshifts":
        hf_split = load_dataset("squadshifts", dataset, trust_remote_code=True)["test"]
    elif dataset_family == "hotpotqa":
        hf_split = load_dataset("hotpotqa/hotpot_qa", dataset, trust_remote_code=True)[
            "validation"
        ]
        rewritten_questions = False
        warnings.warn("Changing rewritten_questions to False")
    else:
        raise ValueError(f"Unsupported dataset family: {dataset_family}")

    # rewritten questions file
    if rewritten_questions:
        assert n_questions <= 1000
        q_path = Path("datasets") / f"{dataset}_filtered.csv"
        rewritten_qs = q_path.read_text(encoding="utf-8").splitlines()
        all_questions = rewritten_qs[: n_questions]  # truncate
    elif openai_rag:
        all_questions = []
        for i, item in enumerate(hf_split):
            if i >= n_questions:
                break
        all_questions.append(item['question'])

    if adapter_id:
        llm = LLM.from_adapter(adapter_id, opening_message=opening_message)
    else:
        llm = LLM(base, opening_message=opening_message)
    llm.load_model()

    if bm25 or openai_rag:
        tokenizer = llm.tokenizer
        contexts = []
        tokenized_contexts = []
        seen = set()
        prev_block = None

        for i, item in enumerate(hf_split):
            if 0 < n_questions_rag <= i:       # honour user limit (0 == unlimited)
                break
            block = tuple(get_rag_context(item, dataset_family))
            if block == prev_block:
                continue
            prev_block = block
            for para in block:
                if para in seen:
                    continue
                seen.add(para)
                contexts.append(para)

                if dataset_family == "squadshifts":
                    tokenized_contexts.append(item["context"].split())
                else:
                    tokenized_contexts.append(_clean_tokenize(para))

        print(f"# unique paragraphs: {len(seen)}", flush=True)
        print(
            f"# total tokens: {sum(len(tok) for tok in tokenized_contexts)}",
            flush=True,
        )

        if bm25:
            bm25_retriever = BM25Okapi(tokenized_contexts)
            if "bm25" not in output_filename:
                warnings.warn(f"WARNING: bm25 not in output_filename — renaming.", stacklevel=2)
                output_filename = "output_bm25.csv"

        elif openai_rag:
            embeddings_path = f"./datasets/{dataset}_embeddings.csv"
 
            # Paragraph-level embeddings
            if os.path.exists(embeddings_path):
                document_embeddings = pd.read_csv(embeddings_path, header=None).values
                print(f"Loaded paragraph embeddings from {embeddings_path}", flush=True)
            else:
                print(f"Embedding {len(contexts)} unique paragraphs", flush=True)
                document_embeddings = np.array(
                    [_get_openai_emb(p) for p in tqdm(contexts)]
                )
                pd.DataFrame(document_embeddings).to_csv(
                    embeddings_path, index=False, header=False
                )
                print(f"Saved embeddings in {embeddings_path}", flush=True)
 
            # Question-level embeddings
            questions_path = f"./datasets/{dataset}_questions_embeddings.csv"
            if os.path.exists(questions_path):
                questions_embeddings = pd.read_csv(questions_path, header=None).values
                print(f"Loaded question embeddings from {questions_path}", flush=True)
            else:
                print(f"Embedding {len(all_questions)} questions", flush=True)
                questions_embeddings = np.array(
                    [_get_openai_emb(q) for q in tqdm(all_questions)]
                )
                pd.DataFrame(questions_embeddings).to_csv(
                    questions_path, index=False, header=False
                )
                print(f"Saved question embeddings in {questions_path}", flush=True)
    elif oracle and 'oracle' not in output_filename:
        warnings.warn(f"WARNING: oracle not in output_filename. Renaming it", stacklevel=2)
        output_filename = "output_oracle.csv"

    if adapter_id:
        marker = "checkpoints/huggingface/"
        adapter_name = adapter_id.split(marker, 1)[1] if marker in adapter_id else adapter_id.rsplit("/", 1)[-1]
        run_dir = Path("outputs") / output_dir_name / adapter_name
    else:
        run_dir = Path("outputs") / output_dir_name / base
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / output_filename
    if out_path.exists():
        warn(f"{out_path} already exists; aborting.")
        return

    print(f"Writing answers to output_file {out_path}", flush=True)

    questions = []
    gt_answers = []
    answers = []
    prompts = []
    for i, item in enumerate(hf_split):
        if i >= n_questions > 0:
            break

        if rewritten_questions:
            question = all_questions[i]
        else:
            question = item['question']

        if openai_rag or bm25:
            if openai_rag:
                embedding = questions_embeddings[i]
                similarities = _cosine_similarity(embedding, document_embeddings)
                top_n_indices = similarities.argsort()[-n_documents_rag:][::-1]
                top_n_documents = [contexts[i] for i in top_n_indices]
                context = "\n\n".join(top_n_documents)
            elif bm25:
                context_tokens = _clean_tokenize(question)
                scores = bm25_retriever.get_scores(context_tokens)
                top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_documents_rag]
                top_contexts = [contexts[i] for i in top_indices]
                context = "\n\n".join(top_contexts)
            prompt = f"{context}\n\nQuestion: {question}"
        elif oracle:
            context = get_prompt_context(item)
            prompt = f"{context}\n\nQuestion: {question}"
        else:
            prompt = question

        gt_answer = str(get_gt_answer(item, dataset_family))

        questions.append(question)
        gt_answers.append(gt_answer)

        answer = _sample_answer(
            llm,
            prompt,
            temperature=temperature,
            max_total_tokens=6128,
            max_new_tokens=500,
        )
        answers.append(answer)
        prompts.append(prompt)

    with open(out_path, "w", newline="", encoding="utf-8") as file:
        print(f"Writing answers to {out_path}", flush=True)
        writer = csv.writer(file, delimiter=";", quotechar='"', quoting=csv.QUOTE_MINIMAL, escapechar="\\")
        for (gt_answer, answer, question) in zip(gt_answers, answers, questions):
            writer.writerow([question.replace('\n', ''), gt_answer.replace('\n', ''), answer.replace('\n', ' ')])
            file.flush()
    print("Writing finished", flush=True)


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(main)

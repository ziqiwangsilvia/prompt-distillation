import aiofiles
import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional


async def _generate_async(
    client: Any,
    model: str,
    prompt: str,
    extra_body: dict[str, Any],
    temperature: float,
    max_tokens: int,
    index: int,
    total: int,
) -> str:
    """
    Fire a single completion request to the vLLM server and return its text.
    """
    resp = await client.completions.create(
        model=model,
        prompt=prompt,
        stream=False,
        extra_body=extra_body,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    completion: str = resp.choices[0].text.strip()
    logging.info("Generated output %d / %d", index + 1, total)
    return completion


async def async_wrapper(
    vllm_client: Any,
    model_id: str,
    prompts: List[str],
    extra_body: dict[str, Any],
    temperature: float,
    max_tokens: int,
    *,
    batch_size: int = 50,
    temp_file: Optional[str] = None,
    custom_fnc: Optional[Callable[..., Awaitable[str]]] = None,
    custom_fnc_extra_kwargs: Optional[dict[str, Any]] = None,
) -> List[str]:
    """
    Generate completions for prompts using the provided vLLM client.
    """
    custom_fnc_extra_kwargs = custom_fnc_extra_kwargs or {}
    fnc = custom_fnc or _generate_async

    total = len(prompts)
    logging.info("Starting async generation for %d prompts.", total)

    outputs: List[str] = []
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_prompts = prompts[start:end]

        logging.info(
            "Batch %d / %d (%d–%d)",
            start // batch_size + 1,
            (total + batch_size - 1) // batch_size,
            start,
            end - 1,
        )

        tasks = [
            fnc(
                vllm_client,
                model_id,
                prompt,
                extra_body,
                temperature,
                max_tokens,
                index=idx,
                total=total,
                **custom_fnc_extra_kwargs,
            )
            for idx, prompt in enumerate(batch_prompts, start=start)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for prompt, result in zip(batch_prompts, results, strict=True):
            if isinstance(result, Exception):
                logging.error("Prompt failed: %s – %s", prompt[:50], result)
                continue

            outputs.append(result)
            # Append results to temp_file asynchronously if provided
            if temp_file:
                async with aiofiles.open(temp_file, "a") as f:
                    await f.write(
                        prompt.replace("\n", "").replace(";", ",")
                        + ";"
                        + result.replace("\n", " ").replace(";", ",")
                        + "\n"
                    )

    logging.info("Async generation finished (%d outputs).", len(outputs))
    return outputs


def get_rag_context(
    item: Dict[str, Any],
    dataset_family: str,
) -> List[str]:
    """
    Return the list of text blocks used as documents for rag.
    """
    if dataset_family == "squadshifts":
        return [item['context']]
    elif dataset_family == "hotpotqa":
        sentences = item['context']['sentences']
        merged_sentences = ["".join(s) for s in sentences]
        return merged_sentences
    raise NotImplementedError(f"Unknown dataset family '{dataset_family}'.")


def get_prompt_context(
    item: Dict[str, Any],
    dataset_family: str,
) -> str:
    """
    Return the full context string that is inserted into the user prompt.
    """
    if dataset_family == "squadshifts":
        return item['context']
    elif dataset_family == "hotpotqa":
        sentences = item['context']['sentences']
        merged_sentences = ["".join(s) for s in sentences]
        return "\n\n".join(merged_sentences)
    raise NotImplementedError(f"Unknown dataset family '{dataset_family}'.")


def get_gt_answer(
    item: Dict[str, Any],
    dataset_family: str,
) -> List[str] | str:
    """
    Extract the ground‑truth answer(s) for *item*.
    """
    if dataset_family == "squadshifts":
        return item['answers']['text']
    elif dataset_family == "hotpotqa":
        return item['answer']
    raise NotImplementedError(f"Unknown dataset family '{dataset_family}'.")

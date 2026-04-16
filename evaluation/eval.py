"""
Run inference on eval data using a trained checkpoint and save results.

Usage:
    python3 evaluation/eval.py
    python3 evaluation/eval.py --vllm_host localhost
    python3 evaluation/eval.py --run_name my_run --pipeline_config config/pipeline.yaml
"""
import asyncio
import json
import re
from pathlib import Path

from data.paths import BASE_PATH
from models.llm import LLM
from models.messages import Message, Role
from training.utils import read_exercises, extract_question


def run_inference(
    run_name: str = "",
    pipeline_config: str = "config/pipeline.yaml",
    vllm_host: str = "",
    temperature: float = 0.1,
    max_new_tokens: int = 500,
    metrics_only: bool = False,
) -> None:
    import yaml
    with open(pipeline_config) as f:
        cfg = yaml.safe_load(f)

    dataset_family = cfg["dataset"]["family"]
    dataset = cfg["dataset"]["name"]
    max_items = cfg["questions"]["max_eval_items"]
    exam_model = cfg["models"]["teacher"]
    student_base = cfg["models"]["student"]
    variant = "default"
    if not run_name:
        run_name = cfg.get("project", {}).get("run_name", cfg.get("training", {}).get("run_name", ""))

    # Load eval exercises
    custom_val = cfg.get("dataset", {}).get("custom_val_data", "")
    from models.configs import create_model_flags, get_model_config
    from data.naming import generate_augmented_filename, generate_exam_name
    if custom_val:
        eval_path = Path(custom_val)
    else:

        base_exam_id = generate_exam_name(dataset_family, dataset, variant, max_items)
        exam_flags = create_model_flags(exam_model)
        eval_file = generate_augmented_filename(base_exam_id, temperature=0.25, model_flags=exam_flags)
        eval_path = BASE_PATH / "output" / "teacher_answers" / eval_file

    if not eval_path.exists():
        raise FileNotFoundError(f"Eval data not found: {eval_path}")

    exercises = read_exercises(eval_path)
    print(f"Loaded {len(exercises)} eval exercises")

    output_dir = BASE_PATH / "output" / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{run_name}_{dataset_family}_{dataset}.jsonl"

    if metrics_only:
        if not output_file.exists():
            raise FileNotFoundError(f"No inference results found: {output_file}. Run inference first.")
        with open(output_file) as f:
            results = [json.loads(l) for l in f]
        print(f"Loaded {len(results)} results from {output_file}")
    else:
        # Build prompts
        student_cfg = get_model_config(student_base)
        llm = LLM(student_base, opening_message=Message(Role.SYSTEM, student_cfg.system_message))

        tools = None
        if cfg.get("training", {}).get("use_tool_token", False):
            tools_path = cfg.get("project", {}).get("tools_schema_path", "")
            if tools_path:
                with open(tools_path) as tf:
                    tools = json.load(tf)
                print(f"Loaded {len(tools)} tool definitions from {tools_path}")

        questions = []
        refs = []
        prompts = []
        for ex in exercises:
            student_content = extract_question(ex)
            questions.append(student_content)
            refs.append(ex.answer_choices[0].content if ex.answer_choices else None)
            msgs = [Message(Role.USER, student_content)]
            prompts.append(llm.messages_to_prompt(msgs, tools=tools))

        # Run inference
        if vllm_host:
            outputs = _vllm_inference(vllm_host, student_cfg.vllm_model, prompts, temperature, max_new_tokens)
        else:
            checkpoint_dir = BASE_PATH / "output" / "checkpoints" / run_name
            if not checkpoint_dir.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")
            print(f"Loading model from {checkpoint_dir}...")
            llm = LLM.from_adapter(str(checkpoint_dir))
            llm.load_model()
            outputs = _local_inference(llm, prompts, temperature, max_new_tokens)

        # Build results
        results = []
        for i, (q, ref, (output, truncated, has_tool_token)) in enumerate(zip(questions, refs, outputs)):
            results.append({
                "index": i,
                "question": q,
                "prediction": output,
                "truncated": truncated,
                "has_tool_token": has_tool_token,
                "reference": ref,
            })
            tag = " [python_tag]" if has_tool_token else ""
            print(f"[{i+1}/{len(questions)}]{tag} Pred: {output[:80]}...")

        with open(output_file, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

    # Metrics
    import sys
    sys.path.insert(0, "/root/tool-eval")
    from evaluator.tool_calling.tool_calling_metrics import (
        get_when2call, get_tool_pickup_and_hallucination,
        get_variable_parsing_and_hallucination, get_exact_match,
    )

    AVAILABLE_TOOLS = ["show_pie_chart", "show_stacked_bar_chart", "show_line_chart"]
    TOOL_SCHEMAS = {
        "show_pie_chart": {"properties": {"data_type": {}, "time_range": {}, "group_by": {}, "categories": {}, "payees": {}, "limit": {}}},
        "show_stacked_bar_chart": {"properties": {"data_type": {}, "time_range": {}, "group_by": {}, "categories": {}, "payees": {}, "limit": {}}},
        "show_line_chart": {"properties": {"chart_type": {}, "time_range": {}}},
    }

    def _to_metric_format(text):
        text = text.strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "name" in parsed:
                return {
                    "type": "tool",
                    "tools": [{"name": parsed["name"], "arguments": parsed.get("parameters", parsed.get("arguments", {}))}],
                }
        except (json.JSONDecodeError, ValueError):
            pass
        return {"type": "text"}

    GUARD_PHRASE = "sorry, i cannot help with this"

    n = len(results)
    when2call_scores = []
    pickup_scores = []
    hallucination_scores = []
    additional_scores = []
    var_pickup_scores = []
    var_correct_scores = []
    var_hallucination_scores = []
    var_additional_scores = []
    exact_scores = []
    guard_total = 0
    guard_correct = 0
    guard_pred_total = 0

    for r in results:
        gt = _to_metric_format(r["reference"] or "")
        pred = _to_metric_format(r["prediction"])

        ref_text = (r["reference"] or "").strip().lower()
        pred_text = r["prediction"].strip().lower()
        ref_is_guard = GUARD_PHRASE in ref_text

        if ref_is_guard:
            guard_total += 1
            if GUARD_PHRASE in pred_text:
                guard_correct += 1
        if GUARD_PHRASE in pred_text:
            guard_pred_total += 1

        when2call = get_when2call(gt, pred)
        when2call_scores.append(when2call)

        if when2call:
            pickup, halluc, additional = get_tool_pickup_and_hallucination(gt, pred, AVAILABLE_TOOLS)
            if pickup is not None:
                pickup_scores.append(pickup)
                hallucination_scores.append(halluc)
                if pickup:
                    additional_scores.append(additional)

                if pickup:
                    vp, vc, vh, va = get_variable_parsing_and_hallucination(gt, pred, TOOL_SCHEMAS)
                    if vp is not None:
                        var_pickup_scores.append(vp)
                        var_correct_scores.append(vc)
                        var_hallucination_scores.append(vh)
                        var_additional_scores.append(va)

        em = get_exact_match(gt, pred)
        if em is not None:
            exact_scores.append(em)

    def _avg(scores):
        return sum(scores) / len(scores) * 100 if scores else 0

    tool_total = len(exact_scores)
    print(f"\n{'='*50}")
    print(f"METRICS ({n} examples, {tool_total} tool calls, {n - tool_total} text, {guard_total} guarded)")
    print(f"{'='*50}")
    print(f"  When2Call:              {_avg(when2call_scores):.1f}%")
    if tool_total:
        print(f"  Tool Pickup Rate:      {_avg(pickup_scores):.1f}%")
        print(f"  Tool Hallucination:    {_avg(hallucination_scores):.1f}%")
        print(f"  Tool Additional:       {_avg(additional_scores):.1f}%")
        print(f"  Var Pickup Rate:       {_avg(var_pickup_scores):.1f}%")
        print(f"  Var Correct Rate:      {_avg(var_correct_scores):.1f}%")
        print(f"  Var Hallucination:     {_avg(var_hallucination_scores):.1f}%")
        print(f"  Var Additional:        {_avg(var_additional_scores):.1f}%")
        print(f"  Exact Match:           {_avg(exact_scores):.1f}%")
    if guard_total:
        guard_recall = guard_correct / guard_total * 100
        guard_precision = guard_correct / guard_pred_total * 100 if guard_pred_total else 0
        print(f"  Guard Precision:       {guard_precision:.1f}% ({guard_correct}/{guard_pred_total})")
        print(f"  Guard Recall:          {guard_recall:.1f}% ({guard_correct}/{guard_total})")
    print(f"\nSaved {n} results to {output_file}")


def _vllm_inference(host, model, prompts, temperature, max_new_tokens):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(base_url=f"http://{host}:8000/v1", api_key="token-abc123")

    async def run():
        tasks = [
            client.completions.create(
                model="student",
                prompt=p,
                temperature=temperature,
                max_tokens=max_new_tokens,
                extra_body={"skip_special_tokens": False},
            )
            for p in prompts
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    print(f"Running async vLLM inference on {len(prompts)} prompts...")
    responses = asyncio.run(run())
    outputs = []
    for resp in responses:
        if isinstance(resp, Exception):
            outputs.append(("ERROR: " + str(resp), True, False))
        else:
            text = resp.choices[0].text.strip()
            outputs.append((text, resp.choices[0].finish_reason != "stop", "<|python_tag|>" in resp.choices[0].text))
    return outputs


def _local_inference(llm, prompts, temperature, max_new_tokens, batch_size=8):
    import torch

    if llm.tokenizer.pad_token_id is None:
        llm.tokenizer.pad_token_id = llm.tokenizer.eos_token_id
    llm.tokenizer.padding_side = "left"

    outputs = []
    terminators = llm.get_terminators()
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        encoded = llm.tokenizer(batch, return_tensors="pt", padding=True).to("cuda")
        tokens = llm.model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            eos_token_id=terminators,
            repetition_penalty=1.1,
        )
        for j, output_ids in enumerate(tokens):
            prompt_len = encoded["input_ids"].shape[1]
            gen_ids = output_ids[prompt_len:]
            truncated = bool(gen_ids[-1] not in terminators)
            python_tag_id = llm.tokenizer.convert_tokens_to_ids("<|python_tag|>")
            has_tool_token = python_tag_id in gen_ids.tolist()
            text = llm.tokenizer.decode(gen_ids, skip_special_tokens=True)
            outputs.append((text.strip(), truncated, has_tool_token))
        print(f"  {min(i+batch_size, len(prompts))}/{len(prompts)} done")
    return outputs


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(run_inference)

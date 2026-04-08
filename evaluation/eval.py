"""
Run inference on eval data using a trained checkpoint and save results.

Usage:
    python3 evaluation/eval.py --run_name pd_fin_llm
    python3 evaluation/eval.py --run_name pd_fin_llm --dataset_family financial --dataset tool_calling
"""
import json
import os
from pathlib import Path
from typing import List

from data.paths import BASE_PATH, TIPS_START, TIPS_END
from models.llm import LLM
from models.messages import Message, Role
from training.utils import read_exercises


def run_inference(
    run_name: str = "",
    dataset_family: str = "financial",
    dataset: str = "tool_calling",
    variant: str = "default",
    max_items: int = 20,
    temperature: float = 0.1,
    max_new_tokens: int = 500,
    exam_model: str = "qwen2.5-7b-instruct",
) -> None:
    checkpoint_dir = BASE_PATH / "output" / "checkpoints" / run_name
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")

    # Load model with adapter
    print(f"Loading model from {checkpoint_dir}...")
    llm = LLM.from_adapter(str(checkpoint_dir))
    llm.load_model()

    # Load eval exercises
    from models.configs import create_model_flags
    from data.naming import generate_augmented_filename, generate_exam_name

    base_exam_id = generate_exam_name(dataset_family, dataset, variant, max_items)
    exam_flags = create_model_flags(exam_model)
    eval_file = generate_augmented_filename(base_exam_id, temperature=0.25, model_flags=exam_flags)
    eval_path = BASE_PATH / "output" / "teacher_answers" / eval_file

    if not eval_path.exists():
        raise FileNotFoundError(f"Eval data not found: {eval_path}")

    exercises = read_exercises(eval_path)
    print(f"Loaded {len(exercises)} eval exercises")

    # Run inference
    results = []
    for i, ex in enumerate(exercises):
        # Strip TIPS to get student prompt (no system prompt material)
        content = ex.messages[-1].content
        # Remove TIPS tags and their content
        import re
        student_content = re.sub(
            f'{re.escape(TIPS_START)}.*?{re.escape(TIPS_END)}',
            '', content, flags=re.DOTALL
        ).strip()

        messages = [Message(Role.USER, student_content)]
        output, truncated = llm.call(
            messages, temperature=temperature, max_new_tokens=max_new_tokens,
        )

        # Get reference answer if available
        ref = ex.answer_choices[0].content if ex.answer_choices else None

        result = {
            "index": i,
            "question": student_content,
            "prediction": output,
            "truncated": truncated,
            "reference": ref,
        }
        results.append(result)
        print(f"[{i+1}/{len(exercises)}] Q: {student_content[:60]}...")
        print(f"  Pred: {output[:80]}...")
        print()

    # Save results
    output_dir = BASE_PATH / "output" / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{run_name}_{dataset_family}_{dataset}.jsonl"

    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Metrics
    n = len(results)
    exact = 0
    tool_correct = 0
    tool_total = 0
    type_match = 0

    for r in results:
        pred = r["prediction"].strip()
        ref = (r["reference"] or "").strip()

        # Classify response type
        pred_is_tool = pred.startswith("{") and '"name"' in pred
        ref_is_tool = ref.startswith("{") and '"name"' in ref

        if pred_is_tool == ref_is_tool:
            type_match += 1

        if pred == ref:
            exact += 1

        if ref_is_tool:
            tool_total += 1
            try:
                pred_json = json.loads(pred.split("\n")[0])
                ref_json = json.loads(ref)
                if pred_json.get("name") == ref_json.get("name"):
                    if pred_json.get("parameters") == ref_json.get("parameters"):
                        tool_correct += 1
            except (json.JSONDecodeError, IndexError):
                pass

    text_total = n - tool_total
    print(f"\n{'='*50}")
    print(f"METRICS ({n} examples)")
    print(f"{'='*50}")
    print(f"  Type match (tool vs text): {type_match}/{n} ({100*type_match/n:.1f}%)")
    print(f"  Exact match:               {exact}/{n} ({100*exact/n:.1f}%)")
    if tool_total:
        print(f"  Tool call accuracy:        {tool_correct}/{tool_total} ({100*tool_correct/tool_total:.1f}%)")
    print(f"  Tool calls: {tool_total}, Text: {text_total}")

    print(f"\nSaved {n} results to {output_file}")


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(run_inference)

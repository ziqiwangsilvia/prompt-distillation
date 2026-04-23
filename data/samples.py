"""Training sample builders for single-turn and multi-turn exercises.

Converts exercises into tokenized training samples with student (open/closed-book)
and teacher prompt/answer pairs. Tool calls are formatted in each model's native format.
"""
import json

from data.tool_call_format import format_tool_call, normalize_tool_call, to_native_format
from data.dataset import prepare_answer_tokens
from models.messages import Message, Role


def _format_assistant_content(msg_dict, model_family):
    """Format an assistant message's content for a specific model family.
    Handles tool_calls by converting to native format."""
    if "tool_calls" in msg_dict:
        tc = msg_dict["tool_calls"][0]
        func = tc if "name" in tc else tc.get("function", tc)
        name = func["name"]
        args = func.get("arguments", func.get("parameters", {}))
        if isinstance(args, str):
            args = json.loads(args)
        return format_tool_call({"name": name, "arguments": args}, model_family)
    return msg_dict.get("content", "")


def build_singleturn_samples(exercise, llm, teacher_llm=None, tools=None,
                             use_tool_token=False, max_length=0, date_str=""):
    """Build training samples from a single-turn exercise.

    Each answer choice becomes a separate sample. Same output format as
    build_multiturn_samples so both dataset classes can consume it identically.
    """
    from training.utils import extract_material_and_question, extract_question, tokenize_teacher_student

    t_llm = teacher_llm or llm
    question = extract_question(exercise)
    material, _ = extract_material_and_question(exercise)
    student_closed_tokens, student_open_tokens, teacher_tokens = tokenize_teacher_student(
        material, question, llm, teacher_llm=teacher_llm, tools=tools,
        student_tools=tools if use_tool_token else None, date_str=date_str)

    samples = []
    for choice in exercise.answer_choices:
        student_content = choice.content
        teacher_content = to_native_format(choice.content, t_llm.model_family) if use_tool_token and teacher_llm else choice.content
        alignment_text = normalize_tool_call(choice.content) if use_tool_token and teacher_llm else choice.content
        answer_tokens = prepare_answer_tokens(llm, student_content, max_length, choice.truncated, use_tool_token=use_tool_token)
        teacher_answer_tokens = prepare_answer_tokens(t_llm, teacher_content, max_length, choice.truncated) if teacher_llm else answer_tokens
        samples.append({
            "student_prompt_tokens": student_closed_tokens,
            "student_open_prompt_tokens": student_open_tokens,
            "teacher_prompt_tokens": teacher_tokens,
            "answer_tokens": answer_tokens,
            "teacher_answer_tokens": teacher_answer_tokens,
            "teacher_answer": alignment_text,
            "student_answer_text": student_content,
            "teacher_answer_text": teacher_content,
        })
    return samples


def build_multiturn_samples(exercise, llm, teacher_llm=None, tools=None,
                            use_tool_token=False, max_length=0, date_str=""):
    """Build training samples from a multi-turn exercise.

    For each assistant turn, creates a sample where:
    - prompt = all messages before this assistant turn
    - answer = this assistant turn's content (in each model's native format)

    Returns list of sample dicts matching StudentTeacherDataset format.
    """
    t_llm = teacher_llm or llm
    raw_msgs = [m.to_dict() if isinstance(m, Message) else m for m in exercise.messages]

    samples = []
    for turn_idx, msg in enumerate(raw_msgs):
        if msg["role"] != "assistant":
            continue

        context = raw_msgs[:turn_idx]

        # Format this assistant turn per model
        student_content = _format_assistant_content(msg, llm.model_family)
        teacher_content = _format_assistant_content(msg, t_llm.model_family) if teacher_llm else student_content
        alignment_text = _format_assistant_content(msg, "llama")
        alignment_text = normalize_tool_call(alignment_text) if "tool_calls" in msg else alignment_text

        # Build context messages with tool calls in student format
        student_ctx = []
        for m in context:
            if m["role"] == "assistant" and "tool_calls" in m:
                content = _format_assistant_content(m, llm.model_family)
            else:
                content = m.get("content", "")
            student_ctx.append(Message(Role.from_value(m["role"]), content))

        # Build context messages with tool calls in teacher format
        teacher_ctx = []
        for m in context:
            if m["role"] == "assistant" and "tool_calls" in m:
                content = _format_assistant_content(m, t_llm.model_family)
            else:
                content = m.get("content", "")
            teacher_ctx.append(Message(Role.from_value(m["role"]), content))

        student_tools = tools if use_tool_token else None
        student_open_tokens = llm.tokenize(llm.messages_to_prompt(student_ctx, tools=student_tools))

        # Closed-book: replace system prompt with default, keep chat history
        student_closed_ctx = [m for m in student_ctx if m.role != Role.SYSTEM]
        student_closed_tokens = llm.tokenize(llm.messages_to_prompt(student_closed_ctx, tools=student_tools, date_string=date_str))

        teacher_tokens = t_llm.tokenize(t_llm.messages_to_prompt(teacher_ctx, tools=tools))

        is_tool = use_tool_token and "tool_calls" in msg
        answer_tokens = prepare_answer_tokens(llm, student_content, max_length, truncated=False, use_tool_token=is_tool)
        teacher_answer_tokens = prepare_answer_tokens(t_llm, teacher_content, max_length, truncated=False) if teacher_llm else answer_tokens

        samples.append({
            "student_prompt_tokens": student_closed_tokens,
            "student_open_prompt_tokens": student_open_tokens,
            "teacher_prompt_tokens": teacher_tokens,
            "answer_tokens": answer_tokens,
            "teacher_answer_tokens": teacher_answer_tokens,
            "teacher_answer": alignment_text,
            "student_answer_text": student_content,
            "teacher_answer_text": teacher_content,
        })

    return samples

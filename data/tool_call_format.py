"""
Adapter to convert tool-call formats between model families.

Supports: qwen <-> llama, and a generic JSON format as common intermediate.
"""
import json
import re
from typing import Optional


# --- Extraction ---

def extract_tool_call_qwen(text: str) -> Optional[dict]:
    """Extract tool call from Qwen format: <tool_call>{"name": ..., "arguments": ...}</tool_call>"""
    m = re.search(r'<tool_call>\s*(\{.*\})', text, re.DOTALL)
    if m:
        raw = m.group(1).replace('</tool_call>', '').strip()
        while '""' in raw:
            raw = raw.replace('""', '"')
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return None


def extract_tool_call_llama(text: str) -> Optional[dict]:
    """Extract tool call from Llama format: <|python_tag|>{"name": ..., "parameters": ...}"""
    m = re.search(r'<\|python_tag\|>\s*(\{.*\})', text, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(1))
            if "parameters" in parsed and "arguments" not in parsed:
                parsed["arguments"] = parsed.pop("parameters")
            return parsed
        except json.JSONDecodeError:
            return None
    return None


def extract_tool_call(text: str, source_family: str) -> Optional[dict]:
    """Extract tool call dict from model-specific format."""
    if source_family == "qwen":
        return extract_tool_call_qwen(text)
    elif source_family == "llama":
        return extract_tool_call_llama(text)
    else:
        raise ValueError(f"Unknown source family: {source_family}")


# --- Formatting ---

def format_tool_call_qwen(tool_call: dict) -> str:
    payload = {"name": tool_call["name"], "arguments": tool_call["arguments"]}
    return f'<tool_call>\n{json.dumps(payload)}\n</tool_call>'


def format_tool_call_llama(tool_call: dict) -> str:
    payload = {"name": tool_call["name"], "parameters": tool_call["arguments"]}
    return json.dumps(payload)


def format_tool_call(tool_call: dict, target_family: str) -> str:
    """Format tool call dict into model-specific format."""
    if target_family == "qwen":
        return format_tool_call_qwen(tool_call)
    elif target_family == "llama":
        return format_tool_call_llama(tool_call)
    else:
        raise ValueError(f"Unknown target family: {target_family}")


# --- Convert ---

def to_native_format(content: str, model_family: str) -> str:
    """Convert stored Llama-format tool call to a model's native format.
    Non-tool-call text is returned unchanged."""
    if not content.lstrip().startswith('{"name"'):
        return content
    if model_family == "qwen":
        try:
            parsed = json.loads(content)
            tc = {"name": parsed["name"], "arguments": parsed.get("parameters", parsed.get("arguments", {}))}
            return format_tool_call_qwen(tc)
        except (json.JSONDecodeError, KeyError):
            return content
    return content


def normalize_tool_call(content: str) -> str:
    """Normalize a tool call to common form for alignment: just name + args as sorted JSON.
    Non-tool-call text is returned unchanged."""
    if not content.lstrip().startswith('{"name"'):
        return content
    try:
        parsed = json.loads(content)
        args = parsed.get("parameters", parsed.get("arguments", {}))
        return json.dumps({"name": parsed["name"], "arguments": args}, sort_keys=True)
    except (json.JSONDecodeError, KeyError):
        return content


def convert_tool_call_format(text: str, source_family: str, target_family: str) -> str:
    """Convert tool-call text from source model format to target model format.
    
    If no tool call is detected (i.e., NLP response), returns text unchanged.
    Preserves any surrounding text (e.g., mixed text + tool call responses).
    """
    if source_family == target_family:
        return text

    if source_family == "qwen":
        m = re.search(r'<tool_call>\s*\{.*\}(\s*</tool_call>)?', text, re.DOTALL)
    elif source_family == "llama":
        m = re.search(r'<\|python_tag\|>\s*\{.*\}', text, re.DOTALL)
    else:
        raise ValueError(f"Unknown source family: {source_family}")

    if not m:
        return text

    tool_call = extract_tool_call(text, source_family)
    if tool_call is None:
        return text

    formatted = format_tool_call(tool_call, target_family)
    return text[:m.start()] + formatted + text[m.end():]

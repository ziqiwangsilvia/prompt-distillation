import json
from dataclasses import dataclass
import os
from typing import Optional, List

from models.messages import Message, Role


@dataclass
class Choice:
    """Represents a possible answer choice."""
    content: str
    truncated: bool = False

    def to_dict(self) -> dict:
        d = {"content": self.content}
        if self.truncated:
            d["truncated"] = True
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Choice":
        return cls(content=d["content"], truncated=d.get("truncated", False))


@dataclass
class ModelAnswer:
    """Represents a model's answer."""
    content: str


@dataclass
class GradingStr:
    """Represents grading string."""
    content: str


class ExerciseWithAnswers:
    """Represents an exercise, its possible answers, and associated information."""
    def __init__(
        self,
        messages: List[Message],
        answer_choices: Optional[List[Choice]] = None,
        lesson_id: Optional[str] = None,
        model_answer: Optional[ModelAnswer] = None,
        grading_str: Optional[GradingStr] = None,
    ):
        self.messages = messages
        self.answer_choices = answer_choices or []
        self.lesson_id = lesson_id
        self.model_answer = model_answer.content if isinstance(model_answer, ModelAnswer) else model_answer
        self.grading_str = grading_str.content if isinstance(grading_str, GradingStr) else grading_str

    @classmethod
    def from_dict(cls, d: dict, lesson_id: str) -> 'ExerciseWithAnswers':
        """Parse ExerciseWithAnswers from a dict."""
        messages = [Message.from_dict(m) for m in d["messages"]]
        answer_choices = [Choice.from_dict(c) for c in d.get("answer_choices", [])]
        model_answer = ModelAnswer(d["model_answer"]) if d.get("model_answer") else None
        grading_str = GradingStr(d["grading_str"]) if d.get("grading_str") else None
        return cls(messages, answer_choices, lesson_id, model_answer, grading_str)

    def to_dict(self) -> dict:
        """Serialize ExerciseWithAnswers to a dict."""
        d = {
            "messages": [m.to_dict() for m in self.messages],
            "answer_choices": [c.to_dict() for c in self.answer_choices],
        }
        if self.model_answer:
            d["model_answer"] = self.model_answer
        if self.grading_str:
            d["grading_str"] = self.grading_str
        return d

    def __str__(self):
        return f"ExerciseWithAnswers: messages={self.messages}, answer_choices={self.answer_choices}"

    def __repr__(self):
        return str(self)


def save_to_json(
    filepath: os.PathLike,
    exercises_with_answers: List[ExerciseWithAnswers]
) -> None:
    """Save a list of ExerciseWithAnswers objects as a JSON file."""
    data = {"exercises_with_answers": [ex.to_dict() for ex in exercises_with_answers]}
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved to {filepath}")

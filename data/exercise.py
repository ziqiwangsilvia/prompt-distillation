from dataclasses import dataclass
from typing import Optional, List

from models.messages import Message


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


class ExerciseWithAnswers:
    """Represents an exercise, its possible answers, and associated information."""
    def __init__(
        self,
        messages: List[Message],
        answer_choices: Optional[List[Choice]] = None,
        lesson_id: Optional[str] = None,
        model_answer: Optional[str] = None,
        grading_str: Optional[str] = None,
        metadata: Optional[dict] = None,
    ):
        self.messages = messages
        self.answer_choices = answer_choices or []
        self.lesson_id = lesson_id
        self.model_answer = model_answer
        self.grading_str = grading_str
        self.metadata = metadata or {}

    @classmethod
    def from_dict(cls, d: dict, lesson_id: str) -> 'ExerciseWithAnswers':
        """Parse ExerciseWithAnswers from a dict."""
        messages = [Message.from_dict(m) for m in d["messages"]]
        answer_choices = [Choice.from_dict(c) for c in d.get("answer_choices", [])]
        metadata = d.get("metadata", {})
        return cls(
            messages, answer_choices, lesson_id,
            model_answer=d.get("model_answer"),
            grading_str=d.get("grading_str"),
            metadata=metadata,
        )

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
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    def __str__(self):
        return f"ExerciseWithAnswers: messages={self.messages}, answer_choices={self.answer_choices}"

    def __repr__(self):
        return str(self)

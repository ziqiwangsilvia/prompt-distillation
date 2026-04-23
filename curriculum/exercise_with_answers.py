import json
import os
from dataclasses import dataclass
from typing import Optional, List

# Re-export core classes so existing curriculum imports still work
from data.exercise import ExerciseWithAnswers, Choice


@dataclass
class ModelAnswer:
    """Represents a model's answer."""
    content: str


@dataclass
class GradingStr:
    """Represents grading string."""
    content: str


def save_to_json(
    filepath: os.PathLike,
    exercises_with_answers: List[ExerciseWithAnswers]
) -> None:
    """Save a list of ExerciseWithAnswers objects as a JSON file."""
    data = {"exercises_with_answers": [ex.to_dict() for ex in exercises_with_answers]}
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved to {filepath}")

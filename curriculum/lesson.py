from collections import OrderedDict
import json
import os
from typing import Optional, List, Dict

from data.paths import BASE_PATH, DELIMITER
from models.utils import remove_empty


class Lesson:
    """Represents a single lesson, including exercises and material."""
    def __init__(self, lesson_dict: dict):
        self.id: str = lesson_dict["id"]
        self.response_format: Optional[str] = lesson_dict.get("response_format")
        self._material: str = lesson_dict.get("material", "")
        self.exercises: List["Exercise"] = [
            Exercise.from_dict(ex) for ex in lesson_dict.get("exercises", [])
        ]

    def __str__(self) -> str:
        txt = "Lesson:"
        if self._material:
            txt += f" material={self._material[:10]}..."
        txt += f" exercises={self.exercises}"
        return txt

    def __repr__(self) -> str:
        return str(self)

    def render_material(self) -> str:
        return self._material.strip() if self._material else ""

    def create_exercise_prompts(self, verbose: bool) -> List["Exercise"]:
        """Attach prompts to each exercise in this lesson."""
        my_print = print if verbose else lambda *x, **y: None
        tips = [self.render_material()]
        tips = remove_empty(tips)
        if tips:
            tips.extend([DELIMITER + "\n\n"])
        material_str = "\n\n".join(tips)

        for exercise in self.exercises:
            student_prompt = str(exercise)
            my_print("student_prompt:", student_prompt)
            teacher_prompt = material_str + student_prompt

            exercise.add_prompts(
                student_prompt=student_prompt,
                teacher_prompt=teacher_prompt,
                material=material_str,
            )

        return self.exercises

    def to_dict(self) -> dict:
        d = {"id": self.id, "material": self._material}
        if self.response_format:
            d["response_format"] = self.response_format
        d["exercises"] = [ex.to_dict() for ex in self.exercises]
        return d


def read_lessons(filepath: os.PathLike, error_if_not_found: bool = True) -> Dict[str, "Lesson"]:
    """Read and parse lessons from a JSON file into Lesson objects."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError as e:
        if error_if_not_found:
            raise e
        else:
            print(f"File {filepath} not found")
            return {}
    return OrderedDict(
        (d["id"], Lesson(d)) for d in data["lessons"]
    )


class Exercise:
    """Represents a single exercise within a lesson."""
    def __init__(
        self,
        exercise: str,
        model_answer: Optional[str] = None,
        grading_str: Optional[str] = None,
    ):
        self.exercise = exercise
        self.model_answer = model_answer
        self.grading_str = grading_str

        self.student_prompt: Optional[str] = None
        self.teacher_prompt: Optional[str] = None
        self.material: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> "Exercise":
        return cls(
            exercise=d["exercise"],
            model_answer=d.get("model_answer"),
            grading_str=d.get("grading_str"),
        )

    def to_dict(self) -> dict:
        d = {"exercise": self.exercise}
        if self.model_answer:
            d["model_answer"] = self.model_answer
        if self.grading_str:
            d["grading_str"] = self.grading_str
        return d

    def __str__(self) -> str:
        return "\n\n".join(remove_empty([self.exercise]))

    def __repr__(self) -> str:
        return f"Exercise(exercise={self.exercise}, model_answer={self.model_answer}, grading_str={self.grading_str})"

    def add_prompts(self, student_prompt: str, teacher_prompt: str, material: str) -> None:
        """Set prompts for the exercise."""
        self.student_prompt = student_prompt
        self.teacher_prompt = teacher_prompt
        self.material = material

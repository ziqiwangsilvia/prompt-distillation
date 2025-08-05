from collections import OrderedDict
import os
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape
from typing import Optional, List, Dict

from core import BASE_PATH
from core.utils import remove_empty
from curriculum import TIPS_START, TIPS_END, DELIMITER

XML_STYLE = True

def _get_element_text(
    lesson: ET.Element,
    tag: str,
    required: bool = False,
    multiple: bool = False
) -> str | List[str]:
    """Get the text from the first tag found in the lesson."""
    elements = lesson.findall(tag)
    if not elements:
        if required:
            raise ValueError(f"no <{tag}> entry found")
        return [] if multiple else ""
    if not multiple:
        assert len(elements) == 1, f"multiple <{tag}> entries found"
        return elements[0].text.strip()
    return [element.text.strip() for element in elements]


class Lesson:
    """Represents a single lesson, including exercises and material."""
    def __init__(self, lesson: ET.Element):
        self.id: str = lesson.get("id")
        self.response_format: Optional[str] = lesson.get("response_format")
        self._element = lesson
        self.material: Optional[ET.Element] = lesson.find("material")
        self.exercises: List["Exercise"] = [
            Exercise.from_xml(exercise)
            for exercise in lesson.findall('exercise')
        ]

    def __str__(self) -> str:
        txt = "Lesson:"
        if self.material is not None:
            txt += f" material={str(self.material)[:10]}..."
        txt += f" exercises={self.exercises}"
        return txt

    def __repr__(self) -> str:
        return str(self)

    def render_material(self) -> str:
        """Render material text, substituting in escaped examples if present."""
        if self.material is None:
            return ""
        return self.material.text.strip()

    def create_exercise_prompts(self, verbose: bool) -> List["Exercise"]:
        """Attach prompts to each exercise in this lesson,"""
        my_print = print if verbose else lambda *x, **y: None
        tips = [self.render_material()]
        tips = remove_empty(tips)
        if tips:
            tips.extend([DELIMITER + "\n\n"])
        tips_str = "\n\n".join(tips)
 
        if tips_str:
            teacher_prompt = tips_str
            teacher_prompt_with_tips_tags = f"{TIPS_START}{tips_str}{TIPS_END}"
        else:
            teacher_prompt = ""
            teacher_prompt_with_tips_tags = ""
 
        for exercise in self.exercises:
            student_prompt = str(exercise)
            my_print("student_prompt:", student_prompt)
            teacher_prompt += student_prompt
            teacher_prompt_with_tips_tags += student_prompt

            exercise.add_prompts(
                student_prompt=student_prompt,
                teacher_prompt=teacher_prompt,
                teacher_prompt_with_tips_tags=teacher_prompt_with_tips_tags,
            )
 
        return self.exercises


def read_lessons(filepath: os.PathLike, error_if_not_found: bool = True) -> Dict[str, Lesson]:
    """Read and parse lessons from an XML file into Lesson objects."""
    try:
        tree = ET.parse(filepath)
    except FileNotFoundError as e:
        if error_if_not_found:
            raise e
        else:
            print(f"File {filepath} not found")
            return {}
    root = tree.getroot()
    lessons = OrderedDict(
        (lesson.get("id"), Lesson(lesson))
        for lesson in root.findall('lesson')
    )
    return lessons


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
        
        # The following are added with add_prompts
        self.student_prompt: Optional[str] = None
        self.teacher_prompt: Optional[str] = None
        self.teacher_prompt_with_tips_tags: Optional[str] = None

    @classmethod
    def from_xml(cls, exercise_elem: ET.Element) -> "Exercise":
        """Create Exercise from an XML element."""
        text_content = [exercise_elem.text.strip()]
        for part in exercise_elem:
            if part.tail:
                text_content.append(part.tail.strip())
        exercise = " ".join(text_content).strip()
        model_answer_elem = exercise_elem.find('model_answer')
        model_answer = model_answer_elem.text.strip() if model_answer_elem is not None else None
        grading_str_elem = exercise_elem.find('grading_str')
        grading_str = grading_str_elem.text.strip() if grading_str_elem is not None else None
        return cls(exercise, model_answer, grading_str)

    def __str__(self) -> str:
        return "\n\n".join(remove_empty([self.exercise]))

    def __repr__(self) -> str:
        return f"Exercise(exercise={self.exercise}, model_answer={self.model_answer}, grading_str={self.grading_str})"

    def add_prompts(self, student_prompt: str, teacher_prompt: str, teacher_prompt_with_tips_tags: str) -> None:
        """Set prompts for the exercise."""
        self.student_prompt = student_prompt
        self.teacher_prompt = teacher_prompt
        self.teacher_prompt_with_tips_tags = teacher_prompt_with_tips_tags

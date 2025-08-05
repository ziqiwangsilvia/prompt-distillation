import json
from dataclasses import dataclass
import os

import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape
from typing import Optional, List

from curriculum import TIPS_START, TIPS_END
from core.messages import Message

def _get_messages(xml_element: ET.Element) -> List[Message]:
    """Parse and return Message objects from a <messages> element."""
    messages_element = xml_element.find('messages')
    assert messages_element is not None, "messages element not found"
    return [
        Message.from_xml_element(message)
        for message in messages_element.findall('message')
    ]


@dataclass
class Choice:
    """Represents a possible answer choice."""
    content: str
    truncated: bool = False


def _get_answer_choices(xml_element: ET.Element) -> List[Choice]:
    """Parse and return Choice objects from <answer_choices> element."""
    answer_choices_element = xml_element.find('answer_choices')
    return [
        Choice(choice.text.strip() if choice.text else ' ', truncated=choice.get('truncated') == "true")
        for choice in answer_choices_element.findall('choice')
    ]


def _get_model_answer(xml_element: ET.Element) -> Optional['ModelAnswer']:
    """Parse and return a ModelAnswer object from <model_answer> element, if present."""
    model_answer_element = xml_element.find('model_answer')
    if model_answer_element is None:
        return None
    return ModelAnswer(model_answer_element.text.strip())


def _get_grading_str(xml_element: ET.Element) -> Optional['GradingStr']:
    """Parse and return a GradingStr object from <grading_str> element, if present."""
    grading_str_element = xml_element.find('grading_str')
    if grading_str_element is None:
        return None
    return GradingStr(grading_str_element.text.strip())


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
    def from_xml(cls, xml_element: ET.Element, lesson_id: str) -> 'ExerciseWithAnswers':
        """Parse ExerciseWithAnswers from XML."""
        messages = _get_messages(xml_element)
        answer_choices = _get_answer_choices(xml_element)
        model_answer = _get_model_answer(xml_element)
        grading_str = _get_grading_str(xml_element)
        return cls(messages, answer_choices, lesson_id, model_answer, grading_str)

    def to_xml(self, parent: ET.Element) -> ET.Element:
        """Serialize ExerciseWithAnswers to XML under the given parent element."""
        element = ET.SubElement(parent, "exercise_with_answers")
        messages_element = ET.SubElement(element, "messages")
        for msg in self.messages:
            msg_element = ET.SubElement(messages_element, "message")
            msg_element.set("role", msg.role.value)
            msg_element.text = msg.content
        choices_element = ET.SubElement(element, "answer_choices")
        for choice in self.answer_choices:
            choice_element = ET.SubElement(choices_element, "choice")
            choice_element.text = choice.content
            if choice.truncated:
                choice_element.set("truncated", "true")
        if self.model_answer and len(self.model_answer):
            model_answer_element = ET.SubElement(element, "model_answer")
            model_answer_element.text = self.model_answer
        if self.grading_str and len(self.grading_str):
            grading_str_element = ET.SubElement(element, "grading_str")
            grading_str_element.text = self.grading_str
        return element

    def __str__(self):
        messages = self.messages
        answer_choices = self.answer_choices
        return f"ExerciseWithAnswers: {messages=}, {answer_choices=}"

    def __repr__(self):
        return str(self)

def xml_dump(element: ET.Element, file) -> None:
    """Write XML for an element to a file, with tag-per-line formatting."""
    xml_content = ET.tostring(element, encoding='unicode')
    # Start each tag on a new line
    xml_content = xml_content.replace("<", "\n<").replace(">", ">\n")
    xml_content = xml_content.replace(escape(TIPS_START), TIPS_START)
    xml_content = xml_content.replace(escape(TIPS_END), TIPS_END)
    file.write(xml_content)

def save_to_xml(
    filepath: os.PathLike,
    exercises_with_answers: List[ExerciseWithAnswers]
) -> None:
    """Save a list of ExerciseWithAnswers objects as an XML file."""
    root = ET.Element("exercises_with_answers")
    for ex in exercises_with_answers:
        ex.to_xml(root)
    with open(filepath, "w") as file:
        xml_dump(root, file)
    print(f"Saved to {filepath}")

from enum import Enum
from typing import Optional, Set, List
import xml.etree.ElementTree as ET

QUESTION_PLACEHOLDER = "[[QUESTION_PLACEHOLDER]]"


class Role(Enum):
    """Enumeration for message roles (system, assistant, user)."""
    SYSTEM = "system"
    AI = "assistant"
    USER = "user"

    @classmethod
    def from_value(cls, value: str) -> "Role":
        """Get Role enum from string value."""
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"No matching role for value: {value}")


class Message:
    """Represents a chat message, with role, content, tags, and an optional short version."""
    content: str
    role: Role
    tags: Set[str]
    short_content: Optional[str]

    def __init__(
        self,
        role: Role,
        content: str,
        tags: Optional[Set[str]] = None,
        short_content: Optional[str] = None
    ):
        self.role = role
        self.content = content
        self.tags = set(tags or {})
        self.short_content = short_content or ""

    def copy(self) -> "Message":
        """Return a copy of the message."""
        return Message(self.role, self.content, self.tags, self.short_content)

    def short_version(self) -> "Message":
        """Return a message with the short content."""
        return Message(self.role, self.short_content, self.tags, self.short_content)

    @classmethod
    def from_xml_element(cls, element: ET.Element) -> "Message":
        """Parse Message from XML element."""
        role = Role.from_value(element.get("role"))
        content = element.text.strip()
        return Message(role, content)

    def _header(self) -> str:
        name = self.__class__.__name__
        return f"{name.upper()} with tags ({', '.join(self.tags)}):\n"

    def __str__(self) -> str:
        role = self.role
        tags = "{" + ", ".join(str(t) for t in self.tags) + "}"
        return f"Message({role=}, tags={tags})\n------------\n{self.content}"

    def __repr__(self) -> str:
        return self.__str__()

    def dump(self) -> dict:
        """Return a dict with role and content."""
        return {
            "role": self.role.value,
            "content": self.content or "",
        }

    def to_xml(self, parent: Optional[ET.Element] = None) -> ET.Element:
        """Convert the message to an XML element (add to parent if given)."""
        if parent is None:
            msg_element = ET.Element("message")
        else:
            msg_element = ET.SubElement(parent, "message")
        msg_element.set("role", self.role.value)
        msg_element.text = self.content
        return msg_element

    def to_dict(self) -> dict:
        """Return the message as a dict (for serialization)."""
        return {
            "role": self.role.value,
            "content": self.content or "",
            "tags": list(self.tags),
            "short_content": self.short_content,
        }


def merge_messages(messages: List[Message]) -> List[Message]:
    """Merge consecutive messages of the same role into one message."""
    merged_messages = []
    new_message = None
    for message in messages:
        if new_message is None:
            new_message = message.copy()
        else:
            if message.role == new_message.role:
                new_message.content += "\n\n" + message.content
            else:
                merged_messages.append(new_message)
                new_message = message.copy()
    if new_message:
        merged_messages.append(new_message)
    return merged_messages

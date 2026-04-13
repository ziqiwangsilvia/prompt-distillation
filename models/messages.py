from enum import Enum
from typing import Optional, Set, List

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
        return Message(self.role, self.content, self.tags, self.short_content)

    def short_version(self) -> "Message":
        return Message(self.role, self.short_content, self.tags, self.short_content)

    @classmethod
    def from_dict(cls, d: dict) -> "Message":
        """Parse Message from a dict."""
        role = Role.from_value(d["role"])
        content = d["content"].strip()
        tags = set(d.get("tags", []))
        short_content = d.get("short_content")
        return Message(role, content, tags, short_content)

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
        return {
            "role": self.role.value,
            "content": self.content or "",
        }

    def to_dict(self) -> dict:
        """Return the message as a dict (for serialization)."""
        return {
            "role": self.role.value,
            "content": self.content or "",
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

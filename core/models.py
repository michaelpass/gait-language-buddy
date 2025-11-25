from dataclasses import dataclass, field
from typing import List


@dataclass
class TextAnalysis:
    """Structured result for writing evaluation."""
    proficiency: str = "A1"
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class MiniLesson:
    """Structured mini-lesson content."""
    points: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    vocabulary: List[str] = field(default_factory=list)


@dataclass
class AudioInfo:
    """
    Represents information about generated audio.
    For now, it's a stub with a placeholder path.
    """
    placeholder_path: str = "audio/lesson_stub.mp3"

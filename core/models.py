from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SceneInfo:
    """Represents a generated scene and associated image prompt."""
    language: str
    scene_description: str           # Detailed description used for evaluation
    image_prompt: str                # Prompt used to generate the image


@dataclass
class TextAnalysis:
    """Structured result for writing evaluation."""
    proficiency: str = "A1"          # e.g. A1, A2, B1, B2, C1, C2
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    score: int = 0                   # 0–100 overall performance score


@dataclass
class LessonPlan:
    """Represents a sequence of lesson 'cards' shown one at a time."""
    cards: List[str] = field(default_factory=list)


@dataclass
class AudioInfo:
    """Placeholder for future audio support (not yet used)."""
    path: Optional[str] = None

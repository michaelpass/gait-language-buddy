from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal


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
class AssessmentResult:
    """Result from the initial 3-stage language assessment."""
    proficiency: str = "A1"          # e.g. A1, A2, B1, B2, C1, C2
    vocabulary_level: str = "A1"
    grammar_level: str = "A1"
    fluency_score: int = 0           # 0–100
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class LessonCard:
    """Represents a single structured lesson card."""
    type: Literal[
        "text_question", "multiple_choice", "image_question", 
        "fill_in_blank", "vocabulary", "audio_transcription", "audio_comprehension",
        "speaking"
    ]
    question: Optional[str] = None          # Question text (for question types)
    instruction: Optional[str] = None       # Optional instruction in English
    image_prompt: Optional[str] = None      # Prompt for image generation
    image_path: Optional[str] = None        # Path to generated image (filled by client)
    
    # For text_question, fill_in_blank, image_question
    correct_answer: Optional[str] = None
    alternatives: List[str] = field(default_factory=list)
    
    # For multiple_choice
    options: List[str] = field(default_factory=list)
    correct_index: Optional[int] = None
    
    # For vocabulary
    word: Optional[str] = None
    translation: Optional[str] = None
    example: Optional[str] = None
    related_words: List[str] = field(default_factory=list)
    
    # For audio_transcription and audio_comprehension
    audio_text: Optional[str] = None        # Text to be converted to speech (in target language)
    audio_path: Optional[str] = None        # Path to generated audio file (filled by client)
    comprehension_questions: List[Dict[str, Any]] = field(default_factory=list)  # For audio_comprehension
    
    # For speaking exercises
    speaking_prompt: Optional[str] = None   # What the user should say (in target language)
    user_recording_path: Optional[str] = None  # Path to user's recorded audio
    user_transcription: Optional[str] = None   # STT result of user's speech
    
    # Feedback and vocabulary expansion
    feedback: Optional[str] = None          # Feedback shown after submission
    vocabulary_expansion: List[str] = field(default_factory=list)  # Additional vocabulary
    
    # User response tracking
    user_response: Optional[str] = None
    user_answer_index: Optional[int] = None
    is_correct: Optional[bool] = None
    card_score: int = 0                     # Score for this specific card (0-100)


@dataclass
class LessonPlan:
    """Represents a sequence of structured lesson cards."""
    cards: List[LessonCard] = field(default_factory=list)
    total_score: int = 0                    # Sum of all card scores
    proficiency_target: str = "A1"          # Target proficiency level for these lessons


@dataclass
class AssessmentCard:
    """A card used in the initial 3-stage assessment."""
    stage: int                              # 1, 2, or 3
    card: LessonCard                        # The actual card content


@dataclass
class AudioInfo:
    """Information about a generated audio file."""
    path: Optional[str] = None              # Path to the audio file
    text: Optional[str] = None              # Original text that was spoken
    language: Optional[str] = None          # Language of the audio
    voice: Optional[str] = None             # Voice used for generation
    duration_seconds: Optional[float] = None  # Duration of the audio

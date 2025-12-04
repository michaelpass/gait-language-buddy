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
    skipped: bool = False                   # True if user skipped this card (not penalized)


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


@dataclass
class TeachingCard:
    """
    A card for teaching new vocabulary, grammar, or concepts.
    These are displayed BEFORE the quiz cards, with prev/next navigation.
    """
    type: Literal[
        "vocabulary_intro",      # Introduce a new word with translation, example, audio
        "grammar_lesson",        # Explain a grammar concept with examples
        "phrase_lesson",         # Teach a common phrase or expression
        "concept_review",        # Review previously learned content
        "cultural_note",         # Cultural context for language usage
        "conjugation_table",     # Verb conjugation table with examples
    ]
    
    # Core content
    title: str = ""                         # Card title (e.g., "New Word: der Apfel")
    explanation: str = ""                   # Main explanation text in English
    
    # For vocabulary_intro
    word: Optional[str] = None              # The word in target language
    translation: Optional[str] = None       # English translation
    pronunciation_hint: Optional[str] = None  # Phonetic hint or IPA
    part_of_speech: Optional[str] = None    # noun, verb, adjective, etc.
    gender: Optional[str] = None            # For gendered languages (der/die/das, le/la, etc.)
    plural_form: Optional[str] = None       # Plural form if applicable
    
    # For conjugation_table (verbs)
    infinitive: Optional[str] = None        # Infinitive form of the verb
    conjugations: Dict[str, str] = field(default_factory=dict)  # e.g., {"ich": "gehe", "du": "gehst", ...}
    tense: Optional[str] = None             # Present, Past, Future, etc.
    verb_type: Optional[str] = None         # Regular, Irregular, Modal, etc.
    conjugation_examples: List[Dict[str, str]] = field(default_factory=list)  # Example sentences for each form
    
    # Examples and context
    example_sentence: Optional[str] = None  # Example sentence in target language
    example_translation: Optional[str] = None  # English translation of example
    additional_examples: List[Dict[str, str]] = field(default_factory=list)  # More examples
    
    # For grammar_lesson
    grammar_rule: Optional[str] = None      # The grammar rule being taught
    grammar_examples: List[Dict[str, str]] = field(default_factory=list)  # Examples of rule
    common_mistakes: List[str] = field(default_factory=list)  # Common errors to avoid
    
    # Related content
    related_words: List[str] = field(default_factory=list)  # Related vocabulary
    synonyms: List[str] = field(default_factory=list)       # Synonyms in target language
    antonyms: List[str] = field(default_factory=list)       # Antonyms
    
    # Multimedia
    image_prompt: Optional[str] = None      # For generating illustrative image
    image_path: Optional[str] = None        # Path to generated image
    audio_text: Optional[str] = None        # Text for TTS (usually the word/phrase)
    audio_path: Optional[str] = None        # Path to generated audio
    
    # Tracking
    is_review: bool = False                 # True if reviewing known content
    is_new: bool = True                     # True if introducing new content
    difficulty_level: str = "A1"            # CEFR level of this content
    
    # Memory tips
    mnemonic: Optional[str] = None          # Memory aid or tip
    usage_notes: Optional[str] = None       # When/how to use this word/concept


@dataclass
class TeachingPlan:
    """A collection of teaching cards for a lesson."""
    cards: List[TeachingCard] = field(default_factory=list)
    proficiency_target: str = "A1"          # Target proficiency level
    theme: Optional[str] = None             # Lesson theme (e.g., "Food", "Travel")
    new_words_count: int = 0                # Number of new words being introduced
    review_words_count: int = 0             # Number of words being reviewed
    grammar_concepts: List[str] = field(default_factory=list)  # Grammar concepts covered


@dataclass
class SessionStats:
    """Statistics for a completed session, used for showing changes."""
    words_learned_before: int = 0
    words_learned_after: int = 0
    fluency_before: int = 0
    fluency_after: int = 0
    proficiency_before: str = "A1"
    proficiency_after: str = "A1"
    cards_completed: int = 0
    correct_answers: int = 0
    session_duration_minutes: int = 0
    streak_before: int = 0
    streak_after: int = 0

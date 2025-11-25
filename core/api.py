"""
OpenAI-backed services for GAIT Language Buddy.

This module wraps:
- LLM evaluation of learner text
- LLM mini-lesson generation
- (For now) text-based scene generation
- (For now) stubbed audio generation

API key is expected in a .env file at the project root:

    OPENAI_API_KEY=sk-...

We use python-dotenv + os.getenv so you can keep secrets out of git.
"""

import json
import os
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from .models import TextAnalysis, MiniLesson, AudioInfo

# ---------------------------------------------------------------------------
# Environment & OpenAI client setup
# ---------------------------------------------------------------------------

# Load environment variables from .env in the project root (if present)
load_dotenv()

OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY:
    client: Optional[OpenAI] = OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None  # Forces fallback behavior if no key

# Choose a model that exists in your account
DEFAULT_MODEL = "gpt-4o-mini"

SUPPORTED_LANGUAGES = [
    "Spanish",
    "French",
    "German",
    "Japanese",
    "Chinese",
    "English",
]

# ---------------------------------------------------------------------------
# Scene generation (text description for now)
# ---------------------------------------------------------------------------


def generate_scene(language: str) -> str:
    """
    Return a text-based scene description.

    In the future this can call an image API and the UI
    can show an image instead of text.
    """
    return (
        f"[Scene for practicing {language}]\n\n"
        "You are on a busy street near a small café. People are sitting at "
        "tables outside, drinking coffee and talking. A dog is lying under one "
        "of the tables. In the background, there is a park with green trees, "
        "a fountain, and children playing."
    )


# ---------------------------------------------------------------------------
# Fallback logic (used if OpenAI is unavailable)
# ---------------------------------------------------------------------------


def _evaluate_user_text_fallback(text: str, language: str) -> TextAnalysis:
    """
    Simple heuristic evaluation used when the OpenAI client
    is not configured or the API call fails.
    """
    length = len(text.split())
    strengths: List[str] = []
    weaknesses: List[str] = []
    suggestions: List[str] = []

    if length < 20:
        proficiency = "A1"
        strengths.append("You are beginning to form simple sentences.")
        weaknesses.append("Your description is quite short; try adding more detail.")
        suggestions.append("Add more adjectives and prepositional phrases.")
    elif length < 60:
        proficiency = "A2"
        strengths.append("You can produce a short paragraph with some detail.")
        weaknesses.append("Some sentences may be repetitive or simple.")
        suggestions.append("Experiment with different sentence structures.")
    else:
        proficiency = "B1"
        strengths.append("You can write longer, more descriptive paragraphs.")
        weaknesses.append("There may still be grammatical errors and awkward phrasing.")
        suggestions.append("Focus on refining grammar and using more precise vocabulary.")

    if any(char.isdigit() for char in text):
        weaknesses.append("Numbers are written, but context may be unclear.")
        suggestions.append("Explain quantities clearly in full sentences.")

    return TextAnalysis(
        proficiency=proficiency,
        strengths=strengths,
        weaknesses=weaknesses,
        suggestions=suggestions,
    )


def _generate_mini_lesson_fallback(analysis: TextAnalysis) -> MiniLesson:
    """
    Simple rule-based mini-lesson used when the OpenAI client
    is not configured or the API call fails.
    """
    points: List[str] = []
    examples: List[str] = []
    vocab: List[str] = []

    if analysis.proficiency == "A1":
        points = [
            "Use 'there is/there are' (or equivalent in the target language) to start descriptions.",
            "Practice using basic adjectives (big, small, old, new).",
        ]
        examples = [
            "There is a small dog under the table.",
            "There are two people sitting near the window.",
        ]
        vocab = ["table", "chair", "dog", "window", "coffee"]
    elif analysis.proficiency == "A2":
        points = [
            "Add prepositional phrases to give more detail (next to, behind, in front of).",
            "Combine simple sentences with connectors (and, but, because).",
        ]
        examples = [
            "The man is sitting next to the window and reading a book.",
            "The café is full, but it is very quiet.",
        ]
        vocab = ["next to", "behind", "crowded", "quiet", "reading"]
    else:  # B1+
        points = [
            "Vary sentence openings to avoid repetition.",
            "Use more precise verbs and adjectives to create a vivid picture.",
        ]
        examples = [
            "Leaning against the wall, a teenager scrolls through their phone.",
            "Busy waiters weave quickly between the tables.",
        ]
        vocab = ["leaning", "weave", "busy", "relaxed", "atmosphere"]

    return MiniLesson(points=points, examples=examples, vocabulary=vocab)


# ---------------------------------------------------------------------------
# OpenAI-powered evaluation & mini-lesson
# ---------------------------------------------------------------------------


def evaluate_user_text(text: str, language: str) -> TextAnalysis:
    """
    Evaluate learner writing using OpenAI, falling back to
    a heuristic approach if the API is unavailable.

    Returns:
        TextAnalysis
    """
    # No client (no key or misconfig): fallback
    if client is None:
        return _evaluate_user_text_fallback(text, language)

    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert language teacher. Analyze the learner's "
                    "writing and return ONLY a JSON object (no extra text) with "
                    "this exact structure:\n\n"
                    "{\n"
                    '  "proficiency": "A1|A2|B1|B2|C1|C2",\n'
                    '  "strengths": ["string", ...],\n'
                    '  "weaknesses": ["string", ...],\n'
                    '  "suggestions": ["string", ...]\n'
                    "}\n\n"
                    "Respond in valid JSON."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Target language: {language}\n\n"
                    f"Learner text:\n{text}\n\n"
                    "Please evaluate and respond in JSON."
                ),
            },
        ]

        completion = client.chat.completions.create(
            model=DEFAULT_MODEL,
            response_format={"type": "json_object"},
            messages=messages,
            temperature=0.2,
        )

        raw = completion.choices[0].message.content
        data = json.loads(raw)

        return TextAnalysis(
            proficiency=data.get("proficiency", "A1"),
            strengths=data.get("strengths", []) or [],
            weaknesses=data.get("weaknesses", []) or [],
            suggestions=data.get("suggestions", []) or [],
        )

    except Exception:
        # During development you can log the exception.
        return _evaluate_user_text_fallback(text, language)


def generate_mini_lesson(analysis: TextAnalysis) -> MiniLesson:
    """
    Generate a short mini-lesson using OpenAI, falling back to
    a rule-based lesson if the API is unavailable.

    Returns:
        MiniLesson
    """
    if client is None:
        return _generate_mini_lesson_fallback(analysis)

    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a language tutor. Based on the learner's analysis, "
                    "create a very short mini-lesson. Return ONLY a JSON object "
                    "with this structure:\n\n"
                    "{\n"
                    '  "points": ["string", ...],\n'
                    '  "examples": ["string", ...],\n'
                    '  "vocabulary": ["string", ...]\n'
                    "}\n\n"
                    "Keep the content focused and classroom-friendly. "
                    "Respond in valid JSON."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "proficiency": analysis.proficiency,
                        "strengths": analysis.strengths,
                        "weaknesses": analysis.weaknesses,
                        "suggestions": analysis.suggestions,
                    },
                    ensure_ascii=False,
                ),
            },
        ]

        completion = client.chat.completions.create(
            model=DEFAULT_MODEL,
            response_format={"type": "json_object"},
            messages=messages,
            temperature=0.5,
        )

        raw = completion.choices[0].message.content
        data = json.loads(raw)

        return MiniLesson(
            points=data.get("points", []) or [],
            examples=data.get("examples", []) or [],
            vocabulary=data.get("vocabulary", []) or [],
        )

    except Exception:
        return _generate_mini_lesson_fallback(analysis)


# ---------------------------------------------------------------------------
# Audio (still stubbed)
# ---------------------------------------------------------------------------


def generate_audio(example_sentences: List[str]) -> AudioInfo:
    """
    Stub for text-to-speech generation.

    Later this can call OpenAI's TTS API, save audio, and return
    a real path/URL. For now, we just return a placeholder.
    """
    # Example of where you'd use example_sentences in the future.
    _ = example_sentences
    return AudioInfo(placeholder_path="audio/lesson_stub.mp3")

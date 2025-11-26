"""
OpenAI-backed services for GAIT Language Buddy (Tkinter version).

This module handles:
- Scene generation (text + image prompt)
- Image generation from an image prompt
- Evaluation of learner text against the scene
- Lesson plan (10 cards) generation

API key is expected in a .env file at the project root:

    OPENAI_API_KEY=sk-...

We use python-dotenv + os.getenv so secrets stay out of git.
"""

import base64
import json
import os
import tempfile
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from .models import SceneInfo, TextAnalysis, LessonPlan

# ---------------------------------------------------------------------------
# Environment & OpenAI client setup
# ---------------------------------------------------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client: Optional[OpenAI] = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Choose a fast-ish model
DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_IMAGE_MODEL = "gpt-image-1"

SUPPORTED_LANGUAGES = [
    "Spanish",
    "French",
    "German",
    "Japanese",
    "Chinese",
    # English intentionally omitted (UI is in English)
]


# ---------------------------------------------------------------------------
# Scene generation
# ---------------------------------------------------------------------------

def generate_scene(language: str) -> SceneInfo:
    """
    Generate a complex scene description (in English) and a concise image prompt.

    Returns a SceneInfo object. If the API is unavailable, a reasonable
    fallback scene is created.
    """
    if client is None:
        # Fallback: static description + simple prompt
        scene_description = (
            "A busy city square in the early evening. There is a large fountain in the center, "
            "surrounded by people sitting on benches and talking. A street musician is playing "
            "guitar near an open guitar case with some coins. On one side, a small food truck is "
            "serving customers, and some people are standing in line. In the background, you can see "
            "apartment buildings with lights turning on in the windows and a sky with pink and orange clouds."
        )
        image_prompt = (
            "busy city square at sunset, large fountain, people on benches, street musician with guitar, "
            "food truck with small line, apartment buildings in background, warm lighting"
        )
        return SceneInfo(language=language, scene_description=scene_description, image_prompt=image_prompt)

    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert visual scene designer helping a language tutor. "
                    "Create a rich, concrete scene that can be described in multiple sentences. "
                    "Return ONLY a JSON object with this structure:\n\n"
                    "{\n"
                    '  "scene_description": "detailed English description (3–6 sentences)",\n'
                    '  "image_prompt": "short, precise prompt for an image model"\n'
                    "}\n\n"
                    "The scene should be realistic and visually clear."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"The learner will be describing this scene in {language}. "
                    "Generate a scene that works well for that language classroom context."
                ),
            },
        ]

        completion = client.chat.completions.create(
            model=DEFAULT_CHAT_MODEL,
            response_format={"type": "json_object"},
            messages=messages,
            temperature=0.5,
        )
        raw = completion.choices[0].message.content
        data = json.loads(raw)

        return SceneInfo(
            language=language,
            scene_description=data.get("scene_description", "").strip(),
            image_prompt=data.get("image_prompt", "").strip(),
        )

    except Exception:
        # Fall back to a static scene on any error
        scene_description = (
            "A busy city square in the early evening. There is a large fountain in the center, "
            "surrounded by people sitting on benches and talking. A street musician is playing "
            "guitar near an open guitar case with some coins. On one side, a small food truck is "
            "serving customers, and some people are standing in line. In the background, you can see "
            "apartment buildings with lights turning on in the windows and a sky with pink and orange clouds."
        )
        image_prompt = (
            "busy city square at sunset, large fountain, people on benches, street musician with guitar, "
            "food truck with small line, apartment buildings in background, warm lighting"
        )
        return SceneInfo(language=language, scene_description=scene_description, image_prompt=image_prompt)


def generate_scene_image(image_prompt: str) -> Optional[str]:
    """
    Generate an image from the image prompt using OpenAI's Images API.

    Returns:
        Path to a temporary PNG file, or None if image generation fails
        or the client is not configured.
    """
    if client is None:
        return None

    try:
        result = client.images.generate(
            model=DEFAULT_IMAGE_MODEL,
            prompt=image_prompt,
            size="512x512",
            n=1,
        )
        b64_data = result.data[0].b64_json
        image_bytes = base64.b64decode(b64_data)

        fd, path = tempfile.mkstemp(suffix=".png", prefix="gait_scene_")
        with os.fdopen(fd, "wb") as f:
            f.write(image_bytes)

        return path
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Evaluation of learner text
# ---------------------------------------------------------------------------

def _evaluate_user_text_fallback(text: str, language: str, scene_description: str) -> TextAnalysis:
    """
    Simple heuristic evaluation used when the OpenAI client is not configured
    or an API call fails.
    """
    words = len(text.split())
    strengths: List[str] = []
    weaknesses: List[str] = []
    suggestions: List[str] = []

    if words < 20:
        proficiency = "A1"
        strengths.append("You are beginning to form simple sentences.")
        weaknesses.append("Your description is quite short; try adding more detail.")
        suggestions.append("Add more adjectives and mention more parts of the scene.")
        score = 40
    elif words < 60:
        proficiency = "A2"
        strengths.append("You can write a short paragraph with some detail.")
        weaknesses.append("Some sentences may be repetitive or simple.")
        suggestions.append("Experiment with longer sentences and connectors like 'because' or 'although'.")
        score = 65
    else:
        proficiency = "B1"
        strengths.append("You can write longer, more descriptive paragraphs.")
        weaknesses.append("There may still be grammatical errors and awkward phrasing.")
        suggestions.append("Focus on refining grammar and using more precise vocabulary.")
        score = 80

    if any(char.isdigit() for char in text):
        weaknesses.append("Numbers are present, but their context may be unclear.")
        suggestions.append("Explain quantities clearly in full sentences.")

    return TextAnalysis(
        proficiency=proficiency,
        strengths=strengths,
        weaknesses=weaknesses,
        suggestions=suggestions,
        score=score,
    )


def evaluate_user_text(text: str, language: str, scene_description: str) -> TextAnalysis:
    """
    Evaluate the learner's description against the original scene.

    Returns a TextAnalysis object with proficiency, strengths,
    weaknesses, suggestions, and a 0–100 score.
    """
    if client is None:
        return _evaluate_user_text_fallback(text, language, scene_description)

    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert language teacher. Compare the learner's description "
                    "in the target language to the original English scene description. "
                    "Evaluate grammatical accuracy, vocabulary richness, and how much of the "
                    "scene is captured. Return ONLY a JSON object:\n\n"
                    "{\n"
                    '  "proficiency": "A1|A2|B1|B2|C1|C2",\n'
                    '  "strengths": ["string", ...],\n'
                    '  "weaknesses": ["string", ...],\n'
                    '  "suggestions": ["string", ...],\n'
                    '  "score": 0-100\n'
                    "}\n\n"
                    "Respond in valid JSON only."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "target_language": language,
                        "scene_description_english": scene_description,
                        "learner_text": text,
                    },
                    ensure_ascii=False,
                ),
            },
        ]

        completion = client.chat.completions.create(
            model=DEFAULT_CHAT_MODEL,
            response_format={"type": "json_object"},
            messages=messages,
            temperature=0.3,
        )

        raw = completion.choices[0].message.content
        data = json.loads(raw)

        return TextAnalysis(
            proficiency=data.get("proficiency", "A1"),
            strengths=data.get("strengths", []) or [],
            weaknesses=data.get("weaknesses", []) or [],
            suggestions=data.get("suggestions", []) or [],
            score=int(data.get("score", 0)),
        )

    except Exception:
        return _evaluate_user_text_fallback(text, language, scene_description)


# ---------------------------------------------------------------------------
# Lesson plan generation
# ---------------------------------------------------------------------------

def _lesson_plan_fallback(analysis: TextAnalysis, scene_description: str) -> LessonPlan:
    """
    Generate a simple 10-card lesson using rules, for when the API
    isn't available.
    """
    cards: List[str] = [
        "Card 1: Re-read your description and underline all nouns. Add 3 more nouns related to the scene.",
        "Card 2: Write 3 new sentences describing what people are doing in the scene.",
        "Card 3: Replace simple verbs like 'to be' or 'to have' with more precise verbs in two sentences.",
        "Card 4: Write one longer sentence (at least 15 words) that connects two ideas about the scene.",
        "Card 5: Choose 5 adjectives that describe the mood of the scene and use them in sentences.",
        "Card 6: Write 3 sentences using time expressions (for example: now, later, yesterday, tomorrow).",
        "Card 7: Rewrite one of your sentences using a different word order but the same meaning.",
        "Card 8: Add one sentence that describes sounds or smells in the scene.",
        "Card 9: Add one sentence that describes what might happen next in the scene.",
        "Card 10: Read your whole paragraph aloud and mark any parts that feel unclear. Rewrite one of them.",
    ]
    return LessonPlan(cards=cards)


def generate_lesson_plan(analysis: TextAnalysis, scene_description: str) -> LessonPlan:
    """
    Generate a 10-card lesson plan using OpenAI.

    Each card is a short activity, prompt, or instruction that
    the learner will see one at a time.
    """
    if client is None:
        return _lesson_plan_fallback(analysis, scene_description)

    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful language tutor. Based on the learner's analysis and "
                    "the original scene, design 10 short lesson 'cards'. Each card is a simple "
                    "task, prompt, or instruction that focuses on vocabulary, sentence writing, "
                    "or elaborating their description.\n\n"
                    "Return ONLY a JSON object:\n"
                    "{\n"
                    '  "lesson_cards": ["card 1 text", "card 2 text", ..., "card 10 text"]\n'
                    "}\n\n"
                    "Cards should be concise, actionable, and written in English even if the "
                    "learner is writing in another language."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "scene_description_english": scene_description,
                        "analysis": {
                            "proficiency": analysis.proficiency,
                            "strengths": analysis.strengths,
                            "weaknesses": analysis.weaknesses,
                            "suggestions": analysis.suggestions,
                            "score": analysis.score,
                        },
                    },
                    ensure_ascii=False,
                ),
            },
        ]

        completion = client.chat.completions.create(
            model=DEFAULT_CHAT_MODEL,
            response_format={"type": "json_object"},
            messages=messages,
            temperature=0.7,
        )

        raw = completion.choices[0].message.content
        data = json.loads(raw)
        cards = data.get("lesson_cards", []) or []

        # Ensure at most 10; if fewer, we let it be; if more, trim.
        cards = cards[:10]
        if not cards:
            return _lesson_plan_fallback(analysis, scene_description)

        # If fewer than 10, we don't force-fill; app can handle <10 too.
        return LessonPlan(cards=cards)

    except Exception:
        return _lesson_plan_fallback(analysis, scene_description)

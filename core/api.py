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
import threading
import time
from typing import List, Optional, Callable, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

from .models import (
    SceneInfo, TextAnalysis, LessonPlan, LessonCard, 
    AssessmentResult, AssessmentCard
)
from .schemas import LESSON_CARD_SCHEMA, ASSESSMENT_CARD_SCHEMA
from .logger import logger, Timer

# ---------------------------------------------------------------------------
# Environment & OpenAI client setup
# ---------------------------------------------------------------------------

logger.separator("GAIT Language Buddy - API Module Initialization")

logger.env("Loading environment variables from .env file...")
dotenv_result = load_dotenv()
if dotenv_result:
    logger.env_success("dotenv file loaded successfully")
else:
    logger.warning("No .env file found or file is empty")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY:
    # Mask the API key for logging (show first 8 and last 4 chars)
    masked_key = f"{OPENAI_API_KEY[:8]}...{OPENAI_API_KEY[-4:]}" if len(OPENAI_API_KEY) > 12 else "***"
    logger.env_success(f"OPENAI_API_KEY found: {masked_key}")
    logger.env("Initializing OpenAI client...")
    client: Optional[OpenAI] = OpenAI(api_key=OPENAI_API_KEY)
    logger.env_success("OpenAI client initialized successfully")
else:
    logger.env_error("OPENAI_API_KEY not found in environment!")
    logger.warning("API calls will use fallback responses (no actual AI generation)")
    client = None

# Choose a fast-ish model
DEFAULT_CHAT_MODEL = "gpt-4o-mini"
# Use DALL-E 3 with standard quality for lowest latency
DEFAULT_IMAGE_MODEL = "dall-e-3"

logger.env(f"Default chat model: {DEFAULT_CHAT_MODEL}")
logger.env(f"Default image model: {DEFAULT_IMAGE_MODEL}")

SUPPORTED_LANGUAGES = [
    "Spanish",
    "French",
    "German",
    "Japanese",
    "Chinese",
    # English intentionally omitted (UI is in English)
]

logger.env(f"Supported languages: {', '.join(SUPPORTED_LANGUAGES)}")
logger.separator("API Module Ready")


# ---------------------------------------------------------------------------
# Scene generation
# ---------------------------------------------------------------------------

def generate_scene(language: str) -> SceneInfo:
    """
    Generate a complex scene description (in English) and a concise image prompt.

    Returns a SceneInfo object. If the API is unavailable, a reasonable
    fallback scene is created.
    """
    logger.api(f"generate_scene() called for language: {language}")
    
    if client is None:
        logger.warning("OpenAI client not available, using fallback scene")
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

        logger.api_call("chat.completions.create", model=DEFAULT_CHAT_MODEL)
        with Timer() as timer:
            completion = client.chat.completions.create(
                model=DEFAULT_CHAT_MODEL,
                response_format={"type": "json_object"},
                messages=messages,
                temperature=0.5,
            )
        logger.api_response("chat.completions.create", duration_ms=timer.duration_ms)
        
        raw = completion.choices[0].message.content
        data = json.loads(raw)
        
        scene = SceneInfo(
            language=language,
            scene_description=data.get("scene_description", "").strip(),
            image_prompt=data.get("image_prompt", "").strip(),
        )
        logger.success(f"Scene generated: {scene.image_prompt[:50]}...")
        return scene

    except Exception as e:
        logger.api_error(f"Scene generation failed: {e}", exc_info=True)
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


def generate_images_parallel(
    image_prompts: List[str],
    callback: Callable[[str, Optional[str]], None]
) -> None:
    """
    Generate multiple images in parallel, each in its own thread.
    Calls callback(image_prompt, image_path) for each completed image.
    This is much faster than sequential generation - all images start generating simultaneously.
    """
    total = len(image_prompts)
    if total == 0:
        logger.img("No images to generate")
        return
    
    logger.img(f"Starting parallel generation of {total} images")
    logger.task_start(f"parallel_image_generation ({total} images)")
    
    completed_count = [0]  # Use list for mutable counter in closure
    
    def _generate_single(prompt: str, index: int):
        logger.img(f"[{index + 1}/{total}] Starting generation...")
        start_time = time.perf_counter()
        
        path = generate_scene_image(prompt)
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        completed_count[0] += 1
        if path:
            logger.img(f"[{index + 1}/{total}] ✓ Complete ({duration_ms:.0f}ms) - {completed_count[0]}/{total} done")
        else:
            logger.img_error(f"[{index + 1}/{total}] ✗ Failed ({duration_ms:.0f}ms)")
        
        callback(prompt, path)
    
    # Start all image generations in parallel immediately
    for i, prompt in enumerate(image_prompts):
        thread = threading.Thread(target=lambda p=prompt, idx=i: _generate_single(p, idx), daemon=True)
        thread.start()
        logger.task(f"Spawned thread for image {i + 1}/{total}")


def generate_image_async(
    image_prompt: str,
    callback: Callable[[Optional[str]], None]
) -> None:
    """
    Generate an image asynchronously in a background thread.
    Calls callback with the image path when done, or None on error.
    """
    logger.task_start("async_image_generation")
    
    def _generate():
        start_time = time.perf_counter()
        path = generate_scene_image(image_prompt)
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        if path:
            logger.task_complete("async_image_generation", duration_ms=duration_ms)
        else:
            logger.task_error("async_image_generation", "Image generation returned None")
        
        callback(path)
    
    thread = threading.Thread(target=_generate, daemon=True)
    thread.start()


def sanitize_image_prompt(prompt: str) -> str:
    """
    Sanitize an image prompt to avoid content policy violations.
    Removes or replaces potentially problematic terms.
    """
    if not prompt:
        logger.debug("Empty prompt, using default")
        return "a simple educational illustration"
    
    # Convert to lowercase for checking
    prompt_lower = prompt.lower()
    
    # List of potentially problematic words/phrases that might trigger safety filters
    # Replace with safer alternatives
    replacements = {
        # Violence-related
        "weapon": "tool",
        "gun": "camera",
        "knife": "utensil",
        "sword": "stick",
        # Adult content indicators
        "naked": "dressed",
        "nude": "clothed",
        # Controversial topics
        "war": "peace",
        "battle": "game",
        "fight": "play",
    }
    
    sanitized = prompt
    replaced_words = []
    for word, replacement in replacements.items():
        if word in prompt_lower:
            # Replace word boundaries only
            import re
            sanitized = re.sub(r'\b' + re.escape(word) + r'\b', replacement, sanitized, flags=re.IGNORECASE)
            replaced_words.append(f"{word}->{replacement}")
    
    if replaced_words:
        logger.debug(f"Sanitized prompt: {', '.join(replaced_words)}")
    
    # Ensure prompt is educational and appropriate
    if not any(word in prompt_lower for word in ["educational", "learning", "simple", "illustration", "drawing", "picture"]):
        sanitized = f"a simple educational illustration of {sanitized}"
    
    # Limit length to avoid overly complex prompts
    if len(sanitized) > 200:
        sanitized = sanitized[:200] + "..."
        logger.debug("Truncated prompt to 200 chars")
    
    return sanitized.strip()


def generate_scene_image(image_prompt: str) -> Optional[str]:
    """
    Generate an image from the image prompt using OpenAI's Images API.
    Uses DALL-E 3 with standard quality for lowest latency.
    Automatically sanitizes prompts to avoid content policy violations.

    Returns:
        Path to a temporary PNG file, or None if image generation fails
        or the client is not configured.
    """
    if client is None:
        logger.warning("OpenAI client not available, skipping image generation")
        return None

    # Sanitize the prompt first
    sanitized_prompt = sanitize_image_prompt(image_prompt)
    logger.img_start(sanitized_prompt)
    
    try:
        logger.api_call("images.generate", model=DEFAULT_IMAGE_MODEL)
        with Timer() as timer:
            result = client.images.generate(
                model=DEFAULT_IMAGE_MODEL,
                prompt=sanitized_prompt,
                size="1024x1024",  # Standard size for DALL-E 3
                quality="standard",  # Use "standard" instead of "hd" for lower latency
                response_format="b64_json",  # Request base64 encoded image
                # Note: DALL-E 3 only generates 1 image, so n parameter is not supported
            )
        logger.api_response("images.generate", duration_ms=timer.duration_ms)
        
        # Extract base64 data from response
        image_data = result.data[0]
        if not hasattr(image_data, 'b64_json') or not image_data.b64_json:
            logger.img_error("Response missing b64_json data")
            return None
            
        image_bytes = base64.b64decode(image_data.b64_json)
        logger.debug(f"Decoded image: {len(image_bytes)} bytes")

        fd, path = tempfile.mkstemp(suffix=".png", prefix="gait_scene_")
        with os.fdopen(fd, "wb") as f:
            f.write(image_bytes)

        logger.img_complete(path, duration_ms=timer.duration_ms)
        return path
        
    except Exception as e:
        # Check if it's a content policy violation
        error_str = str(e)
        if "content_policy" in error_str or "safety" in error_str.lower():
            logger.warning(f"Content policy violation, trying safe fallback prompt")
            logger.debug(f"Blocked prompt: {image_prompt[:100]}...")
            
            try:
                safe_prompt = "a simple educational illustration suitable for language learning"
                logger.api_call("images.generate (fallback)", model=DEFAULT_IMAGE_MODEL)
                
                with Timer() as timer:
                    result = client.images.generate(
                        model=DEFAULT_IMAGE_MODEL,
                        prompt=safe_prompt,
                        size="1024x1024",
                        quality="standard",
                        response_format="b64_json",
                    )
                logger.api_response("images.generate (fallback)", duration_ms=timer.duration_ms)
                
                image_data = result.data[0]
                if hasattr(image_data, 'b64_json') and image_data.b64_json:
                    image_bytes = base64.b64decode(image_data.b64_json)
                    fd, path = tempfile.mkstemp(suffix=".png", prefix="gait_scene_")
                    with os.fdopen(fd, "wb") as f:
                        f.write(image_bytes)
                    logger.img_complete(path, duration_ms=timer.duration_ms)
                    return path
            except Exception as fallback_error:
                logger.img_error(f"Fallback generation also failed: {fallback_error}")
        
        logger.img_error(f"Image generation failed: {e}")
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
    logger.api(f"evaluate_user_text() called for {language}")
    logger.debug(f"Learner text length: {len(text)} chars")
    
    if client is None:
        logger.warning("Using fallback evaluation (no API)")
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

        logger.api_call("chat.completions.create (evaluate_user_text)", model=DEFAULT_CHAT_MODEL)
        with Timer() as timer:
            completion = client.chat.completions.create(
                model=DEFAULT_CHAT_MODEL,
                response_format={"type": "json_object"},
                messages=messages,
                temperature=0.3,
            )
        logger.api_response("chat.completions.create", duration_ms=timer.duration_ms)

        raw = completion.choices[0].message.content
        data = json.loads(raw)
        
        result = TextAnalysis(
            proficiency=data.get("proficiency", "A1"),
            strengths=data.get("strengths", []) or [],
            weaknesses=data.get("weaknesses", []) or [],
            suggestions=data.get("suggestions", []) or [],
            score=int(data.get("score", 0)),
        )
        logger.success(f"Text evaluated: proficiency={result.proficiency}, score={result.score}")
        return result

    except Exception as e:
        logger.api_error(f"Text evaluation failed: {e}", exc_info=True)
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


# ---------------------------------------------------------------------------
# 3-Stage Initial Language Assessment
# ---------------------------------------------------------------------------

def generate_assessment_cards(language: str) -> List[AssessmentCard]:
    """
    Generate 3 assessment cards for initial language fluency evaluation.
    
    Returns a list of 3 AssessmentCard objects with different question types
    to assess vocabulary, grammar, and comprehension.
    """
    logger.separator(f"Generating Assessment Cards for {language}")
    logger.api(f"generate_assessment_cards() called for language: {language}")
    
    if client is None:
        logger.warning("Using fallback assessment cards (no API)")
        return _assessment_cards_fallback(language)
    
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert language assessment designer. Create a 3-stage "
                    "fluency assessment to determine a learner's proficiency level.\n\n"
                    "IMPORTANT: Create RANDOM and VARIED assessment questions each time. "
                    "Do NOT use the same examples (like 'red apple'). Use diverse subjects: "
                    "animals, objects, scenes, activities, food, nature, etc.\n\n"
                    f"{ASSESSMENT_CARD_SCHEMA}\n\n"
                    "Return ONLY a JSON object with this structure:\n"
                    "{\n"
                    '  "assessment_cards": [\n'
                    "    {\n"
                    '      "stage": 1,\n'
                    '      "type": "image_question",\n'
                    '      "question": "...",\n'
                    '      "image_prompt": "...",\n'
                    '      "correct_answer": "...",\n'
                    '      ... (other fields based on type)\n'
                    "    },\n"
                    "    {\n"
                    '      "stage": 2,\n'
                    "      ...\n"
                    "    },\n"
                    "    {\n"
                    '      "stage": 3,\n'
                    "      ...\n"
                    "    }\n"
                    "  ]\n"
                    "}\n\n"
                    "Stage 1: A vivid image-based question that contains multiple objects/actions so learners must describe more than a single noun.\n"
                    "Stage 2: Grammar-focused fill-in-the-blank or sentence transformation question that requires using correct articles, cases, verb conjugations, or word order. Learner must answer via typed text (NO multiple choice).\n"
                    "Stage 3: A descriptive prompt that asks for 2–3 sentences analysing a situation, preference, or opinion to assess writing fluency and vocabulary range.\n"
                    f"All content should be appropriate for {language} learners at various levels.\n\n"
                    "ADDITIONAL REQUIREMENTS:\n"
                    "- Do NOT output 'multiple_choice' cards for assessments. Use text-based responses only.\n"
                    "- Prompts must be rich enough to allow the learner to demonstrate grammar and vocabulary (e.g., include multiple subjects/objects, spatial relations, opinions, tense changes).\n"
                    "- If a question uses a blank, ensure the correct answer and alternative acceptable answers reflect realistic language usage and, for languages with cases (like German), specify the required case/gender.\n"
                    "- Prefer language-ability checks (grammar, vocabulary, comprehension) over general trivia.\n\n"
                    "IMPORTANT: Image prompts must be:\n"
                    "- Educational and appropriate for all ages\n"
                    "- Simple, clear, and safe (e.g., 'a red apple', 'a friendly cat', 'a sunny park')\n"
                    "- Avoid any violence, weapons, adult content, or controversial topics\n"
                    "- Focus on everyday objects, animals, food, nature, or simple scenes\n"
                    "- Use descriptive but safe language (e.g., 'a cozy café', 'a beautiful garden')\n"
                    "- VARY THE SUBJECTS - use different objects, animals, scenes each time"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Generate a 3-stage assessment for {language} learners. "
                    "Use image-based prompts with multiple elements/actions so the learner must use detailed descriptions, "
                    "and ensure EVERY response is typed (no multiple choice buttons). Vary the question types "
                    "to comprehensively assess vocabulary, grammar, and fluency. For languages with grammatical gender or cases "
                    "(e.g., German), explicitly require correct article/case usage in at least one stage."
                ),
            },
        ]
        
        logger.api_call("chat.completions.create (assessment)", model=DEFAULT_CHAT_MODEL)
        with Timer() as timer:
            completion = client.chat.completions.create(
                model=DEFAULT_CHAT_MODEL,
                response_format={"type": "json_object"},
                messages=messages,
                temperature=0.9,  # Higher temperature for more variety/randomness in assessment content
            )
        logger.api_response("chat.completions.create", duration_ms=timer.duration_ms)
        
        raw = completion.choices[0].message.content
        data = json.loads(raw)
        card_data_list = data.get("assessment_cards", [])
        
        logger.debug(f"Received {len(card_data_list)} assessment cards from API")
        
        if len(card_data_list) < 3:
            logger.warning(f"Only {len(card_data_list)} cards received, using fallback")
            return _assessment_cards_fallback(language)
        
        # Convert JSON to AssessmentCard objects
        assessment_cards = []
        for i, card_data in enumerate(card_data_list[:3], 1):
            lesson_card = _json_to_lesson_card(card_data)
            assessment_card = AssessmentCard(
                stage=card_data.get("stage", i),
                card=lesson_card
            )
            assessment_cards.append(assessment_card)
            logger.debug(f"Assessment card {i}: type={lesson_card.type}, has_image={bool(lesson_card.image_prompt)}")
        
        logger.success(f"Generated {len(assessment_cards)} assessment cards successfully")
        return assessment_cards
    
    except Exception as e:
        logger.api_error(f"Assessment generation failed: {e}", exc_info=True)
        return _assessment_cards_fallback(language)


def _assessment_cards_fallback(language: str) -> List[AssessmentCard]:
    """Fallback assessment cards when API is unavailable."""
    cards = [
        AssessmentCard(
            stage=1,
            card=LessonCard(
                type="image_question",
                question="Beschreibe das Bild mit mindestens zwei Sätzen." if language == "German" else "Describe what is happening in this scene using at least two sentences.",
                instruction="Mention the people, objects, and actions you observe.",
                image_prompt="a lively open-air farmer's market in Europe with several stalls of colorful produce, people chatting, a baker handing bread to a customer, and children playing nearby",
                correct_answer="Learner should describe multiple elements of the market scene with complete sentences.",
                feedback="Use full sentences, correct articles, and verbs that match the actions in the scene.",
            )
        ),
        AssessmentCard(
            stage=2,
            card=LessonCard(
                type="fill_in_blank",
                question="Ich gebe ______ Frau die Blumen, weil sie heute Geburtstag hat." if language == "German" else "Complete the sentence with the correct case-sensitive phrase.",
                instruction="Provide the full phrase that correctly completes the sentence (e.g., 'der netten'). Include the correct article, case, and adjective ending.",
                correct_answer="der netten" if language == "German" else "the correct case-marked phrase",
                alternatives=["der lieben", "der freundlichen"] if language == "German" else [],
                feedback="Pay attention to dative case articles/adjective endings when indicating 'to the woman'.",
            )
        ),
        AssessmentCard(
            stage=3,
            card=LessonCard(
                type="text_question",
                question="Vergleiche zwei Reiseziele, die du gerne besuchen würdest, und erkläre warum." if language == "German" else "Compare two travel destinations you would like to visit and explain why in 3–4 sentences.",
                instruction="Use connectors (weil, obwohl / because, however, therefore) and describe preferences in detail.",
                image_prompt="a split scene showing a snowy mountain village on the left and a sunny seaside town on the right, both with detailed elements",
                correct_answer="Learner should provide a multi-sentence comparison demonstrating grammar and vocabulary.",
                feedback="Use comparative structures and give reasons or opinions to demonstrate fluency.",
            )
        ),
    ]
    return cards


def evaluate_assessment_responses(
    language: str,
    assessment_responses: List[Dict[str, Any]]
) -> AssessmentResult:
    """
    Evaluate the 3 assessment responses to determine proficiency level.
    
    Args:
        language: Target language
        assessment_responses: List of dicts with stage, response, card data
        
    Returns:
        AssessmentResult with proficiency level and recommendations
    """
    if client is None:
        return _assessment_result_fallback()
    
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert language assessor. Analyze the learner's responses "
                    "to a 3-stage language assessment and determine their proficiency level.\n\n"
                    "Return ONLY a JSON object:\n"
                    "{\n"
                    '  "proficiency": "A1|A2|B1|B2|C1|C2",\n'
                    '  "vocabulary_level": "A1|A2|B1|B2|C1|C2",\n'
                    '  "grammar_level": "A1|A2|B1|B2|C1|C2",\n'
                    '  "fluency_score": 0-100,\n'
                    '  "strengths": ["strength 1", "strength 2"],\n'
                    '  "weaknesses": ["weakness 1", "weakness 2"],\n'
                    '  "recommendations": ["recommendation 1", "recommendation 2"]\n'
                    "}"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "target_language": language,
                        "assessment_responses": assessment_responses,
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
        
        return AssessmentResult(
            proficiency=data.get("proficiency", "A1"),
            vocabulary_level=data.get("vocabulary_level", "A1"),
            grammar_level=data.get("grammar_level", "A1"),
            fluency_score=int(data.get("fluency_score", 0)),
            strengths=data.get("strengths", []) or [],
            weaknesses=data.get("weaknesses", []) or [],
            recommendations=data.get("recommendations", []) or [],
        )
    
    except Exception:
        return _assessment_result_fallback()


def _assessment_result_fallback() -> AssessmentResult:
    """Fallback assessment result."""
    return AssessmentResult(
        proficiency="A2",
        vocabulary_level="A2",
        grammar_level="A1",
        fluency_score=45,
        strengths=["Basic vocabulary recognition"],
        weaknesses=["Grammar complexity", "Sentence formation"],
        recommendations=["Focus on verb conjugations", "Practice forming complex sentences"],
    )


# ---------------------------------------------------------------------------
# Structured Lesson Plan Generation
# ---------------------------------------------------------------------------

def _clean_option_text(option: str) -> str:
    """Remove any letter/number prefixes from option text (e.g., 'A. ', 'a) ', '1. ')."""
    import re
    # Remove common prefixes like "A. ", "a) ", "1. ", "(a) ", etc.
    cleaned = re.sub(r'^[A-Z]\.\s*', '', option)  # Remove "A. "
    cleaned = re.sub(r'^[a-z]\)\s*', '', cleaned)  # Remove "a) "
    cleaned = re.sub(r'^\(\s*[a-z]\s*\)\s*', '', cleaned)  # Remove "(a) "
    cleaned = re.sub(r'^[0-9]+\.\s*', '', cleaned)  # Remove "1. "
    cleaned = re.sub(r'^[0-9]+\s*\)\s*', '', cleaned)  # Remove "1) "
    return cleaned.strip()


def _json_to_lesson_card(card_data: Dict[str, Any]) -> LessonCard:
    """Convert JSON card data to LessonCard object."""
    card_type = card_data.get("type", "text_question")
    
    # Sanitize image prompt if present
    image_prompt = card_data.get("image_prompt")
    if image_prompt:
        image_prompt = sanitize_image_prompt(image_prompt)
    
    # Clean multiple choice options - remove any letter/number prefixes
    options = card_data.get("options", []) or []
    if options and card_type == "multiple_choice":
        options = [_clean_option_text(opt) for opt in options]
    
    return LessonCard(
        type=card_type,
        question=card_data.get("question"),
        instruction=card_data.get("instruction"),
        image_prompt=image_prompt,
        correct_answer=card_data.get("correct_answer"),
        alternatives=card_data.get("alternatives", []) or [],
        options=options,
        correct_index=card_data.get("correct_index"),
        word=card_data.get("word"),
        translation=card_data.get("translation"),
        example=card_data.get("example"),
        related_words=card_data.get("related_words", []) or [],
        feedback=card_data.get("feedback"),
        vocabulary_expansion=card_data.get("vocabulary_expansion", []) or [],
    )


def generate_lesson_plan_from_assessment_responses(
    assessment_responses: List[Dict[str, Any]],
    language: str
) -> tuple:
    """
    Generate assessment result AND lesson plan in ONE API call for lowest latency.
    This combines what used to be two separate API calls into one.
    
    Returns:
        Tuple of (AssessmentResult, LessonPlan)
    """
    logger.separator(f"Generating Lesson Plan from Assessment ({language})")
    logger.api("generate_lesson_plan_from_assessment_responses() called")
    logger.debug(f"Processing {len(assessment_responses)} assessment responses")
    
    if client is None:
        logger.warning("Using fallback lesson plan (no API)")
        assessment_result = _assessment_result_fallback()
        return assessment_result, _structured_lesson_plan_fallback(assessment_result, language)
    
    try:
        logger.api("Evaluating assessment responses and generating lesson plan...")
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert language assessor and tutor. You will analyze the learner's "
                    "assessment responses to determine their proficiency level, then generate lessons "
                    "targeted specifically at that proficiency level.\n\n"
                    "STEP 1 - ASSESSMENT:\n"
                    "Analyze the assessment_responses provided. Determine the learner's:\n"
                    "- Overall proficiency level (A1, A2, B1, B2, C1, or C2)\n"
                    "- Vocabulary level\n"
                    "- Grammar level\n"
                    "- Fluency score (0-100)\n"
                    "- Strengths and weaknesses\n"
                    "- Recommendations for improvement\n\n"
                    "STEP 2 - LESSON GENERATION:\n"
                    "Based on the proficiency level determined in Step 1, create lesson content "
                    "EXACTLY at that level. For example:\n"
                    "- A1 level: Basic vocabulary, simple sentences, present tense only\n"
                    "- A2 level: Common phrases, past/present tenses, simple descriptions\n"
                    "- B1 level: Complex sentences, multiple tenses, abstract concepts\n"
                    "- B2 level: Nuanced language, conditional/subjunctive, detailed analysis\n"
                    "- C1/C2 level: Advanced vocabulary, idiomatic expressions, sophisticated grammar\n\n"
                    "The lesson content difficulty MUST match the assessed proficiency level.\n\n"
                    f"{LESSON_CARD_SCHEMA}\n\n"
                    "Return ONLY a JSON object:\n"
                    "{\n"
                    '  "assessment": {\n'
                    '    "proficiency": "A1|A2|B1|B2|C1|C2",\n'
                    '    "vocabulary_level": "A1|A2|B1|B2|C1|C2",\n'
                    '    "grammar_level": "A1|A2|B1|B2|C1|C2",\n'
                    '    "fluency_score": 0-100,\n'
                    '    "strengths": ["strength 1"],\n'
                    '    "weaknesses": ["weakness 1"],\n'
                    '    "recommendations": ["recommendation 1"]\n'
                    '  },\n'
                    '  "lesson_cards": [\n'
                    "    {\n"
                    '      "type": "multiple_choice",\n'
                    '      "question": "...",\n'
                    '      "options": ["Option text only, no A/B/C prefixes"],\n'
                    '      "correct_index": 0,\n'
                    '      "image_prompt": "...",\n'
                    '      "feedback": "...",\n'
                    '      "vocabulary_expansion": [...]\n'
                    "    },\n"
                    "    ... (exactly 10 cards)\n"
                    "  ]\n"
                    "}\n\n"
                    "Requirements:\n"
                    "- FIRST: Analyze assessment responses to determine proficiency level\n"
                    "- THEN: Create exactly 10 diverse lesson cards AT that proficiency level\n"
                    "- Use a mix of card types (multiple_choice, fill_in_blank, image_question, vocabulary)\n"
                    "- Include image_prompts for most cards (at least 7 out of 10)\n"
                    "- CRITICAL: Lesson difficulty must match the assessed proficiency level exactly\n"
                    "- Every card MUST focus on building LANGUAGE SKILLS (grammar practice, vocabulary usage, sentence construction, comprehension, speaking/writing prompts, pronunciation cues, etc.). Avoid pure trivia like geography/history unless it explicitly requires using the target language.\n"
                    "- Each card should have feedback and vocabulary_expansion\n"
                    f"- All content should be in {language}\n"
                    "- IMPORTANT: Lessons should be INDEPENDENT and can be completed in any order.\n"
                    "- Each lesson card should be self-contained - don't reference previous cards.\n\n"
                    "CRITICAL RULES:\n"
                    "- For multiple_choice: Do NOT include letter prefixes (A, B, C, D) in option text.\n"
                    "  Just provide plain option text like: ['quadratisch', 'rund', 'dreieckig']\n\n"
                    "CRITICAL: Image prompts MUST be safe and educational:\n"
                    "- Use simple, everyday objects and scenes (e.g., 'a red apple on a white table', 'a friendly dog playing', 'a sunny beach with palm trees')\n"
                    "- Avoid: violence, weapons, adult content, controversial topics\n"
                    "- Focus on: food, animals, nature, everyday objects, simple activities, educational scenes"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "target_language": language,
                        "assessment_responses": assessment_responses,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        
        logger.api_call("chat.completions.create (combined assessment+lesson)", model=DEFAULT_CHAT_MODEL)
        with Timer() as timer:
            completion = client.chat.completions.create(
                model=DEFAULT_CHAT_MODEL,
                response_format={"type": "json_object"},
                messages=messages,
                temperature=0.7,
                max_tokens=4000,  # Limit tokens for faster response (assessment + 10 lesson cards)
            )
        logger.api_response("chat.completions.create", duration_ms=timer.duration_ms)
        
        raw = completion.choices[0].message.content
        data = json.loads(raw)
        
        # Extract assessment result
        logger.debug("Extracting assessment result from response...")
        assessment_data = data.get("assessment", {})
        assessment_result = AssessmentResult(
            proficiency=assessment_data.get("proficiency", "A1"),
            vocabulary_level=assessment_data.get("vocabulary_level", "A1"),
            grammar_level=assessment_data.get("grammar_level", "A1"),
            fluency_score=int(assessment_data.get("fluency_score", 0)),
            strengths=assessment_data.get("strengths", []) or [],
            weaknesses=assessment_data.get("weaknesses", []) or [],
            recommendations=assessment_data.get("recommendations", []) or [],
        )
        logger.success(f"Assessment result: proficiency={assessment_result.proficiency}, "
                      f"fluency_score={assessment_result.fluency_score}")
        
        # Extract lesson cards
        logger.debug("Processing lesson cards...")
        card_data_list = data.get("lesson_cards", []) or []
        if not card_data_list:
            logger.warning("No lesson cards in response, using fallback")
            lesson_plan = _structured_lesson_plan_fallback(assessment_result, language)
        else:
            cards = []
            for i, card_data in enumerate(card_data_list[:10], 1):
                cards.append(_json_to_lesson_card(card_data))
            logger.debug(f"Processed {len(cards)} lesson cards")
            
            lesson_plan = LessonPlan(
                cards=cards,
                proficiency_target=assessment_result.proficiency
            )
        
        # Count cards with images
        cards_with_images = sum(1 for c in lesson_plan.cards if c.image_prompt)
        logger.success(f"Lesson plan generated: {len(lesson_plan.cards)} cards, "
                      f"{cards_with_images} with images")
        return assessment_result, lesson_plan
    
    except Exception as e:
        logger.api_error(f"Combined assessment+lesson generation failed: {e}", exc_info=True)
        assessment_result = _assessment_result_fallback()
        return assessment_result, _structured_lesson_plan_fallback(assessment_result, language)


def generate_structured_lesson_plan(
    assessment_result: AssessmentResult,
    language: str
) -> LessonPlan:
    """
    Generate a 10-card structured lesson plan based on assessment results.
    
    Uses structured JSON format to create varied lesson cards including
    multiple choice, fill-in-blank, image questions, etc.
    """
    logger.api(f"generate_structured_lesson_plan() for {language}, proficiency={assessment_result.proficiency}")
    
    if client is None:
        logger.warning("Using fallback lesson plan (no API)")
        return _structured_lesson_plan_fallback(assessment_result, language)
    
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert language tutor. Create a personalized 10-card "
                    "lesson plan based on the learner's assessment results.\n\n"
                    f"{LESSON_CARD_SCHEMA}\n\n"
                    "Return ONLY a JSON object:\n"
                    "{\n"
                    '  "lesson_cards": [\n'
                    "    {\n"
                    '      "type": "multiple_choice",\n'
                    '      "question": "...",\n'
                    '      "options": ["Option text only, no A/B/C prefixes"],\n'
                    '      "correct_index": 0,\n'
                    '      "image_prompt": "...",\n'
                    '      "feedback": "...",\n'
                    '      "vocabulary_expansion": [...]\n'
                    "    },\n"
                    "    ... (10 cards total)\n"
                    "  ]\n"
                    "}\n\n"
                    "Requirements:\n"
                    "- Create exactly 10 diverse lesson cards\n"
                    "- Use a mix of card types (multiple_choice, fill_in_blank, image_question, vocabulary)\n"
                    "- Include image_prompts for most cards (at least 7 out of 10)\n"
                    "- Target the learner's proficiency level\n"
                    "- Each card should have feedback and vocabulary_expansion\n"
                    "- Vary difficulty slightly but keep it appropriate for the level\n"
                    f"- All content should be in {language}\n\n"
                    "CRITICAL RULES:\n"
                    "- For multiple_choice cards: Do NOT include letter prefixes (A, B, C, D) in option text.\n"
                    "  Just provide plain option text like: ['quadratisch', 'rund', 'dreieckig']\n"
                    "  The UI will automatically add 'A. ', 'B. ', etc.\n\n"
                    "CRITICAL: Image prompts MUST be safe and educational:\n"
                    "- Use simple, everyday objects and scenes (e.g., 'a red apple on a white table', 'a friendly dog playing', 'a sunny beach with palm trees')\n"
                    "- Avoid: violence, weapons, adult content, controversial topics, anything inappropriate\n"
                    "- Focus on: food, animals, nature, everyday objects, simple activities, educational scenes\n"
                    "- Keep prompts clear, descriptive, and suitable for language learning contexts"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "target_language": language,
                        "assessment": {
                            "proficiency": assessment_result.proficiency,
                            "vocabulary_level": assessment_result.vocabulary_level,
                            "grammar_level": assessment_result.grammar_level,
                            "fluency_score": assessment_result.fluency_score,
                            "strengths": assessment_result.strengths,
                            "weaknesses": assessment_result.weaknesses,
                        },
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        
        logger.api_call("chat.completions.create (structured_lesson)", model=DEFAULT_CHAT_MODEL)
        with Timer() as timer:
            completion = client.chat.completions.create(
                model=DEFAULT_CHAT_MODEL,
                response_format={"type": "json_object"},
                messages=messages,
                temperature=0.7,  # Lower temp for faster, more consistent responses
                max_tokens=4000,  # Limit tokens for faster response
            )
        logger.api_response("chat.completions.create", duration_ms=timer.duration_ms)
        
        raw = completion.choices[0].message.content
        data = json.loads(raw)
        card_data_list = data.get("lesson_cards", []) or []
        
        if not card_data_list:
            logger.warning("No lesson cards in response, using fallback")
            return _structured_lesson_plan_fallback(assessment_result, language)
        
        # Convert JSON to LessonCard objects
        cards = []
        for card_data in card_data_list[:10]:
            cards.append(_json_to_lesson_card(card_data))
        
        logger.success(f"Generated {len(cards)} lesson cards")
        return LessonPlan(
            cards=cards,
            proficiency_target=assessment_result.proficiency
        )
    
    except Exception as e:
        logger.api_error(f"Lesson plan generation failed: {e}", exc_info=True)
        return _structured_lesson_plan_fallback(assessment_result, language)


def _structured_lesson_plan_fallback(
    assessment_result: AssessmentResult,
    language: str
) -> LessonPlan:
    """Fallback lesson plan when API is unavailable."""
    cards = [
        LessonCard(
            type="image_question",
            question="What is this?" if language != "Japanese" else "これは何ですか？",
            image_prompt="a simple red apple on a white table, educational illustration",
            correct_answer="apple",
            feedback="Great job!",
            vocabulary_expansion=["fruit", "healthy", "snack"],
        ),
        LessonCard(
            type="multiple_choice",
            question="How do you say 'hello'?",
            options=["hola" if language == "Spanish" else "bonjour" if language == "French" else "hallo", "goodbye", "thanks", "please"],
            correct_index=0,
            feedback="Correct!",
            vocabulary_expansion=["greeting", "polite"],
        ),
    ]
    # Fill to 10 cards with simple variations using safe prompts
    safe_prompts = [
        "a friendly cat sitting on a windowsill, simple illustration",
        "a sunny park with green grass and trees, educational style",
        "a simple house with a red roof and white walls, illustration",
        "a blue bicycle leaning against a wall, clear educational image",
        "a beautiful flower garden with colorful flowers, simple style",
        "a cheerful dog playing with a ball, friendly illustration",
        "a cozy library with books on shelves, educational scene",
        "a happy family having a picnic in a park, simple illustration",
    ]
    card_idx = 0
    prompt_idx = 0
    while len(cards) < 10:
        # Create variation with different safe image prompts
        base_card = cards[card_idx % len(cards)]
        new_card = LessonCard(
            type=base_card.type,
            question=base_card.question,
            image_prompt=safe_prompts[prompt_idx % len(safe_prompts)] if base_card.image_prompt else None,
            correct_answer=base_card.correct_answer,
            feedback=base_card.feedback,
            vocabulary_expansion=base_card.vocabulary_expansion,
        )
        cards.append(new_card)
        card_idx += 1
        prompt_idx += 1
    
    return LessonPlan(cards=cards[:10], proficiency_target=assessment_result.proficiency)


def evaluate_card_response(
    card: LessonCard,
    user_response: str,
    user_answer_index: Optional[int],
    language: str
) -> Dict[str, Any]:
    """
    Evaluate a single card response using LLM API for text/freeform responses.
    
    - Multiple choice: Instant evaluation (no API call)
    - Text/freeform responses: Uses API call with lowest latency model for accurate evaluation
    - Includes image prompt and question context for better evaluation
    
    Returns a dict with:
    - is_correct: bool
    - card_score: int (0-100)
    - feedback: str
    - correct_answer: str
    - alternatives: List[str]
    - vocabulary_expansion: List[str]
    """
    logger.api(f"evaluate_card_response() - type={card.type}")
    
    # Multiple choice: instant evaluation (no API call)
    if card.type == "multiple_choice" and user_answer_index is not None:
        is_correct = (user_answer_index == card.correct_index)
        card_score = 100 if is_correct else 0
        
        logger.debug(f"Multiple choice evaluation (no API): correct={is_correct}, score={card_score}")
        
        return {
            "is_correct": is_correct,
            "card_score": card_score,
            "feedback": card.feedback or ("Correct! Well done." if is_correct else f"Incorrect. The correct answer is: {card.correct_answer or card.options[card.correct_index] if card.correct_index is not None and card.correct_index < len(card.options) else 'Please try again.'}"),
            "correct_answer": card.correct_answer or (card.options[card.correct_index] if card.correct_index is not None and card.correct_index < len(card.options) else ""),
            "alternatives": card.alternatives or [],
            "vocabulary_expansion": card.vocabulary_expansion or [],
        }
    
    # For text-based/freeform responses, use API call for accurate evaluation
    if card.type in ("text_question", "image_question", "fill_in_blank") and user_response:
        logger.debug(f"Text response evaluation (API required), response length: {len(user_response)}")
        return _evaluate_text_response_with_api(card, user_response, language)
    
    # Fallback for other types
    logger.debug("Using fallback evaluation (no response or unknown type)")
    return {
        "is_correct": False,
        "card_score": 0,
        "feedback": "Please provide an answer.",
        "correct_answer": card.correct_answer or "",
        "alternatives": card.alternatives or [],
        "vocabulary_expansion": card.vocabulary_expansion or [],
    }


def _evaluate_text_response_with_api(
    card: LessonCard,
    user_response: str,
    language: str
) -> Dict[str, Any]:
    """
    Evaluate a text/freeform response using LLM API.
    Includes image prompt and question context for accurate evaluation.
    """
    if not client:
        logger.warning("Using fallback evaluation (no API)")
        return _evaluate_card_response_fallback(card, user_response, None)
    
    try:
        # Build context including image prompt if available
        context_parts = []
        if card.question:
            context_parts.append(f"Question: {card.question}")
        if card.image_prompt:
            context_parts.append(f"Image prompt (what the user is seeing): {card.image_prompt}")
        if card.instruction:
            context_parts.append(f"Instruction: {card.instruction}")
        
        context_text = "\n".join(context_parts) if context_parts else "No additional context."
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a language tutor evaluating a student's response. "
                    "Analyze the response for correctness, grammar, vocabulary usage, and meaning. "
                    "Consider alternative correct answers and provide helpful feedback.\n\n"
                    "Return ONLY a JSON object:\n"
                    "{\n"
                    '  "is_correct": true/false,\n'
                    '  "card_score": 0-100,\n'
                    '  "feedback": "Constructive feedback message",\n'
                    '  "correct_answer": "The correct answer",\n'
                    '  "alternatives": ["Alternative correct answer 1", "Alternative 2"],\n'
                    '  "vocabulary_expansion": ["Additional vocabulary word 1", "Word 2"]\n'
                    "}"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "target_language": language,
                        "context": context_text,
                        "expected_answer": card.correct_answer or "",
                        "expected_alternatives": card.alternatives or [],
                        "user_response": user_response,
                        "card_type": card.type,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        
        logger.api_call("chat.completions.create (evaluate_text)", model=DEFAULT_CHAT_MODEL)
        with Timer() as timer:
            completion = client.chat.completions.create(
                model=DEFAULT_CHAT_MODEL,  # gpt-4o-mini for lowest latency
                response_format={"type": "json_object"},
                messages=messages,
                temperature=0.3,
                max_tokens=500,  # Limit for faster response
            )
        logger.api_response("chat.completions.create", duration_ms=timer.duration_ms)
        
        raw = completion.choices[0].message.content
        data = json.loads(raw)
        
        result = {
            "is_correct": data.get("is_correct", False),
            "card_score": int(data.get("card_score", 0)),
            "feedback": data.get("feedback", card.feedback or "Thank you for your response."),
            "correct_answer": data.get("correct_answer", card.correct_answer or ""),
            "alternatives": data.get("alternatives", card.alternatives or []) or [],
            "vocabulary_expansion": data.get("vocabulary_expansion", card.vocabulary_expansion or []) or [],
        }
        
        logger.success(f"Text evaluation: correct={result['is_correct']}, score={result['card_score']}")
        return result
    
    except Exception as e:
        logger.api_error(f"Text evaluation API error: {e}", exc_info=True)
        return _evaluate_card_response_fallback(card, user_response, None)


def _evaluate_card_response_fallback(
    card: LessonCard,
    user_response: str,
    user_answer_index: Optional[int]
) -> Dict[str, Any]:
    """Fallback evaluation when API is unavailable."""
    is_correct = False
    if card.type == "multiple_choice" and user_answer_index is not None:
        is_correct = (user_answer_index == card.correct_index)
    elif card.correct_answer:
        is_correct = user_response.strip().lower() == card.correct_answer.strip().lower()
    
    return {
        "is_correct": is_correct,
        "card_score": 100 if is_correct else 50,
        "feedback": "Good try!" if not is_correct else "Correct!",
        "correct_answer": card.correct_answer or "",
        "alternatives": card.alternatives or [],
        "vocabulary_expansion": card.vocabulary_expansion or [],
    }


def generate_final_summary(
    lesson_plan: LessonPlan,
    assessment_result: AssessmentResult,
    language: str
) -> Dict[str, Any]:
    """
    Generate final summary with scores and study suggestions.
    
    Returns a dict with:
    - overall_score: int (0-100)
    - proficiency_improvement: str
    - study_suggestions: List[str]
    - strengths: List[str]
    - areas_to_improve: List[str]
    """
    logger.separator("Generating Final Summary")
    logger.api(f"generate_final_summary() for {language}")
    
    if client is None:
        logger.warning("Using fallback summary (no API)")
        return _final_summary_fallback(lesson_plan, assessment_result)
    
    try:
        total_score = sum(card.card_score for card in lesson_plan.cards)
        average_score = total_score // len(lesson_plan.cards) if lesson_plan.cards else 0
        logger.debug(f"Calculating summary: total_score={total_score}, avg={average_score}, cards={len(lesson_plan.cards)}")
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a language tutor providing a comprehensive summary of the learner's progress. "
                    "Analyze their performance across all lesson cards and provide actionable recommendations.\n\n"
                    "Return ONLY a JSON object:\n"
                    "{\n"
                    '  "overall_score": 0-100,\n'
                    '  "proficiency_improvement": "Brief assessment of progress",\n'
                    '  "study_suggestions": ["Suggestion 1", "Suggestion 2", "Suggestion 3"],\n'
                    '  "strengths": ["Strength 1", "Strength 2"],\n'
                    '  "areas_to_improve": ["Area 1", "Area 2"]\n'
                    "}"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "target_language": language,
                        "initial_assessment": {
                            "proficiency": assessment_result.proficiency,
                            "vocabulary_level": assessment_result.vocabulary_level,
                            "grammar_level": assessment_result.grammar_level,
                        },
                        "lesson_performance": {
                            "average_score": average_score,
                            "total_cards": len(lesson_plan.cards),
                            "card_scores": [card.card_score for card in lesson_plan.cards],
                            "card_responses": [
                                {
                                    "question": card.question or card.word or "",
                                    "user_response": card.user_response or "",
                                    "correct_answer": card.correct_answer or "",
                                    "is_correct": card.is_correct,
                                    "card_type": card.type,
                                }
                                for card in lesson_plan.cards
                                if card.user_response is not None
                            ],
                        },
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        
        logger.api_call("chat.completions.create (final_summary)", model=DEFAULT_CHAT_MODEL)
        with Timer() as timer:
            completion = client.chat.completions.create(
                model=DEFAULT_CHAT_MODEL,
                response_format={"type": "json_object"},
                messages=messages,
                temperature=0.5,
            )
        logger.api_response("chat.completions.create", duration_ms=timer.duration_ms)
        
        raw = completion.choices[0].message.content
        data = json.loads(raw)
        
        result = {
            "overall_score": int(data.get("overall_score", average_score)),
            "proficiency_improvement": data.get("proficiency_improvement", "You made good progress!"),
            "study_suggestions": data.get("study_suggestions", []) or [],
            "strengths": data.get("strengths", []) or [],
            "areas_to_improve": data.get("areas_to_improve", []) or [],
        }
        
        logger.success(f"Summary generated: overall_score={result['overall_score']}")
        return result
    
    except Exception as e:
        logger.api_error(f"Final summary generation failed: {e}", exc_info=True)
        return _final_summary_fallback(lesson_plan, assessment_result)


def _final_summary_fallback(
    lesson_plan: LessonPlan,
    assessment_result: AssessmentResult
) -> Dict[str, Any]:
    """Fallback final summary."""
    total_score = sum(card.card_score for card in lesson_plan.cards)
    average_score = total_score // len(lesson_plan.cards) if lesson_plan.cards else 0
    
    return {
        "overall_score": average_score,
        "proficiency_improvement": "Keep practicing to improve your skills!",
        "study_suggestions": [
            "Review vocabulary daily",
            "Practice speaking with native speakers",
            "Complete grammar exercises",
        ],
        "strengths": assessment_result.strengths or ["Good effort!"],
        "areas_to_improve": assessment_result.weaknesses or ["Continue practicing"],
    }

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
    AssessmentResult, AssessmentCard, TeachingCard, TeachingPlan
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
    "Arabic",
    "Japanese",
    "Chinese",
    # English intentionally omitted (UI is in English)
]

logger.env(f"Supported languages: {', '.join(SUPPORTED_LANGUAGES)}")
logger.separator("API Module Ready")


def is_api_available() -> bool:
    """Check if the OpenAI API client is properly configured."""
    return client is not None


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
# Text-to-Speech (TTS) Generation
# ---------------------------------------------------------------------------

# Voice options for different languages
# OpenAI TTS supports these voices: alloy, echo, fable, onyx, nova, shimmer
# Different voices work better for different languages
LANGUAGE_VOICE_MAP = {
    "Spanish": "nova",      # Clear female voice, good for Romance languages
    "French": "shimmer",    # Soft female voice, good for French
    "German": "onyx",       # Deep male voice, clear pronunciation
    "Arabic": "echo",       # Male voice, good for Arabic
    "Japanese": "nova",     # Female voice, clear for Japanese
    "Chinese": "nova",      # Female voice, works well for tonal languages
}

DEFAULT_TTS_VOICE = "nova"
DEFAULT_TTS_MODEL = "tts-1"  # Use tts-1 for speed, tts-1-hd for quality

logger.env(f"Default TTS model: {DEFAULT_TTS_MODEL}")


def get_voice_for_language(language: str) -> str:
    """Get the appropriate TTS voice for a given language."""
    return LANGUAGE_VOICE_MAP.get(language, DEFAULT_TTS_VOICE)


def generate_speech(
    text: str,
    language: str,
    voice: Optional[str] = None,
) -> Optional[str]:
    """
    Generate speech audio from text using OpenAI's TTS API.
    
    Args:
        text: The text to convert to speech (in the target language)
        language: The target language (used to select appropriate voice)
        voice: Optional specific voice to use (overrides language default)
    
    Returns:
        Path to the generated MP3 audio file, or None on failure
    """
    if client is None:
        logger.warning("OpenAI client not available, skipping TTS generation")
        return None
    
    if not text or not text.strip():
        logger.warning("Empty text provided for TTS")
        return None
    
    selected_voice = voice or get_voice_for_language(language)
    logger.api(f"generate_speech() - {len(text)} chars, voice={selected_voice}, lang={language}")
    
    try:
        logger.api_call("audio.speech.create", model=DEFAULT_TTS_MODEL, voice=selected_voice)
        
        with Timer() as timer:
            response = client.audio.speech.create(
                model=DEFAULT_TTS_MODEL,
                voice=selected_voice,
                input=text,
                response_format="mp3",
            )
        
        logger.api_response("audio.speech.create", duration_ms=timer.duration_ms)
        
        # Save to temporary file
        fd, path = tempfile.mkstemp(suffix=".mp3", prefix="gait_speech_")
        with os.fdopen(fd, "wb") as f:
            # Stream the response content to file
            for chunk in response.iter_bytes():
                f.write(chunk)
        
        logger.success(f"TTS audio saved: {path} ({timer.duration_ms:.0f}ms)")
        return path
        
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        return None


def generate_speech_async(
    text: str,
    language: str,
    callback: Callable[[Optional[str]], None],
    voice: Optional[str] = None,
) -> None:
    """
    Generate speech asynchronously in a background thread.
    Calls callback with the audio file path when done, or None on error.
    
    Args:
        text: The text to convert to speech
        language: The target language
        callback: Function to call with the result
        voice: Optional specific voice to use
    """
    logger.task_start("async_speech_generation")
    
    def _generate():
        start_time = time.perf_counter()
        path = generate_speech(text, language, voice)
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        if path:
            logger.task_complete("async_speech_generation", duration_ms=duration_ms)
        else:
            logger.task_error("async_speech_generation", "Speech generation returned None")
        
        callback(path)
    
    thread = threading.Thread(target=_generate, daemon=True)
    thread.start()


def generate_audio_for_cards(
    cards: List,
    language: str,
    callback: Callable[[int, Optional[str]], None],
) -> None:
    """
    Generate audio for multiple cards that have audio_text.
    Processes cards in parallel for speed.
    
    Args:
        cards: List of LessonCard objects
        language: Target language for TTS
        callback: Called with (card_index, audio_path) for each completed card
    """
    audio_cards = [
        (i, card) for i, card in enumerate(cards)
        if hasattr(card, 'audio_text') and card.audio_text
    ]
    
    if not audio_cards:
        logger.api("No audio cards to process")
        return
    
    logger.separator(f"Generating Audio for {len(audio_cards)} Cards")
    logger.task_start(f"parallel_audio_generation ({len(audio_cards)} files)")
    
    def _generate_for_card(index: int, card) -> None:
        logger.api(f"[Audio {index + 1}] Generating for: {card.audio_text[:50]}...")
        path = generate_speech(card.audio_text, language)
        callback(index, path)
    
    # Start all audio generations in parallel
    for card_index, card in audio_cards:
        thread = threading.Thread(
            target=lambda idx=card_index, c=card: _generate_for_card(idx, c),
            daemon=True
        )
        thread.start()


# ---------------------------------------------------------------------------
# Speech-to-Text (STT) Transcription
# ---------------------------------------------------------------------------

DEFAULT_STT_MODEL = "whisper-1"

logger.env(f"Default STT model: {DEFAULT_STT_MODEL}")


def transcribe_audio(
    audio_path: str,
    language: Optional[str] = None,
) -> Optional[str]:
    """
    Transcribe audio to text using OpenAI's Whisper API.
    
    Args:
        audio_path: Path to the audio file (mp3, wav, m4a, etc.)
        language: Optional language hint (ISO 639-1 code, e.g., 'es', 'de', 'fr')
    
    Returns:
        Transcribed text, or None on failure
    """
    if client is None:
        logger.warning("OpenAI client not available, cannot transcribe audio")
        return None
    
    if not audio_path:
        logger.warning("No audio path provided for transcription")
        return None
    
    # Map language names to ISO codes for Whisper
    language_codes = {
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Arabic": "ar",
        "Japanese": "ja",
        "Chinese": "zh",
    }
    
    lang_code = language_codes.get(language) if language else None
    
    logger.api(f"transcribe_audio() - file={audio_path}, lang={lang_code}")
    
    try:
        with open(audio_path, "rb") as audio_file:
            logger.api_call("audio.transcriptions.create", model=DEFAULT_STT_MODEL)
            
            with Timer() as timer:
                # Whisper API call
                kwargs = {
                    "model": DEFAULT_STT_MODEL,
                    "file": audio_file,
                    "response_format": "text",
                }
                if lang_code:
                    kwargs["language"] = lang_code
                
                transcription = client.audio.transcriptions.create(**kwargs)
            
            logger.api_response("audio.transcriptions.create", duration_ms=timer.duration_ms)
            
            # Response is just the text when response_format="text"
            result = transcription.strip() if isinstance(transcription, str) else str(transcription).strip()
            logger.success(f"Transcription complete: '{result[:50]}...' ({timer.duration_ms:.0f}ms)")
            return result
            
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return None


def transcribe_audio_async(
    audio_path: str,
    language: Optional[str],
    callback: Callable[[Optional[str]], None],
) -> None:
    """
    Transcribe audio asynchronously in a background thread.
    Calls callback with the transcribed text when done, or None on error.
    """
    logger.task_start("async_transcription")
    
    def _transcribe():
        start_time = time.perf_counter()
        result = transcribe_audio(audio_path, language)
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        if result:
            logger.task_complete("async_transcription", duration_ms=duration_ms)
        else:
            logger.task_error("async_transcription", "Transcription returned None")
        
        callback(result)
    
    thread = threading.Thread(target=_transcribe, daemon=True)
    thread.start()


def evaluate_speaking_response(
    card,
    user_transcription: str,
    language: str
) -> Dict[str, Any]:
    """
    Evaluate a speaking exercise by comparing the user's transcription
    to the expected speaking prompt.
    
    Uses LLM to assess if the user said the correct thing, accounting for
    minor variations and acceptable alternatives.
    """
    logger.api(f"evaluate_speaking_response() - expected: {card.speaking_prompt[:30]}...")
    
    if not client:
        # Simple fallback: compare transcriptions
        expected = (card.speaking_prompt or "").strip().lower()
        actual = user_transcription.strip().lower()
        is_correct = expected == actual or actual in expected or expected in actual
        return {
            "is_correct": is_correct,
            "card_score": 100 if is_correct else 50,
            "feedback": "Good job!" if is_correct else f"Expected: {card.speaking_prompt}",
            "correct_answer": card.speaking_prompt or "",
            "alternatives": card.alternatives or [],
            "vocabulary_expansion": card.vocabulary_expansion or [],
        }
    
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a language tutor evaluating a student's speaking exercise. "
                    "The student was asked to say a phrase, and their speech was transcribed. "
                    "Compare the transcription to the expected phrase. "
                    "Be lenient with minor differences, punctuation, and small variations. "
                    "Focus on whether the student communicated the same meaning.\n\n"
                    "Return ONLY a JSON object:\n"
                    "{\n"
                    '  "is_correct": true/false (true if substantially correct),\n'
                    '  "card_score": 0-100 (100=perfect match, 70-99=minor differences, 40-69=partial, 0-39=incorrect),\n'
                    '  "feedback": "Constructive feedback on their speaking - in English",\n'
                    '  "pronunciation_notes": "Any notes about what they might have mispronounced (based on transcription errors)",\n'
                    '  "vocabulary_expansion": ["Related vocabulary words"]\n'
                    "}"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "target_language": language,
                        "expected_phrase": card.speaking_prompt or "",
                        "acceptable_alternatives": card.alternatives or [],
                        "user_transcription": user_transcription,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        
        logger.api_call("chat.completions.create (speaking evaluation)", model=DEFAULT_CHAT_MODEL)
        with Timer() as timer:
            completion = client.chat.completions.create(
                model=DEFAULT_CHAT_MODEL,
                response_format={"type": "json_object"},
                messages=messages,
                temperature=0.3,
                max_tokens=500,
            )
        logger.api_response("chat.completions.create", duration_ms=timer.duration_ms)
        
        raw = completion.choices[0].message.content
        data = json.loads(raw)
        
        feedback = data.get("feedback", "Check your pronunciation.")
        if data.get("pronunciation_notes"):
            feedback += f"\n\nNote: {data.get('pronunciation_notes')}"
        
        return {
            "is_correct": data.get("is_correct", False),
            "card_score": int(data.get("card_score", 0)),
            "feedback": feedback,
            "correct_answer": card.speaking_prompt or "",
            "alternatives": card.alternatives or [],
            "vocabulary_expansion": data.get("vocabulary_expansion", []) or card.vocabulary_expansion or [],
        }
        
    except Exception as e:
        logger.error(f"Speaking evaluation failed: {e}")
        # Fallback
        expected = (card.speaking_prompt or "").strip().lower()
        actual = user_transcription.strip().lower()
        is_correct = expected == actual
        return {
            "is_correct": is_correct,
            "card_score": 100 if is_correct else 50,
            "feedback": "Good job!" if is_correct else f"Expected: {card.speaking_prompt}",
            "correct_answer": card.speaking_prompt or "",
            "alternatives": card.alternatives or [],
            "vocabulary_expansion": card.vocabulary_expansion or [],
        }


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
    Generate 10-12 comprehensive assessment cards for initial language fluency evaluation.
    
    Returns a list of AssessmentCard objects with varied question types
    to thoroughly assess vocabulary, grammar, comprehension, and fluency.
    """
    logger.separator(f"Generating Comprehensive Assessment Cards for {language}")
    logger.api(f"generate_assessment_cards() called for language: {language}")
    
    if client is None:
        logger.warning("Using fallback assessment cards (no API)")
        return _assessment_cards_fallback(language)
    
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert language assessment designer. Create a COMPREHENSIVE 10-12 question "
                    "fluency assessment to accurately determine a learner's proficiency level (A1-C2).\n\n"
                    
                    "CRITICAL FIELD REQUIREMENTS - READ CAREFULLY:\n"
                    "The 'question' field is what the user sees as the MAIN PROMPT. It must contain ALL "
                    "the information they need to answer. Never leave it vague or generic.\n\n"
                    
                    "CARD TYPE SPECIFICATIONS:\n\n"
                    
                    "1. FOR 'fill_in_blank' TYPE:\n"
                    f"   - 'question' MUST contain the ACTUAL SENTENCE in {language} with ______ for the blank\n"
                    "   - Example: 'Ich habe ______ Hund.' (I have a dog)\n"
                    "   - Example: 'Elle ______ au marché.' (She goes to the market)\n"
                    "   - 'instruction' explains what to fill in: 'Fill in the blank with the correct article'\n"
                    "   - 'correct_answer' is just the word(s) for the blank: 'einen' or 'va'\n"
                    "   - The English translation in parentheses helps the user understand context\n\n"
                    
                    "2. FOR 'text_question' TYPE:\n"
                    "   - 'question' is the complete prompt the user must respond to\n"
                    "   - Example: 'How do you say \"Good morning, how are you?\" in Spanish?'\n"
                    "   - Example: 'Write a sentence describing what you did yesterday.'\n"
                    "   - 'instruction' provides additional guidance if needed\n"
                    "   - 'correct_answer' is an example of a correct response\n\n"
                    
                    "3. FOR 'image_question' TYPE:\n"
                    f"   - 'question' asks about the image: 'What do you see? Describe it in {language}.'\n"
                    "   - 'image_prompt' describes what image to generate (REQUIRED)\n"
                    "   - 'instruction' tells how detailed the response should be\n"
                    "   - 'correct_answer' describes expected content\n\n"
                    
                    "LANGUAGE RULES:\n"
                    "- 'instruction' is ALWAYS in English\n"
                    f"- 'question' can include {language} text (especially for fill_in_blank sentences)\n"
                    "- Include English translations in parentheses for context\n"
                    f"- 'correct_answer' should be in {language}\n\n"
                    
                    "RETURN THIS EXACT JSON STRUCTURE:\n"
                    "{\n"
                    '  "assessment_cards": [\n'
                    "    {\n"
                    '      "stage": 1,\n'
                    '      "type": "image_question",\n'
                    f'      "question": "What is this object? Write the word in {language}.",\n'
                    '      "instruction": "Look at the image and write just the word for what you see.",\n'
                    '      "image_prompt": "a simple red apple on white background, clear educational style",\n'
                    f'      "correct_answer": "[word for apple in {language}]",\n'
                    '      "alternatives": ["[any acceptable variations]"]\n'
                    "    },\n"
                    "    {\n"
                    '      "stage": 2,\n'
                    '      "type": "fill_in_blank",\n'
                    f'      "question": "[ACTUAL SENTENCE IN {language.upper()} WITH ______ BLANK] (English translation)",\n'
                    '      "instruction": "Fill in the blank with the correct verb/article/word.",\n'
                    '      "correct_answer": "[the word that fills the blank]",\n'
                    '      "alternatives": ["[acceptable variations]"]\n'
                    "    },\n"
                    "    ... (10-12 cards total)\n"
                    "  ]\n"
                    "}\n\n"
                    
                    "ASSESSMENT STRUCTURE (progressive difficulty):\n"
                    "Stages 1-3 (Beginner A1-A2): Basic vocabulary, simple words, numbers, greetings\n"
                    "Stages 4-6 (Elementary A2-B1): Basic grammar, articles, simple sentences, common phrases\n"
                    "Stages 7-9 (Intermediate B1-B2): Complex grammar, verb conjugation, tenses, descriptions\n"
                    "Stages 10-12 (Advanced B2-C2): Hypotheticals, abstract concepts, extended writing\n\n"
                    
                    f"LANGUAGE-SPECIFIC FOR {language.upper()}:\n"
                    "- Include grammar features specific to this language (gender, cases, verb forms)\n"
                    "- Use culturally relevant vocabulary and scenarios\n"
                    "- Test features that distinguish proficiency levels in this language\n\n"
                    
                    "CRITICAL REMINDERS:\n"
                    "- NEVER leave 'question' as just 'Fill in the blank' - include the ACTUAL sentence!\n"
                    "- NEVER leave 'question' empty or generic\n"
                    "- ALWAYS provide concrete, complete content the user can respond to\n"
                    "- VARY the subjects: animals, food, travel, family, work, hobbies, etc."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Generate a comprehensive 10-12 question assessment for {language} learners.\n\n"
                    "IMPORTANT: For EVERY fill_in_blank question, the 'question' field MUST contain "
                    f"the actual sentence in {language} with a ______ blank, plus an English translation "
                    "in parentheses. For example:\n"
                    "'Er ______ jeden Tag zur Arbeit. (He goes to work every day.)'\n\n"
                    "For text_question types, the 'question' must clearly state what the user should write.\n"
                    "For image_question types, include an image_prompt describing what to show.\n\n"
                    "Make questions progressively harder from basic vocabulary to complex grammar. "
                    f"Include grammar features specific to {language}. "
                    "NEVER leave any question vague or missing content."
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
        
        if len(card_data_list) < 8:
            logger.warning(f"Only {len(card_data_list)} cards received, using fallback")
            return _assessment_cards_fallback(language)
        
        # Convert JSON to AssessmentCard objects (take up to 12 cards)
        assessment_cards = []
        for i, card_data in enumerate(card_data_list[:12], 1):
            lesson_card = _json_to_lesson_card(card_data)
            assessment_card = AssessmentCard(
                stage=card_data.get("stage", i),
                card=lesson_card
            )
            assessment_cards.append(assessment_card)
            logger.debug(f"Assessment card {i}: type={lesson_card.type}, has_image={bool(lesson_card.image_prompt)}")
        
        logger.success(f"Generated {len(assessment_cards)} comprehensive assessment cards")
        return assessment_cards
    
    except Exception as e:
        logger.api_error(f"Assessment generation failed: {e}", exc_info=True)
        return _assessment_cards_fallback(language)


def _assessment_cards_fallback(language: str) -> List[AssessmentCard]:
    """Fallback assessment cards when API is unavailable - comprehensive 10-question assessment."""
    
    # Language-specific content
    content = {
        "Spanish": {
            "apple": "manzana",
            "numbers": "uno, dos, tres, cuatro, cinco",
            "greeting": "Hola, me llamo...",
            "fill1_q": "Yo ______ estudiante. (I am a student)",
            "fill1_a": "soy",
            "fill1_alt": ["soy un", "soy una"],
            "bathroom": "¿Dónde está el baño?",
            "fill2_q": "Ayer yo ______ al supermercado. (Yesterday I went to the supermarket)",
            "fill2_a": "fui",
        },
        "French": {
            "apple": "pomme",
            "numbers": "un, deux, trois, quatre, cinq",
            "greeting": "Bonjour, je m'appelle...",
            "fill1_q": "Je ______ étudiant. (I am a student)",
            "fill1_a": "suis",
            "fill1_alt": ["suis un", "suis une"],
            "bathroom": "Où sont les toilettes?",
            "fill2_q": "Hier, je ______ au supermarché. (Yesterday I went to the supermarket)",
            "fill2_a": "suis allé",
        },
        "German": {
            "apple": "Apfel",
            "numbers": "eins, zwei, drei, vier, fünf",
            "greeting": "Hallo, ich heiße...",
            "fill1_q": "Ich ______ Student. (I am a student)",
            "fill1_a": "bin",
            "fill1_alt": ["bin ein"],
            "bathroom": "Wo ist die Toilette?",
            "fill2_q": "Gestern ______ ich zum Supermarkt gegangen. (Yesterday I went to the supermarket)",
            "fill2_a": "bin",
        },
    }
    
    # Default to Spanish-style if language not found
    lang_content = content.get(language, content["Spanish"])
    
    cards = [
        # BEGINNER PROBING (Stages 1-3)
        AssessmentCard(
            stage=1,
            card=LessonCard(
                type="image_question",
                question=f"What is this object? Write the word in {language}.",
                instruction="Look at the image and write just the single word for what you see.",
                image_prompt="a simple red apple on a white background, clear and educational",
                correct_answer=lang_content["apple"],
                feedback="Basic vocabulary identification.",
            )
        ),
        AssessmentCard(
            stage=2,
            card=LessonCard(
                type="text_question",
                question=f"Write the numbers 1 through 5 in {language}, spelled out as words.",
                instruction="Write each number word separated by commas (e.g., one, two, three...).",
                correct_answer=lang_content["numbers"],
                feedback="Basic number knowledge.",
            )
        ),
        AssessmentCard(
            stage=3,
            card=LessonCard(
                type="text_question",
                question=f"How do you say 'Hello, my name is [your name]' in {language}? Write the complete greeting.",
                instruction="Write the full sentence as you would introduce yourself.",
                correct_answer=lang_content["greeting"],
                feedback="Basic greeting construction.",
            )
        ),
        # ELEMENTARY PROBING (Stages 4-6)
        AssessmentCard(
            stage=4,
            card=LessonCard(
                type="fill_in_blank",
                question=lang_content["fill1_q"],
                instruction="Fill in the blank with the correct form of the verb 'to be'.",
                correct_answer=lang_content["fill1_a"],
                alternatives=lang_content["fill1_alt"],
                feedback="Basic verb conjugation.",
            )
        ),
        AssessmentCard(
            stage=5,
            card=LessonCard(
                type="image_question",
                question=f"Describe what you see in this image. Write 2-3 sentences in {language}.",
                instruction="Mention the animal, what it's doing, and where it is.",
                image_prompt="a friendly golden retriever dog playing with a red ball in a sunny green park",
                correct_answer="Description of dog playing with ball in park.",
                feedback="Basic descriptive sentences.",
            )
        ),
        AssessmentCard(
            stage=6,
            card=LessonCard(
                type="text_question",
                question=f"How would you ask 'Where is the bathroom?' in {language}? Write the complete question.",
                instruction="Write the question as you would ask it to someone.",
                correct_answer=lang_content["bathroom"],
                feedback="Common phrase usage.",
            )
        ),
        # INTERMEDIATE PROBING (Stages 7-9)
        AssessmentCard(
            stage=7,
            card=LessonCard(
                type="fill_in_blank",
                question=lang_content["fill2_q"],
                instruction="Fill in the blank with the correct past tense verb form.",
                correct_answer=lang_content["fill2_a"],
                feedback="Past tense conjugation.",
            )
        ),
        AssessmentCard(
            stage=8,
            card=LessonCard(
                type="image_question",
                question=f"Describe this busy scene in detail. Write at least 4 sentences in {language}.",
                instruction="Describe the people, what they're doing, the objects, and the setting.",
                image_prompt="a lively outdoor European cafe with people eating at tables, a waiter carrying drinks, and a street musician playing guitar nearby",
                correct_answer="Detailed description with multiple elements and actions.",
                feedback="Complex scene description.",
            )
        ),
        AssessmentCard(
            stage=9,
            card=LessonCard(
                type="text_question",
                question=f"What is your favorite food and why do you like it? Write 3-4 sentences in {language}.",
                instruction="Express your opinion and give at least two reasons. Use connecting words like 'because', 'also', 'and'.",
                correct_answer="Opinion with reasoning using appropriate connectors.",
                feedback="Opinion expression with connectors.",
            )
        ),
        # ADVANCED PROBING (Stage 10)
        AssessmentCard(
            stage=10,
            card=LessonCard(
                type="text_question",
                question=f"If you could travel anywhere in the world, where would you go and what would you do there? Write 4-5 sentences in {language}.",
                instruction="Use conditional structures (if/would) and describe your imaginary trip in detail.",
                correct_answer="Hypothetical scenario using conditional tense with detailed description.",
                feedback="Conditional/subjunctive usage.",
            )
        ),
    ]
    return cards


def evaluate_assessment_responses(
    language: str,
    assessment_responses: List[Dict[str, Any]]
) -> AssessmentResult:
    """
    Evaluate assessment responses to determine proficiency level.
    This is a DEDICATED evaluation - focused only on analyzing the user's language ability.
    
    Args:
        language: Target language
        assessment_responses: List of dicts with stage, response, card data
        
    Returns:
        AssessmentResult with proficiency level and recommendations
    """
    logger.separator(f"Evaluating Assessment Responses ({language})")
    logger.api("evaluate_assessment_responses() called")
    logger.debug(f"Evaluating {len(assessment_responses)} responses")
    
    if client is None:
        logger.warning("No API client - using fallback assessment")
        return _assessment_result_fallback()
    
    # Log full responses for debugging (not truncated!)
    logger.debug("=" * 60)
    logger.debug("FULL ASSESSMENT RESPONSES BEING EVALUATED:")
    logger.debug("=" * 60)
    for i, resp in enumerate(assessment_responses):
        logger.debug(f"\n--- Question {i+1} (Stage {resp.get('stage', '?')}) ---")
        logger.debug(f"Type: {resp.get('card_type', 'unknown')}")
        logger.debug(f"Question: {resp.get('question', 'N/A')}")
        logger.debug(f"Instruction: {resp.get('instruction', 'N/A')}")
        logger.debug(f"USER'S RESPONSE: \"{resp.get('response', '')}\"")
        logger.debug(f"Expected answer: {resp.get('correct_answer', 'N/A')}")
    logger.debug("=" * 60)
    
    try:
        # Build a detailed evaluation prompt
        system_prompt = f"""You are an expert {language} language assessor. Your ONLY task is to carefully evaluate a learner's responses and determine their CEFR proficiency level.

CRITICAL: Focus on WHAT THE LEARNER ACTUALLY WROTE, not whether it matches sample answers.

## EVALUATION METHODOLOGY

For each response, analyze:

### 1. GRAMMAR ANALYSIS
- Verb conjugations (correct person, tense, mood?)
- Case usage (nominative, accusative, dative, genitive - if applicable)
- Word order (correct placement of verbs, adjectives, adverbs?)
- Agreement (gender, number, case agreement?)
- Complex structures (relative clauses, subordinate clauses, conditionals?)

### 2. VOCABULARY ANALYSIS  
- Range: Basic everyday words only, or varied/specialized vocabulary?
- Appropriateness: Words used correctly in context?
- Sophistication: Simple words (A1-A2) or nuanced vocabulary (B2+)?

### 3. SENTENCE COMPLEXITY
- A1: Single words, memorized phrases ("Ich bin...", "Das ist...")
- A2: Simple sentences, basic connectors ("und", "aber", "weil")
- B1: Compound sentences, multiple clauses, expressing opinions
- B2: Complex subordination, conditional/subjunctive mood, nuanced expression
- C1: Sophisticated structures, idiomatic usage, stylistic variation
- C2: Near-native complexity, subtle distinctions, rare errors

### 4. RESPONSE QUALITY
- Does the response address the question meaningfully?
- Is communication successful even if not perfect?
- Length and detail appropriate to the prompt?

## PROFICIENCY LEVEL GUIDELINES

**A1 (Beginner)**: 
- Single words or very short phrases
- Many basic errors
- Limited to memorized expressions
- Example: "Ich... Apfel" or "Gut"

**A2 (Elementary)**:
- Simple sentences with basic structure
- Common vocabulary, some errors
- Can describe simple things
- Example: "Ich sehe ein Apfel auf dem Tisch"

**B1 (Intermediate)**:
- Connected sentences, can express opinions
- Multiple tenses used correctly
- Can discuss familiar topics
- Example: "In meinem Klassenzimmer gibt es viele Tische. Die Lehrerin steht vor der Tafel."

**B2 (Upper Intermediate)**:
- Complex sentences with subordination
- Conditional/subjunctive mood (Konjunktiv II in German)
- Good vocabulary range, few errors
- Example: "Wenn ich im Lotto gewinnen würde, würde ich mir ein Haus am Meer kaufen."

**C1 (Advanced)**:
- Sophisticated sentence structures
- Idiomatic expressions
- Subtle nuances, rare errors
- Example: "Obwohl ich eigentlich keine Zeit hatte, habe ich mich dennoch entschlossen, an der Veranstaltung teilzunehmen."

**C2 (Mastery)**:
- Near-native fluency
- Stylistic sophistication
- Exceptional range and precision

## SCORING GUIDELINES

**Fluency Score (0-100)**:
- 0-20: Minimal language production, mostly incorrect
- 21-40: Basic phrases, frequent errors, limited communication
- 41-60: Functional communication, moderate errors, handles familiar topics
- 61-80: Good command, occasional errors, can discuss complex topics
- 81-100: Excellent/near-native, rare errors, sophisticated expression

## CRITICAL RULES

1. **DO NOT penalize for different-but-correct answers!** 
   - If the question asks "Describe what you see" and the user writes grammatically correct sentences, that's good - even if different from the sample answer.

2. **Look for evidence of ABILITY, not just correctness**
   - Using Konjunktiv II correctly = B2+ evidence
   - Using complex subordinate clauses = B1+ evidence  
   - Using varied vocabulary appropriately = higher level evidence

3. **Evaluate the BEST evidence shown**
   - If a user shows B2-level grammar in some responses, they're likely B2 even if simpler responses exist

4. **Be generous but accurate**
   - If unsure between two levels, consider the higher one if there's clear evidence

Return ONLY a JSON object with your evaluation:
{{
  "proficiency": "A1|A2|B1|B2|C1|C2",
  "vocabulary_level": "A1|A2|B1|B2|C1|C2", 
  "grammar_level": "A1|A2|B1|B2|C1|C2",
  "fluency_score": 0-100,
  "strengths": ["specific strength with example from their writing"],
  "weaknesses": ["specific weakness with example"],
  "recommendations": ["actionable recommendation"],
  "evaluation_notes": "Brief explanation of how you determined this level"
}}"""

        user_content = f"""Please evaluate these {language} assessment responses:

TARGET LANGUAGE: {language}
NUMBER OF RESPONSES: {len(assessment_responses)}

RESPONSES TO EVALUATE:
{json.dumps(assessment_responses, ensure_ascii=False, indent=2)}

Remember: Focus on the 'response' field - that's what the learner actually wrote. Evaluate the QUALITY of their {language}, not just whether it matches the expected answer."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        
        logger.api_call("chat.completions.create (assessment evaluation)", model=DEFAULT_CHAT_MODEL)
        with Timer() as timer:
            completion = client.chat.completions.create(
                model=DEFAULT_CHAT_MODEL,
                response_format={"type": "json_object"},
                messages=messages,
                temperature=0.3,  # Lower temperature for more consistent evaluation
                max_tokens=1500,
            )
        logger.api_response("chat.completions.create (assessment)", duration_ms=timer.duration_ms)
        
        raw = completion.choices[0].message.content
        data = json.loads(raw)
        
        # Log the evaluation result
        logger.debug("=" * 60)
        logger.debug("ASSESSMENT EVALUATION RESULT:")
        logger.debug(f"  Proficiency: {data.get('proficiency', 'A1')}")
        logger.debug(f"  Vocabulary: {data.get('vocabulary_level', 'A1')}")
        logger.debug(f"  Grammar: {data.get('grammar_level', 'A1')}")
        logger.debug(f"  Fluency Score: {data.get('fluency_score', 0)}")
        logger.debug(f"  Evaluation Notes: {data.get('evaluation_notes', 'N/A')}")
        logger.debug("=" * 60)
        
        result = AssessmentResult(
            proficiency=data.get("proficiency", "A1"),
            vocabulary_level=data.get("vocabulary_level", "A1"),
            grammar_level=data.get("grammar_level", "A1"),
            fluency_score=int(data.get("fluency_score", 0)),
            strengths=data.get("strengths", []) or [],
            weaknesses=data.get("weaknesses", []) or [],
            recommendations=data.get("recommendations", []) or [],
        )
        
        logger.success(f"Assessment complete: {result.proficiency} level, {result.fluency_score}/100 fluency")
        return result
    
    except Exception as e:
        logger.error(f"Assessment evaluation failed: {e}")
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


def _normalize_word(word: str, language: str = "") -> str:
    """
    Normalize a word for duplicate detection.
    Strips articles, lowercases, removes extra whitespace.
    """
    if not word:
        return ""
    
    word = word.lower().strip()
    
    # Remove common articles for various languages
    articles = {
        # German
        "der ", "die ", "das ", "den ", "dem ", "des ",
        "ein ", "eine ", "einen ", "einem ", "einer ", "eines ",
        # Spanish
        "el ", "la ", "los ", "las ", "un ", "una ", "unos ", "unas ",
        # French
        "le ", "la ", "les ", "l'", "un ", "une ", "des ",
        # Italian
        "il ", "lo ", "la ", "i ", "gli ", "le ", "un ", "uno ", "una ",
        # Portuguese
        "o ", "a ", "os ", "as ", "um ", "uma ", "uns ", "umas ",
    }
    
    for article in articles:
        if word.startswith(article):
            word = word[len(article):]
            break
    
    # Also handle articles at the end (for some display formats like "Apfel, der")
    for article in [", der", ", die", ", das", ", el", ", la", ", le"]:
        if word.endswith(article):
            word = word[:-len(article)]
            break
    
    return word.strip()


def _json_to_lesson_card(card_data: Dict[str, Any]) -> LessonCard:
    """Convert JSON card data to LessonCard object."""
    card_type = card_data.get("type", "text_question")
    question = card_data.get("question")
    instruction = card_data.get("instruction")
    correct_answer = card_data.get("correct_answer")
    
    # VALIDATION: Fix incomplete cards missing questions
    
    # For multiple_choice: MUST have a question
    if card_type == "multiple_choice":
        # Check if question is missing, empty, or just whitespace
        question_missing = not question or not question.strip() or len(question.strip()) < 5
        if question_missing:
            options = card_data.get("options", [])
            correct_idx = card_data.get("correct_index")
            # Try to generate a meaningful question
            if correct_answer:
                question = f"Which option means '{correct_answer}'?"
            elif correct_idx is not None and options and 0 <= correct_idx < len(options):
                # Use the correct option to form a question
                correct_option = options[correct_idx]
                question = f"What does '{correct_option}' mean? Select the matching option."
            elif options and len(options) > 0:
                # Generic but descriptive question
                question = f"Select the correct {card_data.get('word') or 'translation'} from the options below:"
            else:
                question = "Choose the correct option:"
            logger.warning(f"Multiple choice card missing question - generated: '{question}'")
    
    # For fill_in_blank: MUST have a sentence with blank
    elif card_type == "fill_in_blank":
        if not question or "______" not in question:
            logger.warning(f"Incomplete fill_in_blank card detected: question='{question}', instruction='{instruction}'")
            if correct_answer:
                question = f"______ ({instruction or 'Fill in the blank'}). Answer: {correct_answer}"
                logger.warning(f"Reconstructed question: '{question}'")
            else:
                question = f"[INCOMPLETE CARD] {instruction or 'Fill in the blank with the correct word.'}"
    
    # For image_question: MUST have a question about the image
    elif card_type == "image_question":
        if not question or len(question.strip()) < 5:
            question = "What is shown in this image?"
            logger.warning(f"Image question card missing question - using default: '{question}'")
    
    # For text_question: MUST have a question
    elif card_type == "text_question":
        if not question or len(question.strip()) < 5:
            if correct_answer:
                question = f"Translate: {correct_answer}"
            else:
                question = instruction or "Answer the following:"
            logger.warning(f"Text question card missing question - generated: '{question}'")
    
    # For audio_transcription: Generate question from instruction
    elif card_type == "audio_transcription":
        if not question:
            question = instruction or "Listen and write what you hear."
    
    # For audio_comprehension: Must have a question about the audio
    elif card_type == "audio_comprehension":
        if not question or len(question.strip()) < 5:
            question = "What did you hear in the audio?"
            logger.warning(f"Audio comprehension card missing question - using default: '{question}'")
    
    # For speaking: Question should explain what to say
    elif card_type == "speaking":
        speaking_prompt = card_data.get("speaking_prompt")
        if not question and speaking_prompt:
            question = f"Say the following out loud: {speaking_prompt}"
        elif not question:
            question = instruction or "Say the phrase shown below:"
    
    # For word_order: Question should explain the task
    elif card_type == "word_order":
        source_sentence = card_data.get("source_sentence")
        if not question and source_sentence:
            question = f"Arrange the words to translate: {source_sentence}"
        elif not question:
            question = "Arrange the words in the correct order:"
    
    # Sanitize image prompt if present
    image_prompt = card_data.get("image_prompt")
    if image_prompt:
        image_prompt = sanitize_image_prompt(image_prompt)
    
    # Clean multiple choice options - remove any letter/number prefixes
    options = card_data.get("options", []) or []
    if options and card_type == "multiple_choice":
        options = [_clean_option_text(opt) for opt in options]
    
    # Handle comprehension questions for audio_comprehension cards
    comprehension_questions = card_data.get("comprehension_questions", []) or []
    
    # Handle reading comprehension fields
    reading_questions = card_data.get("reading_questions", []) or []
    vocabulary_highlights = card_data.get("vocabulary_highlights", []) or []
    
    return LessonCard(
        type=card_type,
        question=question,  # Use validated question
        instruction=instruction,
        image_prompt=image_prompt,
        correct_answer=correct_answer,
        alternatives=card_data.get("alternatives", []) or [],
        options=options,
        correct_index=card_data.get("correct_index"),
        word=card_data.get("word"),
        translation=card_data.get("translation"),
        example=card_data.get("example"),
        related_words=card_data.get("related_words", []) or [],
        audio_text=card_data.get("audio_text"),
        comprehension_questions=comprehension_questions,
        speaking_prompt=card_data.get("speaking_prompt"),
        feedback=card_data.get("feedback"),
        vocabulary_expansion=card_data.get("vocabulary_expansion", []) or [],
        # Reading comprehension fields
        reading_passage=card_data.get("reading_passage"),
        reading_translation=card_data.get("reading_translation"),
        reading_questions=reading_questions,
        vocabulary_highlights=vocabulary_highlights,
        # Writing practice fields
        writing_prompt=card_data.get("writing_prompt"),
        writing_min_words=card_data.get("writing_min_words", 20),
        writing_max_words=card_data.get("writing_max_words", 100),
        # Word order fields (Duolingo-style sentence building)
        scrambled_words=card_data.get("scrambled_words", []) or [],
        correct_word_order=card_data.get("correct_word_order", []) or [],
        distractor_words=card_data.get("distractor_words", []) or [],
        source_sentence=card_data.get("source_sentence"),
        # Difficulty level
        difficulty_level=card_data.get("difficulty_level", "A1"),
    )


def generate_lesson_plan_from_assessment_responses(
    assessment_responses: List[Dict[str, Any]],
    language: str,
    learner_context: Optional[str] = None
) -> tuple:
    """
    Generate assessment result AND lesson plan in TWO SEPARATE API calls.
    First evaluates the assessment, then generates lessons based on the result.
    
    Args:
        assessment_responses: List of assessment response dictionaries
        language: Target language
        learner_context: Optional learner profile context string from database
    
    Returns:
        Tuple of (AssessmentResult, LessonPlan)
    """
    logger.separator(f"Assessment & Lesson Generation ({language})")
    logger.api("generate_lesson_plan_from_assessment_responses() called")
    logger.debug(f"Processing {len(assessment_responses)} assessment responses")
    
    if client is None:
        logger.warning("Using fallback (no API)")
        assessment_result = _assessment_result_fallback()
        return assessment_result, _structured_lesson_plan_fallback(assessment_result, language)
    
    # STEP 1: Evaluate assessment with dedicated, focused API call
    logger.separator("Step 1: Assessment Evaluation")
    assessment_result = evaluate_assessment_responses(language, assessment_responses)
    
    # STEP 2: Generate lessons based on the evaluated proficiency
    logger.separator("Step 2: Lesson Generation")
    logger.api(f"Generating lessons for {assessment_result.proficiency} level learner...")
    
    lesson_plan = generate_structured_lesson_plan(
        assessment_result=assessment_result,
        language=language,
        learner_context=learner_context
    )
    
    logger.success(f"Complete! Assessment: {assessment_result.proficiency}, Lessons: {len(lesson_plan.cards)} cards")
    return assessment_result, lesson_plan



def generate_structured_lesson_plan(
    assessment_result: AssessmentResult,
    language: str,
    learner_context: Optional[str] = None,
    teaching_plan: Optional[TeachingPlan] = None
) -> LessonPlan:
    """
    Generate a 12-card quiz based on assessment results and teaching content.
    
    Uses structured JSON format to create varied lesson cards including
    multiple choice, fill-in-blank, image questions, etc.
    
    The quiz cards should primarily test vocabulary and concepts from the 
    teaching phase, plus reinforce existing knowledge from the learner's history.
    
    Args:
        assessment_result: The learner's assessment result
        language: Target language
        learner_context: Optional learner profile context string from database
        teaching_plan: Optional TeachingPlan with vocabulary/grammar taught this session
    """
    logger.api(f"generate_structured_lesson_plan() for {language}, proficiency={assessment_result.proficiency}")
    if learner_context:
        logger.debug(f"Including learner context ({len(learner_context)} chars)")
    if teaching_plan:
        logger.debug(f"Including teaching content: {len(teaching_plan.cards)} cards")
    
    if client is None:
        logger.warning("Using fallback lesson plan (no API)")
        return _structured_lesson_plan_fallback(assessment_result, language)
    
    # Initialize word lists (used in user message)
    all_words: List[str] = []
    all_verbs: List[str] = []
    
    # Build teaching content context if available
    teaching_context = ""
    if teaching_plan and teaching_plan.cards:
        taught_vocabulary = []
        taught_verbs = []
        taught_grammar = []
        taught_phrases = []
        
        for card in teaching_plan.cards:
            # Vocabulary cards (nouns, adjectives, etc.)
            if card.type == "vocabulary_intro" and card.word:
                taught_vocabulary.append({
                    "word": card.word,
                    "translation": card.translation,
                    "part_of_speech": card.part_of_speech or "unknown",
                    "example": card.example_sentence,
                    "is_new": card.is_new,
                })
            # Conjugation tables (VERBS - very important to test!)
            elif card.type == "conjugation_table" and card.infinitive:
                taught_verbs.append({
                    "verb": card.infinitive,
                    "translation": card.translation,
                    "tense": card.tense or "Present",
                    "conjugations": card.conjugations,
                    "examples": card.conjugation_examples[:2] if card.conjugation_examples else [],
                })
            # Grammar lessons
            elif card.type == "grammar_lesson" and card.grammar_rule:
                taught_grammar.append({
                    "rule": card.grammar_rule,
                    "explanation": card.explanation[:200] if card.explanation else "",
                })
            # Phrases
            elif card.type == "phrase_lesson" and card.word:
                taught_phrases.append({
                    "phrase": card.word,
                    "translation": card.translation,
                    "usage": card.explanation[:100] if card.explanation else "",
                })
            # Concept review (repeat of known vocab)
            elif card.type == "concept_review" and card.word:
                taught_vocabulary.append({
                    "word": card.word,
                    "translation": card.translation,
                    "part_of_speech": card.part_of_speech or "unknown",
                    "example": card.example_sentence,
                    "is_new": False,  # Review word
                })
        
        # Build explicit word list for emphasis
        all_words = [v["word"] for v in taught_vocabulary if v.get("word")]
        all_verbs = [v["verb"] for v in taught_verbs if v.get("verb")]
        
        # Debug logging for quiz generation
        logger.debug(f"Quiz will test vocabulary: {all_words[:10]}...")
        logger.debug(f"Quiz will test verbs: {all_verbs[:5]}...")
        logger.debug(f"Teaching cards breakdown: {len(taught_vocabulary)} vocab, {len(taught_verbs)} verbs, {len(taught_grammar)} grammar, {len(taught_phrases)} phrases")
        
        teaching_context = f"""
═══════════════════════════════════════════════════════════════════════════════
🎯 VOCABULARY & CONCEPTS TAUGHT THIS SESSION - QUIZ MUST TEST THESE! 🎯
═══════════════════════════════════════════════════════════════════════════════

Theme: {teaching_plan.theme or 'General'}
New words introduced: {teaching_plan.new_words_count}
Review words: {teaching_plan.review_words_count}

📚 EXACT WORDS TO TEST (vocabulary - test at least 5 of these):
{', '.join(all_words) if all_words else '(none)'}

{json.dumps(taught_vocabulary, ensure_ascii=False, indent=2)}

📝 EXACT VERBS TO TEST (with conjugations - test at least 3 of these):
{', '.join(all_verbs) if all_verbs else '(none)'}

{json.dumps(taught_verbs, ensure_ascii=False, indent=2)}

📖 GRAMMAR CONCEPTS TO TEST:
{json.dumps(taught_grammar, ensure_ascii=False, indent=2)}

💬 PHRASES TO TEST:
{json.dumps(taught_phrases, ensure_ascii=False, indent=2)}

🚨🚨🚨 CRITICAL REQUIREMENTS 🚨🚨🚨
1. At LEAST 70% of quiz questions MUST test words from the lists above
2. For VERBS: Test conjugation (fill_in_blank) using the exact conjugations shown
3. For VOCABULARY: Use multiple_choice, image_question, or fill_in_blank
4. Do NOT invent new vocabulary - USE THE WORDS LISTED ABOVE
5. The user JUST learned these - this quiz reinforces their learning!
═══════════════════════════════════════════════════════════════════════════════
"""
        logger.debug(f"Teaching context: {len(taught_vocabulary)} vocab, {len(taught_verbs)} verbs, {len(taught_grammar)} grammar, {len(taught_phrases)} phrases")
        logger.debug(f"Words to test: {all_words[:10]}... Verbs: {all_verbs[:5]}...")
    
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert language tutor creating a QUIZ to test what the learner just studied.\n\n"
                    "IMPORTANT: This quiz should primarily test vocabulary and concepts that were JUST TAUGHT "
                    "in the teaching phase (provided below). The learner has just reviewed these words/concepts, "
                    "so the quiz reinforces their learning.\n\n"
                    f"{LESSON_CARD_SCHEMA}\n\n"
                    "Return ONLY a JSON object:\n"
                    "{\n"
                    '  "lesson_cards": [\n'
                    "    {\n"
                    '      "type": "multiple_choice",\n'
                    '      "question": "What is the German word for apple?",\n'
                    '      "options": ["Apfel", "Birne", "Orange", "Banane"],\n'
                    '      "correct_index": 0,\n'
                    '      "feedback": "Correct! Apfel means apple.",\n'
                    '      "vocabulary_expansion": ["der Apfel", "die Birne"]\n'
                    "    },\n"
                    "    {\n"
                    '      "type": "fill_in_blank",\n'
                    '      "question": "Ich ______ müde. (I am tired.)",\n'
                    '      "instruction": "Fill in the verb to be",\n'
                    '      "correct_answer": "bin",\n'
                    '      "alternatives": ["bin"],\n'
                    '      "feedback": "Correct! bin is the first person form of sein.",\n'
                    '      "vocabulary_expansion": ["sein", "müde"]\n'
                    "    },\n"
                    "    ... (12 cards total)\n"
                    "  ]\n"
                    "}\n\n"
                    "Requirements:\n"
                    "- Create exactly 12 quiz cards\n"
                    "- **CRITICAL**: At least 8-10 cards should test vocabulary/grammar from the TEACHING CONTENT below\n"
                    "- The remaining 2-4 cards can test general knowledge at the learner's level\n"
                    "- Use a mix of card types - REQUIRED DISTRIBUTION:\n"
                    "  • 2-3 multiple_choice - vocabulary/translation questions\n"
                    "  • 2-3 fill_in_blank - grammar/conjugation practice\n"
                    "  • 2-3 image_question - visual vocabulary (MUST have image_prompt)\n"
                    "  • 1-2 audio_transcription - listen and write (TTS cards)\n"
                    "  • 1 audio_comprehension - listen to passage and answer\n"
                    "  • 1-2 speaking - say phrase out loud (STT cards)\n"
                    "  • 1 word_order - arrange scrambled words to form sentence\n"
                    "  • 0-1 reading_comprehension - read passage, answer questions\n"
                    "  • 0-1 writing_practice - free writing with feedback\n"
                    "- TOTAL: Exactly 12 cards\n\n"
                    "MULTIMEDIA REQUIREMENTS:\n"
                    "- Include image_prompts for at least 5 cards (image_question, multiple_choice, etc.)\n"
                    "- Audio cards (audio_transcription, audio_comprehension) need audio_text field\n"
                    "- Speaking cards need speaking_prompt field\n"
                    "- Word order cards need source_sentence, correct_word_order, scrambled_words\n"
                    "- Target the learner's proficiency level\n"
                    "- Each card should have feedback and vocabulary_expansion\n\n"
                    "🚨🚨🚨 CRITICAL - QUESTION FIELD IS MANDATORY 🚨🚨🚨\n"
                    "EVERY card MUST have a non-empty 'question' field that tells the user what to do!\n\n"
                    "✅ GOOD multiple_choice example:\n"
                    "{\n"
                    '  "type": "multiple_choice",\n'
                    '  "question": "What is the German word for \'table\'?",\n'
                    '  "options": ["Stuhl", "Tisch", "Lampe", "Freund"],\n'
                    '  "correct_index": 1,\n'
                    '  "feedback": "Tisch means table!"\n'
                    "}\n\n"
                    "❌ BAD example (NEVER DO THIS):\n"
                    "{\n"
                    '  "type": "multiple_choice",\n'
                    '  "question": "",  // EMPTY - USER WON\'T KNOW WHAT TO DO!\n'
                    '  "options": ["Stuhl", "Tisch", "Lampe", "Freund"]\n'
                    "}\n\n"
                    "OTHER RULES:\n"
                    "- Do NOT create cards that just display information\n"
                    "- Do NOT use 'vocabulary' type for quiz cards\n"
                    "- Each card must have: question, correct_answer, and a way to respond\n\n"
                    "**FILL_IN_BLANK REQUIREMENTS** (CRITICAL - MUST INCLUDE SENTENCE):\n"
                    "- The 'question' field MUST contain the COMPLETE SENTENCE with a blank (______)\n"
                    "- Format: 'SENTENCE WITH ______ BLANK. (English translation)'\n"
                    f"- Example for German: question='Ich ______ müde. (I am tired.)'\n"
                    f"- Example for Spanish: question='Yo ______ estudiante. (I am a student.)'\n"
                    "- The blank should be represented as ______ (6 underscores)\n"
                    "- correct_answer: The word that fills the blank\n"
                    "- instruction: Brief hint like 'Fill in the correct verb form'\n"
                    "- NEVER create a fill_in_blank card with just an instruction and no sentence!\n"
                    "- BAD: question='Fill in the blank with the correct verb.' (NO SENTENCE!)\n"
                    "- GOOD: question='Er ______ ein Buch. (He reads a book.)' with instruction='Fill in the verb'\n\n"
                    "**TEXT_QUESTION REQUIREMENTS**:\n"
                    "- question: Clear question asking for a translation or answer\n"
                    "- Example: question='Translate to German: The cat is sleeping.'\n"
                    "- correct_answer: The expected answer in the target language\n\n"
                    "**IMAGE_QUESTION REQUIREMENTS**:\n"
                    "- question: Ask what the image shows (e.g., 'What is this in German?')\n"
                    "- image_prompt: Description for image generation\n"
                    "- correct_answer: The word/phrase for what's shown\n\n"
                    "Examples of BAD quiz cards (DO NOT DO):\n"
                    "- Just showing a word and translation (that's teaching, not testing)\n"
                    "- Cards without a question field\n"
                    "- Cards where the user doesn't need to respond\n"
                    "- fill_in_blank without the actual sentence in the question field!\n\n"
                    "AUDIO CARD REQUIREMENTS:\n"
                    "- audio_transcription: Short 1-3 sentences in the target language for dictation practice\n"
                    f"  - audio_text must be in {language} (target language)\n"
                    "  - correct_answer should match audio_text exactly\n"
                    "  - Instruction in English: 'Listen and write what you hear'\n"
                    "- audio_comprehension: Longer passage (paragraph) with comprehension question\n"
                    f"  - audio_text: 3-6 sentences in {language} telling a story or describing a situation\n"
                    "  - question: Ask about the content IN ENGLISH\n"
                    f"  - correct_answer: The answer (can be in {language} or English depending on question)\n"
                    "- For beginners (A1-A2): Use simple sentences, common vocabulary\n"
                    "- For intermediate (B1-B2): Use more complex sentences, varied vocabulary\n"
                    "- For advanced (C1-C2): Use native-speed pacing, idioms, complex structures\n\n"
                    "SPEAKING CARD REQUIREMENTS:\n"
                    "- speaking: User records themselves saying a phrase, which is transcribed and compared\n"
                    f"  - speaking_prompt: The phrase the user should say in {language}\n"
                    "  - correct_answer: Same as speaking_prompt\n"
                    "  - instruction: 'Say the following phrase out loud:' (in English)\n"
                    "  - alternatives: Acceptable variations of the phrase\n"
                    "- For beginners (A1-A2): Simple greetings, numbers, basic vocabulary (1 sentence)\n"
                    "- For intermediate (B1-B2): Common phrases, short sentences (1-2 sentences)\n"
                    "- For advanced (C1-C2): Complex sentences, idioms (2-3 sentences)\n\n"
                    "WORD ORDER REQUIREMENTS (Duolingo-style sentence building):\n"
                    "- word_order: User arranges scrambled words to form a correct sentence\n"
                    "  - source_sentence: The sentence to translate (in English)\n"
                    f"  - correct_word_order: List of words in CORRECT order (in {language})\n"
                    "  - scrambled_words: Same words but SHUFFLED randomly\n"
                    "  - distractor_words: 1-2 WRONG words to make it harder (optional, for B1+)\n"
                    "  - correct_answer: The complete correct sentence as a string\n"
                    "  - question: 'Arrange the words to translate:' followed by source_sentence\n"
                    "- Example for German A1:\n"
                    '  source_sentence: "The dog is big"\n'
                    '  correct_word_order: ["Der", "Hund", "ist", "groß"]\n'
                    '  scrambled_words: ["groß", "Hund", "Der", "ist"]\n'
                    '  correct_answer: "Der Hund ist groß"\n'
                    "- For beginners (A1-A2): 3-5 words, no distractors\n"
                    "- For intermediate (B1-B2): 5-8 words, 1-2 distractors\n"
                    "- For advanced (C1-C2): 8+ words, complex structures, 2-3 distractors\n"
                    "- IMPORTANT: Include 1-2 word_order cards in each quiz!\n\n"
                    "READING COMPREHENSION REQUIREMENTS:\n"
                    "- reading_comprehension: A passage to read with comprehension questions\n"
                    f"  - reading_passage: A paragraph (3-8 sentences) in {language}\n"
                    "  - reading_translation: English translation (shown after answering)\n"
                    "  - reading_questions: List of comprehension questions about the passage\n"
                    "  - vocabulary_highlights: Key vocabulary from the passage with translations\n"
                    "  - question: Main question asking about the passage content\n"
                    "  - correct_answer: The expected answer\n"
                    "- For beginners: Short, simple passages about daily life, with basic vocabulary\n"
                    "- For intermediate: More complex topics, varied sentence structures\n"
                    "- For advanced: Authentic-style texts, idioms, cultural content\n\n"
                    "WRITING PRACTICE REQUIREMENTS:\n"
                    "- writing_practice: User writes original text based on a prompt\n"
                    "  - writing_prompt: A topic/scenario to write about (in English for clarity)\n"
                    "  - writing_min_words: Minimum word count (default 20)\n"
                    "  - writing_max_words: Maximum word count (default 100)\n"
                    "  - instruction: Clear instructions on what to write\n"
                    "- For beginners: Simple prompts like 'Describe your family' or 'Write 3 sentences about your day'\n"
                    "- For intermediate: More complex prompts like 'Write about a memorable trip' or 'Explain your opinion on...'\n"
                    "- For advanced: Essay-style prompts, argument construction, creative writing\n\n"
                    "CRITICAL LANGUAGE RULES:\n"
                    + (
                        # For B1+ learners, questions should be in target language
                        f"- The learner is at {assessment_result.proficiency} level (intermediate/advanced).\n"
                        f"- Questions and instructions should be written in {language} (the target language) for immersive practice.\n"
                        f"- This provides authentic language exposure appropriate for their level.\n"
                        if assessment_result.proficiency in ("B1", "B2", "C1", "C2") else
                        # For A1-A2 beginners, questions should be in English
                        f"- The learner is at {assessment_result.proficiency} level (beginner).\n"
                        "- ALL questions and instructions MUST be written in ENGLISH.\n"
                        "- This is essential because beginners would find questions in the target language intimidating.\n"
                        f"- Example: Question='What is the word for \"house\" in {language}?'\n"
                    ) +
                    f"- The learner's expected answers should ALWAYS be in {language} (target language)\n"
                    f"- Multiple choice OPTIONS should ALWAYS be in {language}\n"
                    f"- audio_text MUST ALWAYS be in {language} (it will be converted to speech)\n"
                    "- Feedback text should ALWAYS be in ENGLISH (to ensure comprehension)\n\n"
                    "OTHER RULES:\n"
                    "- For multiple_choice cards: Do NOT include letter prefixes (A, B, C, D) in option text.\n"
                    "  Just provide plain option text. The UI will automatically add 'A. ', 'B. ', etc.\n\n"
                    "CRITICAL: Image prompts MUST be safe and educational:\n"
                    "- Use simple, everyday objects and scenes (e.g., 'a red apple on a white table', 'a friendly dog playing', 'a sunny beach with palm trees')\n"
                    "- Avoid: violence, weapons, adult content, controversial topics, anything inappropriate\n"
                    "- Focus on: food, animals, nature, everyday objects, simple activities, educational scenes\n"
                    "- Keep prompts clear, descriptive, and suitable for language learning contexts"
                    + (
                        f"\n\n{teaching_context}"
                        if teaching_context else ""
                    )
                    + (
                        f"\n\n{learner_context}\n\n"
                        "PERSONALIZATION INSTRUCTIONS:\n"
                        "- Review the learner profile above to understand their history\n"
                        "- Include weak vocabulary words from the database in the quiz to reinforce them\n"
                        "- But prioritize testing the TAUGHT vocabulary from this session\n"
                        "- Create progressively challenging content based on their recent performance"
                        if learner_context else ""
                    )
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
                        # Explicitly pass words to test from teaching phase
                        "words_to_test": all_words[:15] if all_words else [],
                        "verbs_to_test": all_verbs[:10] if all_verbs else [],
                        "instruction": "Generate 12 quiz cards. PRIORITIZE testing the words and verbs listed above - they were JUST taught in this session's lesson!",
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
        for card_data in card_data_list[:12]:
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
    # Language-specific audio examples
    audio_examples = {
        "Spanish": {
            "transcription": "Hola, ¿cómo estás?",
            "comprehension": "María va al mercado todos los sábados. Ella compra frutas frescas y verduras. Le gusta especialmente las manzanas rojas.",
        },
        "French": {
            "transcription": "Bonjour, comment allez-vous?",
            "comprehension": "Pierre va au marché chaque samedi. Il achète des fruits frais et des légumes. Il aime particulièrement les pommes rouges.",
        },
        "German": {
            "transcription": "Guten Tag, wie geht es Ihnen?",
            "comprehension": "Maria geht jeden Samstag zum Markt. Sie kauft frisches Obst und Gemüse. Besonders mag sie rote Äpfel.",
        },
        "Japanese": {
            "transcription": "こんにちは、お元気ですか？",
            "comprehension": "マリアは毎週土曜日に市場に行きます。彼女は新鮮な果物と野菜を買います。特に赤いりんごが好きです。",
        },
        "Chinese": {
            "transcription": "你好，你好吗？",
            "comprehension": "玛丽亚每周六去市场。她买新鲜的水果和蔬菜。她特别喜欢红苹果。",
        },
        "Arabic": {
            "transcription": "مرحبا، كيف حالك؟",
            "comprehension": "ماريا تذهب إلى السوق كل يوم سبت. تشتري الفواكه الطازجة والخضروات. تحب التفاح الأحمر بشكل خاص.",
        },
    }
    
    audio_data = audio_examples.get(language, audio_examples["Spanish"])
    
    # Language-specific speaking examples
    speaking_examples = {
        "Spanish": "Buenos días, me llamo María.",
        "French": "Bonjour, je m'appelle Marie.",
        "German": "Guten Tag, ich heiße Maria.",
        "Japanese": "おはようございます。私はマリアです。",
        "Chinese": "早上好，我叫玛丽亚。",
        "Arabic": "صباح الخير، اسمي ماريا.",
    }
    speaking_prompt = speaking_examples.get(language, speaking_examples["Spanish"])
    
    cards = [
        LessonCard(
            type="image_question",
            question="What is this?",
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
        # Audio transcription card
        LessonCard(
            type="audio_transcription",
            instruction="Listen to the audio and write what you hear.",
            audio_text=audio_data["transcription"],
            correct_answer=audio_data["transcription"],
            feedback="Great listening skills!",
            vocabulary_expansion=["greeting", "question"],
        ),
        # Audio comprehension card
        LessonCard(
            type="audio_comprehension",
            instruction="Listen to the passage and answer the question below.",
            audio_text=audio_data["comprehension"],
            question="Where does Maria go every Saturday?",
            correct_answer="the market",
            alternatives=["market", "to the market"],
            feedback="Good comprehension!",
            vocabulary_expansion=["market", "Saturday", "fresh"],
        ),
        # Speaking card
        LessonCard(
            type="speaking",
            instruction="Say the following phrase out loud:",
            speaking_prompt=speaking_prompt,
            correct_answer=speaking_prompt,
            feedback="Good speaking practice!",
            vocabulary_expansion=["greeting", "introduction", "name"],
        ),
    ]
    # Fill to 12 cards with simple variations using safe prompts
    safe_prompts = [
        "a friendly cat sitting on a windowsill, simple illustration",
        "a sunny park with green grass and trees, educational style",
        "a simple house with a red roof and white walls, illustration",
        "a blue bicycle leaning against a wall, clear educational image",
        "a beautiful flower garden with colorful flowers, simple style",
        "a cheerful dog playing with a ball, friendly illustration",
        "a stack of colorful books on a desk, educational style",
    ]
    card_idx = 0
    prompt_idx = 0
    while len(cards) < 12:
        # Create variation with different safe image prompts (skip audio/speaking cards for variations)
        base_card = cards[card_idx % 2]  # Only use first two visual cards as templates
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
    
    return LessonPlan(cards=cards[:12], proficiency_target=assessment_result.proficiency)


# ---------------------------------------------------------------------------
# Teaching Content Generation
# ---------------------------------------------------------------------------

TEACHING_CARD_SCHEMA = """
TEACHING CARD TYPES:

1. vocabulary_intro - Introduce a new word
   Required fields:
   - title: "New Word: [word]" or similar
   - word: The word in target language (e.g., "der Apfel")
   - translation: English translation (e.g., "the apple")
   - part_of_speech: "noun", "verb", "adjective", etc.
   - gender: For gendered languages (e.g., "masculine" for German)
   - explanation: English explanation with context and usage
   - example_sentence: Example in target language
   - example_translation: English translation of example
   - pronunciation_hint: Phonetic guide or pronunciation tip
   - mnemonic: Optional memory aid
   - image_prompt: A simple image to illustrate the word
   - audio_text: The word/phrase for TTS (same as 'word')
   - related_words: 1-3 related words
   - is_new: true if new word, false if review

2. grammar_lesson - Teach a grammar concept
   Required fields:
   - title: "Grammar: [concept]"
   - grammar_rule: The rule being taught
   - explanation: Clear English explanation
   - grammar_examples: Array of {"target": "...", "english": "..."} examples
   - common_mistakes: Common errors to avoid
   - audio_text: A key example sentence for TTS

3. phrase_lesson - Teach a common phrase or expression
   Required fields:
   - title: "Phrase: [phrase type]"
   - word: The full phrase
   - translation: English meaning
   - explanation: When/how to use it
   - example_sentence: Example in context
   - example_translation: English translation
   - audio_text: The phrase for TTS

4. concept_review - Review previously learned content
   - Same structure as vocabulary_intro or phrase_lesson
   - is_review: true
   - is_new: false
"""


def _generate_conjugation_table(
    verb: str,
    translation: str,
    language: str,
    proficiency: str = "A1",
    is_new: bool = True,
    is_review: bool = False,
) -> Optional[TeachingCard]:
    """
    Generate a conjugation table for a verb that's missing one.
    Makes a quick API call to get proper conjugations.
    """
    if not verb or client is None:
        return None
    
    logger.api(f"_generate_conjugation_table() for '{verb}' in {language}")
    
    try:
        # Quick focused prompt for conjugation only
        prompt = f"""Generate conjugation for the {language} verb "{verb}" ({translation}).

Return ONLY a JSON object with:
{{
  "infinitive": "{verb}",
  "translation": "{translation}",
  "tense": "Present",
  "verb_type": "regular" or "irregular",
  "conjugations": {{
    // Use the appropriate person labels for {language}
    // German: "ich", "du", "er/sie/es", "wir", "ihr", "sie/Sie"
    // Spanish: "yo", "tú", "él/ella/usted", "nosotros", "vosotros", "ellos/ustedes"
    // French: "je", "tu", "il/elle", "nous", "vous", "ils/elles"
  }},
  "conjugation_examples": [
    {{"sentence": "Example sentence using verb", "translation": "English translation"}}
  ]
}}

Fill in the conjugations for {verb} in present tense. Include 2-3 example sentences."""

        with Timer() as timer:
            completion = client.chat.completions.create(
                model=DEFAULT_CHAT_MODEL,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": f"You are a {language} grammar expert. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500,
            )
        logger.api_response("conjugation_table", duration_ms=timer.duration_ms)
        
        raw = completion.choices[0].message.content
        data = json.loads(raw)
        
        # Build conjugation examples
        examples = []
        for ex in data.get("conjugation_examples", []):
            if isinstance(ex, dict):
                examples.append({
                    "sentence": ex.get("sentence", ""),
                    "translation": ex.get("translation", "")
                })
        
        conjugation_card = TeachingCard(
            type="conjugation_table",
            title=f"Conjugation: {verb}",
            infinitive=data.get("infinitive", verb),
            translation=data.get("translation", translation),
            tense=data.get("tense", "Present"),
            verb_type=data.get("verb_type", "regular"),
            conjugations=data.get("conjugations", {}),
            conjugation_examples=examples,
            explanation=f"Practice conjugating '{verb}' in different forms.",
            is_new=is_new,
            is_review=is_review,
            difficulty_level=proficiency,
        )
        
        logger.success(f"Generated conjugation table for '{verb}': {len(conjugation_card.conjugations)} forms")
        return conjugation_card
        
    except Exception as e:
        logger.error(f"Failed to generate conjugation table for '{verb}': {e}")
        # Return a minimal stub so the card isn't completely empty
        return TeachingCard(
            type="conjugation_table",
            title=f"Conjugation: {verb}",
            infinitive=verb,
            translation=translation,
            tense="Present",
            verb_type="unknown",
            conjugations={},
            conjugation_examples=[],
            explanation=f"Practice conjugating '{verb}' - look up the conjugation forms.",
            is_new=is_new,
            is_review=is_review,
            difficulty_level=proficiency,
        )


def generate_teaching_batch(
    language: str,
    proficiency: str,
    batch_type: str,  # "vocabulary", "grammar", "review"
    num_cards: int = 5,
    theme: Optional[str] = None,
    learner_context: Optional[str] = None,
    existing_words: Optional[List[str]] = None,  # Words already generated (avoid duplicates)
) -> List[TeachingCard]:
    """
    Generate a small batch of teaching cards quickly.
    
    This is designed for progressive loading - call multiple times with different
    batch_types to build up a complete lesson.
    
    Args:
        language: Target language
        proficiency: CEFR level
        batch_type: "vocabulary" (new words), "grammar", or "review"
        num_cards: How many cards in this batch (default 5)
        theme: Optional theme
        learner_context: User's learning history
        existing_words: Words already generated to avoid duplicates
    
    Returns:
        List of TeachingCard objects
    """
    logger.api(f"generate_teaching_batch({batch_type}) - {num_cards} cards")
    
    if client is None:
        return []
    
    try:
        # Focused prompt with diversity in language components
        type_instructions = {
            "vocabulary": f"""Generate {num_cards} NEW vocabulary cards with DIVERSE language components.

REQUIRED DIVERSITY - Include a mix of:
• NOUNS (with articles/gender where applicable) - common objects, places, people, concepts
• VERBS (action words) - everyday actions, modal verbs, reflexive verbs
• ADJECTIVES - descriptive words, colors, sizes, emotions
• ADVERBS - manner, time, place, frequency
• PREPOSITIONS - spatial relationships, time expressions
• PRONOUNS - personal, possessive, demonstrative
• CONJUNCTIONS - connecting words
• NUMBERS/QUANTIFIERS - if beginner level
• COMMON PHRASES - greetings, expressions, idioms (at appropriate level)

For {language}, also consider language-specific elements like:
- Articles and gender (German: der/die/das, French: le/la, Spanish: el/la)
- Cases (German: nominative, accusative, dative, genitive)
- Verb aspects (Russian, Polish: perfective/imperfective)
- Honorifics (Japanese, Korean)
- Tones (Chinese, Vietnamese)

🚨🚨🚨 ABSOLUTE REQUIREMENT FOR VERBS 🚨🚨🚨
When you include a verb (part_of_speech: "verb"), you MUST output TWO consecutive cards:

CARD 1: vocabulary_intro for the verb
  - type: "vocabulary_intro"
  - word: the infinitive (e.g., "gehen")
  - part_of_speech: "verb"
  - translation, explanation, example_sentence, etc.

CARD 2: IMMEDIATELY AFTER, a conjugation_table
  - type: "conjugation_table"  
  - infinitive: same verb (e.g., "gehen")
  - conjugations: {{"ich": "gehe", "du": "gehst", "er/sie/es": "geht", "wir": "gehen", "ihr": "geht", "sie/Sie": "gehen"}}
  - conjugation_examples: [{{"sentence": "Ich gehe nach Hause.", "translation": "I go home."}}]

FAILURE TO INCLUDE CONJUGATION TABLE = REJECTED CARD
Every single verb MUST have its conjugation table. No exceptions.
🚨🚨🚨 END VERB REQUIREMENT 🚨🚨🚨""",
            "grammar": f"""Generate {num_cards} grammar lesson cards for {proficiency} level {language}.

Cover DIVERSE grammar topics appropriate for this level:
• Sentence structure and word order
• Verb tenses and moods (present, past, future, subjunctive, etc.)
• Noun declensions and cases (if applicable)
• Agreement rules (gender, number, case)
• Pronouns and their usage
• Prepositions and their cases
• Articles (definite, indefinite, partitive)
• Adjective placement and agreement
• Question formation
• Negation
• Comparison (comparative, superlative)
• Relative clauses
• Reported speech

Pick topics most important for {proficiency} level.""",
            "review": f"""Generate {num_cards} REVIEW cards for vocabulary the learner needs to practice.

Include a DIVERSE mix of word types: nouns, verbs, adjectives, adverbs, etc.

🚨 MANDATORY FOR VERBS IN REVIEW 🚨
If reviewing a VERB (part_of_speech: "verb"):
1. First: vocabulary_intro card with the verb
2. IMMEDIATELY AFTER: conjugation_table card with full conjugations

Example for reviewing "sprechen" (to speak):
- Card 1: {{"type": "vocabulary_intro", "word": "sprechen", "part_of_speech": "verb", ...}}
- Card 2: {{"type": "conjugation_table", "infinitive": "sprechen", "conjugations": {{"ich": "spreche", ...}}, ...}}

EVERY verb in review MUST have a conjugation_table immediately following it.""",
        }
        
        avoid_words = ""
        if existing_words:
            # Show all existing words to avoid duplicates across batches AND previous sessions
            # Limit display to avoid overwhelming the context (but we still filter post-generation)
            display_words = existing_words[:100]  # Show first 100 for context
            avoid_words = f"""

🚫🚫🚫 ABSOLUTELY FORBIDDEN - DUPLICATE PREVENTION 🚫🚫🚫
The following {len(existing_words)} words are BANNED. Do NOT use ANY of them:
{', '.join(display_words)}{'...' if len(existing_words) > 100 else ''}

⚠️ STRICT RULE: If you generate ANY word from this list, it will be REJECTED.
⚠️ Generate COMPLETELY DIFFERENT vocabulary.
⚠️ If you're struggling, try different categories: colors, weather, body parts, emotions, food, animals, etc.
🚫🚫🚫 END FORBIDDEN LIST 🚫🚫🚫"""
        
        conjugation_schema = """
For VERB cards (part_of_speech: "verb"), ALWAYS follow with a conjugation_table card:
{
  "type": "conjugation_table",
  "title": "Conjugation: [infinitive]",
  "infinitive": "gehen",
  "translation": "to go",
  "tense": "Present",
  "verb_type": "irregular",
  "conjugations": {
    "ich": "gehe",
    "du": "gehst", 
    "er/sie/es": "geht",
    "wir": "gehen",
    "ihr": "geht",
    "sie/Sie": "gehen"
  },
  "conjugation_examples": [
    {"sentence": "Ich gehe zur Schule.", "translation": "I go to school."},
    {"sentence": "Du gehst ins Kino.", "translation": "You go to the cinema."},
    {"sentence": "Wir gehen zusammen.", "translation": "We go together."}
  ],
  "explanation": "This verb is irregular. Note the stem change in du/er forms."
}
"""
        
        # Build learner context section
        learner_section = ""
        if learner_context:
            learner_section = f"""
{learner_context}

⚠️ STRICT RULES:
1. NEVER generate any word listed under "WORDS ALREADY KNOWN" - these are FORBIDDEN
2. For REVIEW cards: Use words from "WORDS NEEDING PRACTICE" 
3. Match their current proficiency level
4. If you can't think of new words, use different word categories (colors, emotions, food, etc.)
"""
        
        # Include conjugation schema for vocabulary and review batches (both can have verbs)
        include_conjugation = batch_type in ("vocabulary", "review")
        
        system_prompt = f"""Generate teaching cards for {proficiency} level {language} learners.
{type_instructions.get(batch_type, type_instructions["vocabulary"])}
{avoid_words}
{learner_section}
{conjugation_schema if include_conjugation else ""}

Return ONLY JSON: {{"cards": [...]}}
Each card needs: type, title, word, translation, explanation, example_sentence, example_translation, image_prompt, audio_text, is_new, is_review, difficulty_level, part_of_speech.
For conjugation_table cards: infinitive, translation, tense, verb_type, conjugations (dict), conjugation_examples (list).
{f'Theme: {theme}' if theme else ''}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate {num_cards} {batch_type} cards now."},
        ]
        
        logger.api_call(f"chat.completions.create (teaching_batch_{batch_type})", model=DEFAULT_CHAT_MODEL)
        with Timer() as timer:
            completion = client.chat.completions.create(
                model=DEFAULT_CHAT_MODEL,
                response_format={"type": "json_object"},
                messages=messages,
                temperature=0.7,
                max_tokens=2000,  # Smaller for faster response
            )
        logger.api_response(f"teaching_batch_{batch_type}", duration_ms=timer.duration_ms)
        
        raw = completion.choices[0].message.content
        data = json.loads(raw)
        cards_data = data.get("cards", data.get("teaching_cards", []))
        
        # Normalize existing words for comparison (lowercase, strip articles)
        normalized_existing = set()
        for w in (existing_words or []):
            normalized_existing.add(_normalize_word(w, language))
        
        cards = []
        seen_in_batch = set()  # Track words seen in THIS batch too
        
        for card_data in cards_data:
            card = _json_to_teaching_card(card_data)
            
            # Check for duplicates (both against existing_words AND within this batch)
            word_to_check = card.word or card.infinitive or ""
            normalized_word = _normalize_word(word_to_check, language)
            
            if normalized_word and normalized_word in normalized_existing:
                logger.warning(f"Filtering duplicate word from batch: '{word_to_check}' (normalized: '{normalized_word}')")
                continue  # Skip this duplicate card
            
            if normalized_word and normalized_word in seen_in_batch:
                logger.warning(f"Filtering duplicate within batch: '{word_to_check}'")
                continue  # Skip duplicate within same batch
            
            # Track this word
            if normalized_word:
                seen_in_batch.add(normalized_word)
            
            # Set batch type flags
            if batch_type == "vocabulary":
                card.is_new = True
                card.is_review = False
            elif batch_type == "review":
                card.is_new = False
                card.is_review = True
            cards.append(card)
        
        # POST-PROCESSING: Ensure verbs have conjugation tables
        final_cards = []
        i = 0
        while i < len(cards):
            card = cards[i]
            final_cards.append(card)
            
            # Check if this is a verb card that needs a conjugation table
            is_verb = (
                card.part_of_speech and card.part_of_speech.lower() == "verb"
                and card.type != "conjugation_table"
            )
            
            if is_verb:
                # Check if next card is already a conjugation table for this verb
                has_following_conjugation = (
                    i + 1 < len(cards)
                    and cards[i + 1].type == "conjugation_table"
                    and cards[i + 1].infinitive 
                    and _normalize_word(cards[i + 1].infinitive, language) == _normalize_word(card.word or "", language)
                )
                
                if not has_following_conjugation:
                    # Generate a conjugation table card for this verb via quick API call
                    logger.warning(f"Verb '{card.word}' missing conjugation table - generating via API")
                    conjugation_card = _generate_conjugation_table(
                        verb=card.word or "",
                        translation=card.translation or "",
                        language=language,
                        proficiency=proficiency,
                        is_new=card.is_new,
                        is_review=card.is_review,
                    )
                    if conjugation_card:
                        final_cards.append(conjugation_card)
            
            i += 1
        
        logger.success(f"Batch complete: {len(final_cards)} {batch_type} cards (filtered {len(cards_data) - len(cards)} duplicates, added {len(final_cards) - len(cards)} conjugation tables)")
        return final_cards
        
    except Exception as e:
        logger.api_error(f"Teaching batch failed: {e}")
        return []


def generate_teaching_content(
    language: str,
    proficiency: str,
    learner_context: Optional[str] = None,
    num_new_words: int = 12,
    num_review_words: int = 5,
    theme: Optional[str] = None,
    on_batch_ready: Optional[Callable[[List[TeachingCard], int, int], None]] = None,
) -> TeachingPlan:
    """
    Generate teaching content (vocabulary, grammar, phrases) for a lesson.
    
    This creates educational cards that TEACH new content before the quiz.
    The cards have multimedia support (images, audio) and clear explanations.
    
    If on_batch_ready is provided, generates content in batches and calls the callback
    after each batch for progressive UI updates.
    
    Args:
        language: Target language (e.g., "German")
        proficiency: CEFR level (A1, A2, B1, B2, C1, C2)
        learner_context: User's learning history from database
        num_new_words: How many new words to introduce (default 12)
        num_review_words: How many known words to review (default 5)
        theme: Optional theme for the lesson (e.g., "Food", "Travel")
        on_batch_ready: Optional callback(cards, batch_num, total_batches) for progressive loading
    
    Returns:
        TeachingPlan with vocabulary, grammar, and phrase cards
    """
    logger.separator(f"Generating Teaching Content ({language}, {proficiency})")
    logger.api(f"generate_teaching_content() - {num_new_words} new, {num_review_words} review")
    
    if client is None:
        logger.warning("No API client - using fallback teaching content")
        return _teaching_content_fallback(language, proficiency)
    
    # Use batched generation for progressive loading
    if on_batch_ready:
        return _generate_teaching_content_batched(
            language, proficiency, learner_context,
            num_new_words, num_review_words, theme, on_batch_ready
        )
    
    # Original single-call approach (still available for backward compatibility)
    try:
        # Determine appropriate content complexity based on proficiency
        content_guidance = {
            "A1": "Very basic vocabulary: colors, numbers 1-10, greetings, family members, common objects. Simple present tense only.",
            "A2": "Everyday vocabulary: food, weather, hobbies, basic verbs. Present and simple past tenses.",
            "B1": "Intermediate vocabulary: emotions, opinions, travel, work. Past, present, future tenses. Introduction to subjunctive.",
            "B2": "Advanced vocabulary: abstract concepts, idioms, business terms. Complex tenses, conditional mood.",
            "C1": "Sophisticated vocabulary: nuanced expressions, formal language, specialized terms. All tenses and moods.",
            "C2": "Near-native vocabulary: rare words, literary expressions, subtle distinctions. Stylistic variation.",
        }
        
        level_guidance = content_guidance.get(proficiency, content_guidance["A1"])
        
        system_prompt = f"""You are an expert {language} language teacher creating educational content.

Your task is to generate TEACHING cards that introduce and explain new vocabulary, grammar, and phrases.
These cards will be shown to learners BEFORE they are tested - this is the learning phase.

TARGET LEVEL: {proficiency}
LEVEL GUIDANCE: {level_guidance}

{TEACHING_CARD_SCHEMA}

REQUIREMENTS:
1. Generate {num_new_words} NEW vocabulary/phrase cards (is_new: true)
2. Generate {num_review_words} REVIEW cards (is_review: true) - common words at this level that reinforce learning
3. Include 2-3 GRAMMAR cards that explain key grammar points at this level
4. Total cards should be around {num_new_words + num_review_words + 3}

CONTENT GUIDELINES FOR {proficiency}:
- All explanations in ENGLISH (clear and helpful)
- All examples and target content in {language}
- Vocabulary appropriate for {proficiency} level
- Grammar concepts appropriate for {proficiency} level

CARD DESIGN:
- Each card should be self-contained and educational
- Include image_prompt for vocabulary cards (simple, clear images)
- Include audio_text for TTS pronunciation (the word/phrase in {language})
- Include mnemonic or memory aids where helpful
- Show related words to build vocabulary networks

{f'THEME: Focus on vocabulary related to "{theme}"' if theme else 'Choose a cohesive theme (e.g., food, daily routine, travel)'}

{learner_context if learner_context else ''}
{f"PERSONALIZATION: Use the learner context above to include words they need to review (weak vocabulary) and avoid words they've mastered." if learner_context else ""}

Return ONLY a JSON object:
{{
  "theme": "the theme of this lesson",
  "teaching_cards": [
    {{
      "type": "vocabulary_intro",
      "title": "New Word: der Hund",
      "word": "der Hund",
      "translation": "the dog",
      "part_of_speech": "noun",
      "gender": "masculine",
      "explanation": "This is a common noun for a dog. In German, all nouns are capitalized and have a gender...",
      "example_sentence": "Der Hund spielt im Garten.",
      "example_translation": "The dog plays in the garden.",
      "pronunciation_hint": "hoont (the 'u' sounds like 'oo' in 'book')",
      "mnemonic": "Think of a dog 'hounding' you for treats!",
      "image_prompt": "a friendly golden retriever dog sitting",
      "audio_text": "der Hund",
      "related_words": ["die Katze (cat)", "das Tier (animal)"],
      "is_new": true,
      "is_review": false,
      "difficulty_level": "A1"
    }},
    // ... more cards
  ]
}}"""

        user_content = f"""Generate teaching content for a {proficiency} level {language} learner.

Create:
- {num_new_words} new vocabulary/phrase cards
- {num_review_words} review cards  
- 2-3 grammar lesson cards

Make the content engaging, educational, and appropriate for the {proficiency} level.
{f'Focus on the theme: {theme}' if theme else 'Choose a cohesive theme like food, daily activities, or travel.'}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        
        logger.api_call("chat.completions.create (teaching_content)", model=DEFAULT_CHAT_MODEL)
        with Timer() as timer:
            completion = client.chat.completions.create(
                model=DEFAULT_CHAT_MODEL,
                response_format={"type": "json_object"},
                messages=messages,
                temperature=0.7,
                max_tokens=6000,  # Teaching content needs more tokens
            )
        logger.api_response("chat.completions.create (teaching)", duration_ms=timer.duration_ms)
        
        raw = completion.choices[0].message.content
        data = json.loads(raw)
        
        # Parse teaching cards
        cards_data = data.get("teaching_cards", [])
        theme = data.get("theme", "General Vocabulary")
        
        cards = []
        new_count = 0
        review_count = 0
        grammar_concepts = []
        
        for card_data in cards_data:
            card = _json_to_teaching_card(card_data)
            cards.append(card)
            
            if card.is_new:
                new_count += 1
            if card.is_review:
                review_count += 1
            if card.type == "grammar_lesson" and card.grammar_rule:
                grammar_concepts.append(card.grammar_rule)
        
        logger.success(f"Generated {len(cards)} teaching cards: {new_count} new, {review_count} review, {len(grammar_concepts)} grammar")
        
        return TeachingPlan(
            cards=cards,
            proficiency_target=proficiency,
            theme=theme,
            new_words_count=new_count,
            review_words_count=review_count,
            grammar_concepts=grammar_concepts,
        )
    
    except Exception as e:
        logger.api_error(f"Teaching content generation failed: {e}", exc_info=True)
        return _teaching_content_fallback(language, proficiency)


def _generate_teaching_content_batched(
    language: str,
    proficiency: str,
    learner_context: Optional[str],
    num_new_words: int,
    num_review_words: int,
    theme: Optional[str],
    on_batch_ready: Callable[[List[TeachingCard], int, int], None],
) -> TeachingPlan:
    """
    Generate teaching content in batches for progressive loading.
    
    Calls on_batch_ready after each batch so UI can update progressively.
    """
    logger.api("Using batched generation for progressive loading")
    
    all_cards: List[TeachingCard] = []
    
    # Pre-populate with known words from learner context to avoid duplicates
    generated_words: List[str] = []
    if learner_context:
        # Extract known words from learner context string
        # Look for the "WORDS ALREADY KNOWN" section
        if "WORDS ALREADY KNOWN" in learner_context:
            try:
                # Find the line with the words
                lines = learner_context.split('\n')
                for i, line in enumerate(lines):
                    if "WORDS ALREADY KNOWN" in line:
                        # Next line(s) contain the words
                        if i + 1 < len(lines):
                            words_line = lines[i + 1].strip()
                            if words_line:
                                known_words = [w.strip() for w in words_line.split(',') if w.strip()]
                                generated_words.extend(known_words)
                                logger.debug(f"Pre-loaded {len(known_words)} known words to avoid duplicates")
                        break
            except Exception as e:
                logger.warning(f"Could not parse known words from context: {e}")
    
    # Calculate batches: vocab(6), vocab(6), grammar(3), review(5)
    batches = [
        ("vocabulary", min(6, num_new_words)),
        ("vocabulary", max(0, num_new_words - 6)),
        ("grammar", 3),
        ("review", num_review_words),
    ]
    # Filter out empty batches
    batches = [(t, n) for t, n in batches if n > 0]
    total_batches = len(batches)
    
    discovered_theme = theme or "Daily Life"
    
    logger.debug(f"Starting batch generation with {len(generated_words)} forbidden words")
    
    for batch_num, (batch_type, num_cards) in enumerate(batches, 1):
        logger.api(f"Generating batch {batch_num}/{total_batches}: {batch_type} ({num_cards} cards), avoiding {len(generated_words)} words")
        
        cards = generate_teaching_batch(
            language=language,
            proficiency=proficiency,
            batch_type=batch_type,
            num_cards=num_cards,
            theme=discovered_theme,
            learner_context=learner_context,  # Always pass learner context to avoid duplicates
            existing_words=generated_words,
        )
        
        # Track words to avoid duplicates (including verb infinitives)
        # Track both original and normalized forms for robust duplicate detection
        for card in cards:
            if card.word:
                generated_words.append(card.word)
                # Also add normalized version for matching
                normalized = _normalize_word(card.word, language)
                if normalized and normalized not in generated_words:
                    generated_words.append(normalized)
            # Also track infinitives from conjugation tables
            if card.infinitive:
                if card.infinitive not in generated_words:
                    generated_words.append(card.infinitive)
                normalized_inf = _normalize_word(card.infinitive, language)
                if normalized_inf and normalized_inf not in generated_words:
                    generated_words.append(normalized_inf)
        
        all_cards.extend(cards)
        
        # Call the callback with progress
        logger.ui(f"Batch {batch_num}/{total_batches} ready: {len(cards)} {batch_type} cards")
        on_batch_ready(cards, batch_num, total_batches)
    
    # Count totals
    new_count = sum(1 for c in all_cards if c.is_new)
    review_count = sum(1 for c in all_cards if c.is_review)
    grammar_concepts = [c.grammar_rule for c in all_cards if c.type == "grammar_lesson" and c.grammar_rule]
    
    logger.success(f"Batched generation complete: {len(all_cards)} cards total")
    
    return TeachingPlan(
        cards=all_cards,
        proficiency_target=proficiency,
        theme=discovered_theme,
        new_words_count=new_count,
        review_words_count=review_count,
        grammar_concepts=grammar_concepts,
    )


def _json_to_teaching_card(card_data: Dict[str, Any]) -> TeachingCard:
    """Convert JSON data to TeachingCard object."""
    card_type = card_data.get("type", "vocabulary_intro")
    
    # Sanitize image prompt if present
    image_prompt = card_data.get("image_prompt")
    if image_prompt:
        image_prompt = sanitize_image_prompt(image_prompt)
    
    # Parse grammar examples if present
    grammar_examples = card_data.get("grammar_examples", []) or []
    
    # Parse additional examples if present
    additional_examples = card_data.get("additional_examples", []) or []
    
    # Parse conjugation examples if present
    conjugation_examples = card_data.get("conjugation_examples", []) or []
    
    # Parse conjugations dictionary
    conjugations = card_data.get("conjugations", {}) or {}
    
    return TeachingCard(
        type=card_type,
        title=card_data.get("title", ""),
        explanation=card_data.get("explanation", ""),
        word=card_data.get("word") or card_data.get("infinitive"),  # Support both
        translation=card_data.get("translation"),
        pronunciation_hint=card_data.get("pronunciation_hint"),
        part_of_speech=card_data.get("part_of_speech"),
        gender=card_data.get("gender"),
        plural_form=card_data.get("plural_form"),
        # Conjugation fields
        infinitive=card_data.get("infinitive"),
        conjugations=conjugations,
        tense=card_data.get("tense"),
        verb_type=card_data.get("verb_type"),
        conjugation_examples=conjugation_examples,
        # Examples
        example_sentence=card_data.get("example_sentence"),
        example_translation=card_data.get("example_translation"),
        additional_examples=additional_examples,
        grammar_rule=card_data.get("grammar_rule"),
        grammar_examples=grammar_examples,
        common_mistakes=card_data.get("common_mistakes", []) or [],
        related_words=card_data.get("related_words", []) or [],
        synonyms=card_data.get("synonyms", []) or [],
        antonyms=card_data.get("antonyms", []) or [],
        image_prompt=image_prompt,
        audio_text=card_data.get("audio_text"),
        is_review=card_data.get("is_review", False),
        is_new=card_data.get("is_new", True),
        difficulty_level=card_data.get("difficulty_level", "A1"),
        mnemonic=card_data.get("mnemonic"),
        usage_notes=card_data.get("usage_notes"),
    )


def _teaching_content_fallback(language: str, proficiency: str) -> TeachingPlan:
    """Fallback teaching content when API is unavailable."""
    logger.warning("Using fallback teaching content")
    
    # Language-specific examples
    examples = {
        "German": [
            TeachingCard(
                type="vocabulary_intro",
                title="New Word: der Apfel",
                word="der Apfel",
                translation="the apple",
                part_of_speech="noun",
                gender="masculine",
                explanation="'Apfel' is a masculine noun in German. Like most fruits, it follows regular declension patterns.",
                example_sentence="Der Apfel ist rot.",
                example_translation="The apple is red.",
                pronunciation_hint="AHP-fell",
                mnemonic="Think of 'apple' - they sound similar!",
                image_prompt="a shiny red apple on a white surface",
                audio_text="der Apfel",
                related_words=["die Birne (pear)", "die Orange (orange)"],
                is_new=True,
                difficulty_level=proficiency,
            ),
            TeachingCard(
                type="grammar_lesson",
                title="Grammar: German Articles",
                grammar_rule="German nouns have three genders: masculine (der), feminine (die), and neuter (das)",
                explanation="Unlike English, German nouns have grammatical gender. You must learn the article with each noun.",
                grammar_examples=[
                    {"target": "der Mann", "english": "the man (masculine)"},
                    {"target": "die Frau", "english": "the woman (feminine)"},
                    {"target": "das Kind", "english": "the child (neuter)"},
                ],
                common_mistakes=["Forgetting to learn the article with the noun", "Assuming gender matches meaning"],
                audio_text="der, die, das",
                is_new=True,
                difficulty_level=proficiency,
            ),
        ],
        "Spanish": [
            TeachingCard(
                type="vocabulary_intro",
                title="New Word: la manzana",
                word="la manzana",
                translation="the apple",
                part_of_speech="noun",
                gender="feminine",
                explanation="'Manzana' is a feminine noun in Spanish. Most nouns ending in -a are feminine.",
                example_sentence="La manzana es roja.",
                example_translation="The apple is red.",
                pronunciation_hint="man-SAH-nah",
                image_prompt="a shiny red apple on a white surface",
                audio_text="la manzana",
                related_words=["la pera (pear)", "la naranja (orange)"],
                is_new=True,
                difficulty_level=proficiency,
            ),
        ],
    }
    
    cards = examples.get(language, examples.get("Spanish", []))
    
    return TeachingPlan(
        cards=cards,
        proficiency_target=proficiency,
        theme="Basic Vocabulary",
        new_words_count=len([c for c in cards if c.is_new]),
        review_words_count=0,
        grammar_concepts=["German Articles"] if language == "German" else [],
    )


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
    
    # Audio transcription: compare user's transcription to the original audio text
    if card.type == "audio_transcription" and user_response:
        logger.debug(f"Audio transcription evaluation, response length: {len(user_response)}")
        return _evaluate_audio_transcription(card, user_response, language)
    
    # Audio comprehension: evaluate answer to comprehension question
    if card.type == "audio_comprehension" and user_response:
        logger.debug(f"Audio comprehension evaluation, response length: {len(user_response)}")
        return _evaluate_text_response_with_api(card, user_response, language)
    
    # Speaking: compare user's speech transcription to expected phrase
    if card.type == "speaking" and user_response:
        logger.debug(f"Speaking evaluation, transcription length: {len(user_response)}")
        return evaluate_speaking_response(card, user_response, language)
    
    # Reading comprehension: evaluate answers to reading questions
    if card.type == "reading_comprehension" and user_response:
        logger.debug(f"Reading comprehension evaluation")
        return _evaluate_reading_comprehension(card, user_response, language)
    
    # Writing practice: detailed writing feedback
    if card.type == "writing_practice" and user_response:
        logger.debug(f"Writing practice evaluation, response length: {len(user_response)}")
        return _evaluate_writing_practice(card, user_response, language)
    
    # Word order: compare user's word ordering to correct order
    if card.type == "word_order" and user_response:
        logger.debug(f"Word order evaluation")
        return _evaluate_word_order(card, user_response, language)
    
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


def _evaluate_audio_transcription(
    card: LessonCard,
    user_response: str,
    language: str
) -> Dict[str, Any]:
    """
    Evaluate an audio transcription response.
    Uses API to check similarity accounting for minor spelling variations.
    """
    if not client:
        # Simple fallback: exact match check
        is_correct = user_response.strip().lower() == (card.audio_text or "").strip().lower()
        return {
            "is_correct": is_correct,
            "card_score": 100 if is_correct else 50,
            "feedback": "Correct!" if is_correct else f"The correct transcription was: {card.audio_text}",
            "correct_answer": card.audio_text or "",
            "alternatives": card.alternatives or [],
            "vocabulary_expansion": card.vocabulary_expansion or [],
        }
    
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a language tutor evaluating a student's audio transcription. "
                    "Compare the student's transcription to the original text. "
                    "Be lenient with minor spelling errors, punctuation, and capitalization. "
                    "Focus on whether the student correctly heard and understood the audio.\n\n"
                    "Return ONLY a JSON object:\n"
                    "{\n"
                    '  "is_correct": true/false (true if substantially correct),\n'
                    '  "card_score": 0-100 (100=perfect, 80-99=minor errors, 50-79=partial, 0-49=mostly wrong),\n'
                    '  "feedback": "Constructive feedback on their listening/transcription",\n'
                    '  "errors": ["List of specific errors if any"],\n'
                    '  "vocabulary_expansion": ["Key vocabulary from the transcription"]\n'
                    "}"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "target_language": language,
                        "original_text": card.audio_text or "",
                        "user_transcription": user_response,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        
        logger.api_call("chat.completions.create (transcription evaluation)", model=DEFAULT_CHAT_MODEL)
        with Timer() as timer:
            completion = client.chat.completions.create(
                model=DEFAULT_CHAT_MODEL,
                response_format={"type": "json_object"},
                messages=messages,
                temperature=0.3,
                max_tokens=500,
            )
        logger.api_response("chat.completions.create", duration_ms=timer.duration_ms)
        
        raw = completion.choices[0].message.content
        data = json.loads(raw)
        
        return {
            "is_correct": data.get("is_correct", False),
            "card_score": int(data.get("card_score", 0)),
            "feedback": data.get("feedback", "Check your transcription."),
            "correct_answer": card.audio_text or "",
            "alternatives": card.alternatives or [],
            "vocabulary_expansion": data.get("vocabulary_expansion", []) or card.vocabulary_expansion or [],
        }
        
    except Exception as e:
        logger.error(f"Transcription evaluation failed: {e}")
        # Fallback to simple comparison
        is_correct = user_response.strip().lower() == (card.audio_text or "").strip().lower()
        return {
            "is_correct": is_correct,
            "card_score": 100 if is_correct else 50,
            "feedback": "Correct!" if is_correct else f"The correct transcription was: {card.audio_text}",
            "correct_answer": card.audio_text or "",
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


def _evaluate_word_order(
    card: LessonCard,
    user_response: str,
    language: str
) -> Dict[str, Any]:
    """
    Evaluate word ordering exercise.
    user_response is the user's ordered words as a space-separated string.
    """
    # Parse user's word order
    user_words = user_response.strip().split()
    correct_words = card.correct_word_order or []
    correct_sentence = card.correct_answer or " ".join(correct_words)
    
    # Check if exactly correct
    is_exact_match = user_words == correct_words
    
    # Calculate partial score based on correct positions
    if not correct_words:
        return {
            "is_correct": False,
            "card_score": 0,
            "feedback": "No correct answer provided for this card.",
            "correct_answer": correct_sentence,
            "alternatives": card.alternatives or [],
            "vocabulary_expansion": card.vocabulary_expansion or [],
        }
    
    # Count words in correct position
    correct_positions = 0
    total_positions = len(correct_words)
    
    for i, word in enumerate(user_words):
        if i < len(correct_words) and word.lower() == correct_words[i].lower():
            correct_positions += 1
    
    # Penalize for wrong number of words
    length_penalty = abs(len(user_words) - len(correct_words)) * 10
    
    # Calculate score
    position_score = (correct_positions / total_positions * 100) if total_positions > 0 else 0
    card_score = max(0, int(position_score - length_penalty))
    
    # Consider it correct if exact match or very close (90%+ and same length)
    is_correct = is_exact_match or (card_score >= 90 and len(user_words) == len(correct_words))
    
    # Generate feedback
    if is_correct:
        feedback = f"Perfect! '{correct_sentence}' is correct."
    elif card_score >= 70:
        feedback = f"Almost! The correct order is: '{correct_sentence}'"
    elif card_score >= 40:
        # Find specific errors
        errors = []
        for i, (user_word, correct_word) in enumerate(zip(user_words, correct_words)):
            if user_word.lower() != correct_word.lower():
                errors.append(f"Position {i+1}: '{user_word}' should be '{correct_word}'")
        error_hint = "; ".join(errors[:2]) if errors else ""
        feedback = f"Keep trying! {error_hint}. Correct: '{correct_sentence}'"
    else:
        feedback = f"The correct sentence is: '{correct_sentence}'. Study the word order!"
    
    return {
        "is_correct": is_correct,
        "card_score": 100 if is_correct else card_score,
        "feedback": feedback,
        "correct_answer": correct_sentence,
        "alternatives": card.alternatives or [],
        "vocabulary_expansion": card.vocabulary_expansion or [],
    }


def _evaluate_card_response_fallback(
    card: LessonCard,
    user_response: str,
    user_answer_index: Optional[int]
) -> Dict[str, Any]:
    """Fallback evaluation when API is unavailable."""
    is_correct = False
    if card.type == "multiple_choice" and user_answer_index is not None:
        is_correct = (user_answer_index == card.correct_index)
    elif card.type == "word_order":
        # Use word order evaluation even in fallback
        result = _evaluate_word_order(card, user_response, "")
        return result
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


def _evaluate_reading_comprehension(
    card: LessonCard,
    user_response: str,
    language: str
) -> Dict[str, Any]:
    """
    Evaluate reading comprehension answers.
    The user_response contains answers to reading questions (may be multiple).
    """
    if not client:
        return _evaluate_card_response_fallback(card, user_response, None)
    
    try:
        # Build context from the reading passage and questions
        passage = card.reading_passage or ""
        questions = card.reading_questions or []
        proficiency = card.difficulty_level or "A1"
        
        messages = [
            {
                "role": "system",
                "content": f"""You are a {language} language tutor evaluating reading comprehension answers.

LEARNER LEVEL: {proficiency} (evaluate appropriately for this level)

The student read a passage and answered questions about it. Evaluate their understanding.

PASSAGE:
{passage}

QUESTIONS:
{json.dumps(questions, indent=2)}

Evaluate the student's answers for:
1. Comprehension of the passage (considering their {proficiency} level)
2. Accuracy of answers
3. Language quality if answers are in {language}

SCORING GUIDELINES for {proficiency}:
- A1-A2: Focus on basic understanding, accept simple answers
- B1-B2: Expect more detail and inference
- C1-C2: Expect nuanced understanding and sophisticated responses

Return ONLY a JSON object:
{{
    "is_correct": true if comprehension is good FOR THEIR LEVEL,
    "card_score": 0-100 based on comprehension (calibrated for {proficiency}),
    "feedback": "Detailed feedback on comprehension and answers",
    "correct_answer": "The correct answers summarized",
    "alternatives": [],
    "vocabulary_expansion": ["Key vocabulary from passage"],
    "comprehension_details": {{
        "main_idea_understood": true/false,
        "details_understood": true/false,
        "inference_ability": "good/fair/poor"
    }}
}}"""
            },
            {
                "role": "user",
                "content": f"Student's answers:\n{user_response}"
            }
        ]
        
        logger.api_call("chat.completions.create (reading_comprehension)", model=DEFAULT_CHAT_MODEL)
        with Timer() as timer:
            completion = client.chat.completions.create(
                model=DEFAULT_CHAT_MODEL,
                response_format={"type": "json_object"},
                messages=messages,
                temperature=0.3,
                max_tokens=800,
            )
        logger.api_response("reading_comprehension", duration_ms=timer.duration_ms)
        
        raw = completion.choices[0].message.content
        data = json.loads(raw)
        
        return {
            "is_correct": data.get("is_correct", False),
            "card_score": data.get("card_score", 50),
            "feedback": data.get("feedback", "Review your answers."),
            "correct_answer": data.get("correct_answer", ""),
            "alternatives": data.get("alternatives", []),
            "vocabulary_expansion": data.get("vocabulary_expansion", []),
        }
        
    except Exception as e:
        logger.api_error(f"Reading comprehension evaluation failed: {e}")
        return _evaluate_card_response_fallback(card, user_response, None)


def _evaluate_writing_practice(
    card: LessonCard,
    user_response: str,
    language: str
) -> Dict[str, Any]:
    """
    Evaluate writing practice with detailed feedback on grammar, vocabulary, style.
    Returns comprehensive feedback for language improvement.
    """
    if not client:
        return _evaluate_card_response_fallback(card, user_response, None)
    
    try:
        writing_prompt = card.writing_prompt or card.question or ""
        min_words = card.writing_min_words
        max_words = card.writing_max_words
        proficiency = card.difficulty_level or "A1"
        
        # Count words in response
        word_count = len(user_response.split())
        
        messages = [
            {
                "role": "system",
                "content": f"""You are an expert {language} language tutor providing detailed writing feedback.

LEARNER LEVEL: {proficiency}
WRITING PROMPT: {writing_prompt}
TARGET LENGTH: {min_words}-{max_words} words
ACTUAL LENGTH: {word_count} words

LEVEL-APPROPRIATE EXPECTATIONS for {proficiency}:
- A1: Simple sentences, basic vocabulary, spelling attempts acceptable
- A2: Short paragraphs, common words, minor grammar errors OK
- B1: Connected text, varied vocabulary, some complex sentences
- B2: Clear arguments, good range, mostly correct grammar
- C1: Sophisticated expression, precise vocabulary, complex structures
- C2: Near-native quality, subtle nuances, rare errors

Analyze the student's writing comprehensively (calibrated for {proficiency}):

1. GRAMMAR ANALYSIS
   - Identify specific grammar errors
   - Note correct grammar usage
   - Provide corrections with explanations

2. VOCABULARY ASSESSMENT
   - Evaluate vocabulary range and appropriateness
   - Suggest more advanced alternatives where applicable
   - Note any incorrect word usage

3. STRUCTURE & STYLE
   - Comment on sentence structure variety
   - Assess coherence and flow
   - Note any stylistic improvements

4. ERROR PATTERNS
   - Identify recurring error types for targeted practice
   - Categorize errors: gender_agreement, verb_conjugation, word_order, spelling, etc.

Return ONLY a JSON object:
{{
    "is_correct": true if writing demonstrates {proficiency}-level proficiency,
    "card_score": 0-100 based on quality FOR A {proficiency} LEARNER,
    "feedback": "Overall assessment and encouragement",
    "correct_answer": "A corrected version of their writing",
    "alternatives": [],
    "vocabulary_expansion": ["Advanced vocabulary suggestions"],
    "writing_feedback": {{
        "grammar_errors": [
            {{"error": "original text", "correction": "corrected text", "explanation": "why", "error_type": "category"}}
        ],
        "vocabulary_suggestions": [
            {{"original": "basic word", "advanced": "better alternative", "context": "when to use"}}
        ],
        "strengths": ["What they did well"],
        "areas_to_improve": ["What to focus on"],
        "style_notes": "Comments on writing style",
        "error_types": ["list of error category strings for tracking"]
    }}
}}"""
            },
            {
                "role": "user",
                "content": f"Student's writing:\n\n{user_response}"
            }
        ]
        
        logger.api_call("chat.completions.create (writing_practice)", model=DEFAULT_CHAT_MODEL)
        with Timer() as timer:
            completion = client.chat.completions.create(
                model=DEFAULT_CHAT_MODEL,
                response_format={"type": "json_object"},
                messages=messages,
                temperature=0.3,
                max_tokens=1500,
            )
        logger.api_response("writing_practice", duration_ms=timer.duration_ms)
        
        raw = completion.choices[0].message.content
        data = json.loads(raw)
        
        # Store detailed feedback in the card for UI display
        card.writing_feedback = data.get("writing_feedback", {})
        
        # Extract error types for tracking
        error_types = data.get("writing_feedback", {}).get("error_types", [])
        if error_types:
            card.error_types = error_types
        
        return {
            "is_correct": data.get("is_correct", False),
            "card_score": data.get("card_score", 50),
            "feedback": data.get("feedback", "Keep practicing!"),
            "correct_answer": data.get("correct_answer", ""),
            "alternatives": data.get("alternatives", []),
            "vocabulary_expansion": data.get("vocabulary_expansion", []),
        }
        
    except Exception as e:
        logger.api_error(f"Writing practice evaluation failed: {e}")
        return _evaluate_card_response_fallback(card, user_response, None)


def evaluate_quiz_performance(
    lesson_plan: LessonPlan,
    current_assessment: AssessmentResult,
    language: str,
    learner_context: Optional[str] = None,
) -> AssessmentResult:
    """
    Re-evaluate proficiency based on quiz performance AND full learning history.
    
    This performs a HOLISTIC evaluation considering:
    - Current quiz performance
    - All vocabulary learned and their strength ratings
    - Historical session performance
    - Grammar patterns mastered
    - Overall learning trajectory
    
    Proficiency can go UP or DOWN based on this comprehensive assessment.
    
    Args:
        lesson_plan: The completed quiz with scored cards
        current_assessment: The user's current assessment result
        language: Target language
        learner_context: Full learning history from database (vocabulary, sessions, etc.)
        
    Returns:
        Updated AssessmentResult with new fluency_score and potentially new proficiency
    """
    logger.separator("Evaluating Quiz Performance (Holistic)")
    logger.api(f"evaluate_quiz_performance() for {language}")
    
    if client is None or not lesson_plan.cards:
        logger.warning("No API or no cards - returning current assessment unchanged")
        return current_assessment
    
    try:
        # Collect quiz performance data (excluding skipped cards - STT/TTS exercises)
        cards_data = []
        skipped_count = 0
        for card in lesson_plan.cards:
            if card.skipped:
                skipped_count += 1
                continue  # Don't include skipped cards in evaluation
            cards_data.append({
                "question": card.question[:100] if card.question else "",
                "type": card.type,
                "difficulty": card.difficulty_level,
                "is_correct": card.is_correct,
                "score": card.card_score,
                "user_response": card.user_response[:100] if card.user_response else "",
            })
        
        if skipped_count > 0:
            logger.debug(f"Excluded {skipped_count} skipped cards from evaluation")
        
        # Calculate stats excluding skipped cards
        scored_cards = [c for c in lesson_plan.cards if not c.skipped]
        correct_count = sum(1 for c in scored_cards if c.is_correct)
        total_count = len(scored_cards)
        quiz_percentage = (correct_count / total_count * 100) if total_count > 0 else 0
        
        logger.debug(f"Quiz results: {correct_count}/{total_count} correct ({quiz_percentage:.1f}%), {skipped_count} skipped")
        if learner_context:
            logger.debug(f"Learner context: {len(learner_context)} chars of history data")
        
        # Build learner history section for holistic evaluation
        history_section = ""
        if learner_context:
            history_section = f"""
=== COMPLETE LEARNING HISTORY ===
{learner_context}
=== END LEARNING HISTORY ===

Use the learning history above to make a HOLISTIC assessment. Consider:
- How many words they've learned and their retention (strength ratings)
- Their performance trend across multiple sessions
- Which grammar patterns they've mastered vs struggling with
- Their overall learning trajectory (improving, plateauing, declining?)
"""
        
        system_prompt = f"""You are evaluating a {language} learner's proficiency based on their COMPLETE learning history and current quiz performance.

This is a HOLISTIC evaluation - consider ALL available data, not just the current quiz.

CURRENT PROFICIENCY:
- Overall Level: {current_assessment.proficiency}
- Vocabulary: {current_assessment.vocabulary_level}
- Grammar: {current_assessment.grammar_level}
- Fluency Score: {current_assessment.fluency_score}/100

TODAY'S QUIZ RESULTS:
- Questions: {total_count}
- Correct: {correct_count} ({quiz_percentage:.1f}%)
{history_section}

HOLISTIC SCORING GUIDELINES:

1. FLUENCY SCORE (0-100) - Based on overall competence:
   - Consider vocabulary breadth AND retention (mastered vs weak words)
   - Consider consistency across sessions (not just this quiz)
   - Consider grammar pattern mastery
   - Adjust by -15 to +15 based on trajectory:
     * Consistently improving with good retention: +10 to +15
     * Good quiz, stable performance: +5 to +10
     * Average performance: -5 to +5
     * Declining performance or poor retention: -10 to -5
     * Significantly struggling: -15 to -10

2. PROFICIENCY LEVEL (A1 → A2 → B1 → B2 → C1 → C2):
   - Upgrade if: Mastered 80%+ of current level vocabulary, strong grammar, consistent good performance
   - Downgrade if: Struggling with current level content across multiple sessions, poor retention
   - Look at the TREND, not just one quiz
   - A1→A2: Basic survival phrases + simple present tense mastered
   - A2→B1: Can discuss routine matters, past/future tenses, 1000+ words
   - B1→B2: Can discuss abstract topics, conditional, 2500+ words
   - B2→C1: Nuanced expression, complex grammar, 5000+ words
   - C1→C2: Near-native fluency, idioms, rare vocabulary

3. VOCABULARY vs GRAMMAR assessment:
   - If vocabulary strong but grammar weak: Higher vocab level, lower grammar level
   - These should be assessed INDEPENDENTLY based on the evidence

Return ONLY a JSON object:
{{
    "proficiency": "A1" or "A2" or "B1" or "B2" or "C1" or "C2",
    "vocabulary_level": "A1" to "C2",
    "grammar_level": "A1" to "C2",
    "fluency_score": 0-100 (holistically adjusted),
    "strengths": ["Updated strength 1", "Updated strength 2", "Updated strength 3"],
    "weaknesses": ["Updated weakness 1", "Updated weakness 2"],
    "recommendations": ["Specific recommendation 1", "Specific recommendation 2"],
    "performance_summary": "Brief explanation of the holistic assessment and trajectory"
}}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Today's quiz card details:\n{json.dumps(cards_data, indent=2)}"},
        ]
        
        logger.api_call("chat.completions.create (quiz_evaluation)", model=DEFAULT_CHAT_MODEL)
        with Timer() as timer:
            completion = client.chat.completions.create(
                model=DEFAULT_CHAT_MODEL,
                response_format={"type": "json_object"},
                messages=messages,
                temperature=0.3,
                max_tokens=1000,
            )
        logger.api_response("quiz_evaluation", duration_ms=timer.duration_ms)
        
        raw = completion.choices[0].message.content
        data = json.loads(raw)
        
        # Log the changes
        old_fluency = current_assessment.fluency_score
        new_fluency = data.get("fluency_score", old_fluency)
        old_prof = current_assessment.proficiency
        new_prof = data.get("proficiency", old_prof)
        
        logger.debug(f"Fluency: {old_fluency} → {new_fluency} (delta: {new_fluency - old_fluency:+d})")
        logger.debug(f"Proficiency: {old_prof} → {new_prof}")
        if data.get("performance_summary"):
            logger.debug(f"Summary: {data['performance_summary'][:100]}...")
        
        # Create updated assessment result
        updated_result = AssessmentResult(
            proficiency=data.get("proficiency", current_assessment.proficiency),
            vocabulary_level=data.get("vocabulary_level", current_assessment.vocabulary_level),
            grammar_level=data.get("grammar_level", current_assessment.grammar_level),
            fluency_score=data.get("fluency_score", current_assessment.fluency_score),
            strengths=data.get("strengths", current_assessment.strengths),
            weaknesses=data.get("weaknesses", current_assessment.weaknesses),
            recommendations=data.get("recommendations", current_assessment.recommendations),
        )
        
        logger.success(f"Quiz evaluation complete: {new_prof} ({new_fluency}/100)")
        return updated_result
        
    except Exception as e:
        logger.api_error(f"Quiz evaluation failed: {e}")
        return current_assessment


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

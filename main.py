"""
GAIT Language Buddy - PySimpleGUI Prototype (OpenAI-enabled)

How to run:

    python -m venv .venv
    source .venv/bin/activate   # or .venv\Scripts\activate on Windows
    pip install -r requirements.txt

    # Make sure .env exists at repo root with OPENAI_API_KEY set
    python main.py
"""

import PySimpleGUI as sg

from core.api import (
    SUPPORTED_LANGUAGES,
    generate_scene,
    evaluate_user_text,
    generate_mini_lesson,
    generate_audio,
)
from core.models import TextAnalysis, MiniLesson


def build_layout() -> list[list[sg.Element]]:
    """Define the PySimpleGUI window layout."""
    sg.theme("SystemDefault")

    language_row = [
        sg.Text("Target language:"),
        sg.Combo(
            SUPPORTED_LANGUAGES,
            default_value=SUPPORTED_LANGUAGES[0],
            key="-LANGUAGE-",
            readonly=True,
            size=(20, 1),
        ),
        sg.Button("New Scene", key="-NEW_SCENE-"),
    ]

    scene_frame = [
        [sg.Text("Scene description (placeholder for image):")],
        [
            sg.Multiline(
                "Click 'New Scene' to generate a scene description.\n"
                "(This will later be replaced by an image.)",
                key="-SCENE-",
                size=(60, 5),
                disabled=True,
                autoscroll=True,
            )
        ],
    ]

    input_frame = [
        [sg.Text("Your description in the target language:")],
        [
            sg.Multiline(
                "",
                key="-USER_TEXT-",
                size=(60, 8),
                autoscroll=True,
            )
        ],
        [sg.Button("Evaluate Writing", key="-EVALUATE-")],
    ]

    feedback_frame = [
        [sg.Text("Feedback & Proficiency:")],
        [
            sg.Multiline(
                "",
                key="-FEEDBACK-",
                size=(60, 10),
                disabled=True,
                autoscroll=True,
            )
        ],
    ]

    lesson_frame = [
        [sg.Text("Mini-lesson:")],
        [
            sg.Multiline(
                "",
                key="-LESSON-",
                size=(60, 10),
                disabled=True,
                autoscroll=True,
            )
        ],
        [
            sg.Button("Generate Audio", key="-AUDIO-"),
            sg.Text("", key="-AUDIO_STATUS-", size=(30, 1)),
        ],
    ]

    layout = [
        language_row,
        [sg.Frame("Scene", scene_frame, expand_x=True)],
        [sg.Frame("Your Writing", input_frame, expand_x=True)],
        [sg.Frame("Feedback", feedback_frame, expand_x=True)],
        [sg.Frame("Mini-Lesson", lesson_frame, expand_x=True)],
    ]

    return layout


def render_feedback(analysis: TextAnalysis, feedback_elem: sg.Multiline) -> None:
    """Format and display analysis feedback."""
    if analysis is None:
        feedback_elem.update("No analysis available.")
        return

    strengths = analysis.strengths or ["(none detected)"]
    weaknesses = analysis.weaknesses or ["(none detected)"]
    suggestions = analysis.suggestions or ["(none provided)"]

    lines = [
        f"Detected proficiency: {analysis.proficiency}",
        "",
        "Strengths:",
        *[f"  • {s}" for s in strengths],
        "",
        "Areas for improvement:",
        *[f"  • {w}" for w in weaknesses],
        "",
        "Suggestions:",
        *[f"  • {s}" for s in suggestions],
    ]
    feedback_elem.update("\n".join(lines))


def render_lesson(lesson: MiniLesson, lesson_elem: sg.Multiline) -> None:
    """Format and display mini-lesson."""
    if lesson is None:
        lesson_elem.update("No lesson generated yet.")
        return

    points = lesson.points or ["(no points)"]
    examples = lesson.examples or ["(no examples)"]
    vocab = lesson.vocabulary or ["(no vocabulary)"]

    lines = [
        "Mini-lesson focus:",
        *[f"  • {p}" for p in points],
        "",
        "Example sentences:",
        *[f"  • {ex}" for ex in examples],
        "",
        "Vocabulary:",
        *[f"  • {item}" for item in vocab],
    ]
    lesson_elem.update("\n".join(lines))


def main() -> None:
    window = sg.Window("GAIT Language Buddy (Prototype)", build_layout())
    current_analysis: TextAnalysis | None = None

    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break

        if event == "-NEW_SCENE-":
            language = values["-LANGUAGE-"]
            scene_text = generate_scene(language)
            window["-SCENE-"].update(scene_text)

        elif event == "-EVALUATE-":
            language = values["-LANGUAGE-"]
            user_text = (values["-USER_TEXT-"] or "").strip()

            if not user_text:
                sg.popup("Please write a description before evaluating.")
                continue

            analysis = evaluate_user_text(user_text, language)
            current_analysis = analysis
            render_feedback(analysis, window["-FEEDBACK-"])

            lesson = generate_mini_lesson(analysis)
            render_lesson(lesson, window["-LESSON-"])

        elif event == "-AUDIO-":
            if current_analysis is None:
                window["-AUDIO_STATUS-"].update("No analysis yet.")
                continue

            # In the future, you might cache the last MiniLesson;
            # for now we just regenerate it.
            lesson = generate_mini_lesson(current_analysis)
            audio_info = generate_audio(lesson.examples)
            window["-AUDIO_STATUS-"].update(
                f"Audio (stub): {audio_info.placeholder_path}"
            )

    window.close()


if __name__ == "__main__":
    main()

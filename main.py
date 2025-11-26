"""
GAIT Language Buddy - Tkinter (card-based) prototype

Flow:
1. Intro card: explanation + language selection.
2. Scene card: generate scene + image, show image, learner writes description.
3. Evaluation: compare learner text to scene.
4. Lesson card: show 10 lesson prompts one at a time.
5. Summary card: show score, ask if they want to go again.

Setup (from repo root):

    python -m venv .venv
    source .venv/bin/activate   # or .venv\\Scripts\\activate on Windows
    pip install -r requirements.txt

Ensure .env contains:
    OPENAI_API_KEY=sk-...

Then run:
    python main.py
"""

import tkinter as tk
from tkinter import ttk, messagebox

from typing import Optional

from PIL import Image, ImageTk

from core.api import (
    SUPPORTED_LANGUAGES,
    generate_scene,
    generate_scene_image,
    evaluate_user_text,
    generate_lesson_plan,
)
from core.models import SceneInfo, TextAnalysis, LessonPlan


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class LanguageBuddyApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()

        self.title("GAIT Language Buddy (Prototype)")
        self.geometry("700x800")

        # State
        self.selected_language: Optional[str] = None
        self.scene: Optional[SceneInfo] = None
        self.scene_image_path: Optional[str] = None
        self.scene_photo: Optional[ImageTk.PhotoImage] = None
        self.learner_text: str = ""
        self.analysis: Optional[TextAnalysis] = None
        self.lesson_plan: Optional[LessonPlan] = None
        self.lesson_index: int = 0

        # Container for "cards"
        container = ttk.Frame(self)
        container.pack(fill="both", expand=True)

        self.cards = {}

        for CardClass in (IntroCard, SceneCard, LessonCardView, SummaryCard):
            card = CardClass(parent=container, controller=self)
            self.cards[CardClass.__name__] = card
            card.grid(row=0, column=0, sticky="nsew")

        self.show_card("IntroCard")

    # Card management -------------------------------------------------------

    def show_card(self, name: str) -> None:
        card = self.cards[name]
        card.tkraise()

    # High-level flow methods ----------------------------------------------

    def start_session(self, language: str) -> None:
        """Triggered from Intro card when user picks a language."""
        self.selected_language = language
        self.scene = None
        self.analysis = None
        self.lesson_plan = None
        self.lesson_index = 0

        # Generate scene + image (blocking for now)
        self.scene = generate_scene(language)
        self.scene_image_path = generate_scene_image(self.scene.image_prompt)

        scene_card: SceneCard = self.cards["SceneCard"]
        scene_card.set_scene(self.scene, self.scene_image_path)
        self.show_card("SceneCard")

    def submit_learner_text(self, text: str) -> None:
        """Called by SceneCard when learner submits their description."""
        text = (text or "").strip()
        if not text:
            messagebox.showinfo("No text", "Please write a description before continuing.")
            return

        if self.scene is None:
            messagebox.showerror("Error", "No scene is loaded.")
            return

        self.learner_text = text

        # Evaluate and generate lesson plan (blocking calls)
        self.analysis = evaluate_user_text(
            text=self.learner_text,
            language=self.selected_language or "Spanish",
            scene_description=self.scene.scene_description,
        )
        self.lesson_plan = generate_lesson_plan(self.analysis, self.scene.scene_description)
        self.lesson_index = 0

        lesson_card: LessonCardView = self.cards["LessonCardView"]
        lesson_card.set_lesson(self.lesson_plan, self.analysis)
        self.show_card("LessonCardView")

    def next_lesson_card(self) -> None:
        """Move to the next lesson card or to summary when done."""
        if not self.lesson_plan:
            return

        self.lesson_index += 1
        if self.lesson_index >= len(self.lesson_plan.cards):
            # Done -> go to summary
            summary_card: SummaryCard = self.cards["SummaryCard"]
            summary_card.set_summary(self.analysis)
            self.show_card("SummaryCard")
        else:
            lesson_card: LessonCardView = self.cards["LessonCardView"]
            lesson_card.show_card_index(self.lesson_index)

    def restart(self) -> None:
        """Restart the whole experience from the intro."""
        self.selected_language = None
        self.scene = None
        self.scene_image_path = None
        self.scene_photo = None
        self.learner_text = ""
        self.analysis = None
        self.lesson_plan = None
        self.lesson_index = 0

        intro: IntroCard = self.cards["IntroCard"]
        intro.reset()
        self.show_card("IntroCard")


# ---------------------------------------------------------------------------
# Intro card
# ---------------------------------------------------------------------------

class IntroCard(ttk.Frame):
    def __init__(self, parent, controller: LanguageBuddyApp) -> None:
        super().__init__(parent)
        self.controller = controller

        self.columnconfigure(0, weight=1)

        title = ttk.Label(
            self,
            text="GAIT Language Buddy",
            font=("Helvetica", 20, "bold"),
            anchor="center",
        )
        title.grid(row=0, column=0, pady=(30, 10), padx=20, sticky="ew")

        desc_text = (
            "Welcome! This tool helps you practice descriptive writing in another language.\n\n"
            "You will:\n"
            "  1. Choose a target language.\n"
            "  2. See a complex scene image.\n"
            "  3. Describe the scene in your own words.\n"
            "  4. Receive feedback and work through 10 short lesson prompts.\n\n"
            "At the end, you'll see a score and can choose to try another scene."
        )
        desc = ttk.Label(self, text=desc_text, wraplength=600, justify="left")
        desc.grid(row=1, column=0, padx=40, pady=(0, 20), sticky="w")

        lang_frame = ttk.Frame(self)
        lang_frame.grid(row=2, column=0, padx=40, pady=(0, 20), sticky="w")

        ttk.Label(lang_frame, text="Choose your target language:").grid(
            row=0, column=0, sticky="w"
        )

        self.language_var = tk.StringVar()
        self.language_combo = ttk.Combobox(
            lang_frame,
            textvariable=self.language_var,
            values=SUPPORTED_LANGUAGES,
            state="readonly",
            width=20,
        )
        self.language_combo.grid(row=0, column=1, padx=(10, 0))

        start_button = ttk.Button(
            self,
            text="Start Session",
            command=self._on_start_clicked,
        )
        start_button.grid(row=3, column=0, pady=(10, 40))

    def _on_start_clicked(self) -> None:
        language = self.language_var.get()
        if not language:
            messagebox.showinfo("Choose a language", "Please select a target language to continue.")
            return

        # Start the session in the controller
        self.controller.start_session(language)

    def reset(self) -> None:
        self.language_var.set("")


# ---------------------------------------------------------------------------
# Scene + writing card
# ---------------------------------------------------------------------------

class SceneCard(ttk.Frame):
    def __init__(self, parent, controller: LanguageBuddyApp) -> None:
        super().__init__(parent)
        self.controller = controller

        self.columnconfigure(0, weight=1)

        self.title_label = ttk.Label(
            self,
            text="Describe the Scene",
            font=("Helvetica", 18, "bold"),
            anchor="center",
        )
        self.title_label.grid(row=0, column=0, pady=(20, 10), sticky="ew")

        self.instructions_label = ttk.Label(
            self,
            text="Look at the image and describe it in your own words.\n"
                 "Use as much detail as possible in the target language.",
            wraplength=650,
            justify="left",
        )
        self.instructions_label.grid(row=1, column=0, padx=20, pady=(0, 10), sticky="w")

        # Image area
        self.image_label = ttk.Label(self)
        self.image_label.grid(row=2, column=0, padx=20, pady=(0, 10))

        # Text area
        text_frame = ttk.Frame(self)
        text_frame.grid(row=3, column=0, padx=20, pady=(10, 10), sticky="nsew")
        self.rowconfigure(3, weight=1)

        ttk.Label(text_frame, text="Your description in the target language:").grid(
            row=0, column=0, sticky="w"
        )

        self.text_widget = tk.Text(text_frame, height=12, wrap="word")
        self.text_widget.grid(row=1, column=0, sticky="nsew", pady=(5, 0))
        text_frame.rowconfigure(1, weight=1)
        text_frame.columnconfigure(0, weight=1)

        scrollbar = ttk.Scrollbar(
            text_frame,
            orient="vertical",
            command=self.text_widget.yview,
        )
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.text_widget.configure(yscrollcommand=scrollbar.set)

        submit_button = ttk.Button(
            self,
            text="Submit Description",
            command=self._on_submit_clicked,
        )
        submit_button.grid(row=4, column=0, pady=(10, 20))

    def set_scene(self, scene: SceneInfo, image_path: Optional[str]) -> None:
        """Set the current scene and image on this card."""
        lang = scene.language
        self.title_label.configure(text=f"Describe the Scene ({lang})")

        # Load and display image if available
        if image_path:
            try:
                img = Image.open(image_path)
                img.thumbnail((600, 400))
                photo = ImageTk.PhotoImage(img)
                # Keep reference on controller to prevent GC
                self.controller.scene_photo = photo
                self.image_label.configure(image=photo)
            except Exception:
                self.image_label.configure(
                    text="[Could not display image, but the scene is loaded.]"
                )
        else:
            self.image_label.configure(
                text="[Image generation is unavailable. Imagine the scene described in the lesson.]"
            )

        # Clear previous text
        self.text_widget.delete("1.0", tk.END)

    def _on_submit_clicked(self) -> None:
        text = self.text_widget.get("1.0", tk.END)
        self.controller.submit_learner_text(text)


# ---------------------------------------------------------------------------
# Lesson card (10 prompts, one at a time)
# ---------------------------------------------------------------------------

class LessonCardView(ttk.Frame):
    def __init__(self, parent, controller: LanguageBuddyApp) -> None:
        super().__init__(parent)
        self.controller = controller

        self.columnconfigure(0, weight=1)

        self.title_label = ttk.Label(
            self,
            text="Lesson Prompts",
            font=("Helvetica", 18, "bold"),
        )
        self.title_label.grid(row=0, column=0, pady=(20, 10))

        self.subtitle_label = ttk.Label(
            self,
            text="Work through the prompts one by one.",
            wraplength=650,
            justify="left",
        )
        self.subtitle_label.grid(row=1, column=0, padx=20, pady=(0, 10), sticky="w")

        self.progress_label = ttk.Label(self, text="")
        self.progress_label.grid(row=2, column=0, pady=(0, 10))

        # Lesson card text
        self.card_text = tk.Text(self, height=10, wrap="word", state="disabled")
        self.card_text.grid(row=3, column=0, padx=20, pady=(0, 10), sticky="nsew")
        self.rowconfigure(3, weight=1)

        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.card_text.yview)
        scrollbar.grid(row=3, column=1, sticky="ns")
        self.card_text.configure(yscrollcommand=scrollbar.set)

        self.next_button = ttk.Button(
            self, text="Next", command=self._on_next_clicked
        )
        self.next_button.grid(row=4, column=0, pady=(10, 20))

        self.analysis_label = ttk.Label(self, text="", wraplength=650, justify="left")
        self.analysis_label.grid(row=5, column=0, padx=20, pady=(0, 20), sticky="w")

    def set_lesson(self, lesson_plan: LessonPlan, analysis: Optional[TextAnalysis]) -> None:
        if not lesson_plan.cards:
            # Very defensive; should not happen.
            lesson_plan.cards = ["No lesson cards were generated. Please try again."]
        self.controller.lesson_plan = lesson_plan
        self.show_card_index(0)

        if analysis:
            self.analysis_label.configure(
                text=f"Estimated proficiency: {analysis.proficiency} | Score: {analysis.score}/100"
            )
        else:
            self.analysis_label.configure(text="")

    def show_card_index(self, index: int) -> None:
        plan = self.controller.lesson_plan
        if not plan:
            return

        total = len(plan.cards)
        index = max(0, min(index, total - 1))
        self.controller.lesson_index = index

        card_text = plan.cards[index]

        self.card_text.configure(state="normal")
        self.card_text.delete("1.0", tk.END)
        self.card_text.insert("1.0", card_text)
        self.card_text.configure(state="disabled")

        self.progress_label.configure(text=f"Card {index + 1} of {total}")

        if index == total - 1:
            self.next_button.configure(text="Finish Lesson")
        else:
            self.next_button.configure(text="Next")

    def _on_next_clicked(self) -> None:
        self.controller.next_lesson_card()


# ---------------------------------------------------------------------------
# Summary card
# ---------------------------------------------------------------------------

class SummaryCard(ttk.Frame):
    def __init__(self, parent, controller: LanguageBuddyApp) -> None:
        super().__init__(parent)
        self.controller = controller

        self.columnconfigure(0, weight=1)

        title = ttk.Label(
            self,
            text="Session Summary",
            font=("Helvetica", 18, "bold"),
        )
        title.grid(row=0, column=0, pady=(20, 10))

        self.summary_label = ttk.Label(self, text="", wraplength=650, justify="left")
        self.summary_label.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="w")

        button_frame = ttk.Frame(self)
        button_frame.grid(row=2, column=0, pady=(10, 30))

        again_button = ttk.Button(
            button_frame,
            text="Try another scene",
            command=self.controller.restart,
        )
        again_button.grid(row=0, column=0, padx=10)

        exit_button = ttk.Button(
            button_frame,
            text="Exit",
            command=self.controller.destroy,
        )
        exit_button.grid(row=0, column=1, padx=10)

    def set_summary(self, analysis: Optional[TextAnalysis]) -> None:
        if not analysis:
            text = "No analysis is available for this session."
        else:
            strengths = analysis.strengths or ["(none listed)"]
            weaknesses = analysis.weaknesses or ["(none listed)"]
            suggestions = analysis.suggestions or ["(none listed)"]

            text_lines = [
                f"Overall score: {analysis.score}/100",
                f"Estimated proficiency: {analysis.proficiency}",
                "",
                "Strengths:",
                *[f"  • {s}" for s in strengths],
                "",
                "Areas to improve:",
                *[f"  • {w}" for w in weaknesses],
                "",
                "Next steps:",
                *[f"  • {s}" for s in suggestions],
                "",
                "Would you like to try another scene?",
            ]
            text = "\n".join(text_lines)

        self.summary_label.configure(text=text)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = LanguageBuddyApp()
    app.mainloop()

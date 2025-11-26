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
import threading

from typing import Optional, List, Dict, Any, Set

from PIL import Image, ImageTk

from core.api import (
    SUPPORTED_LANGUAGES,
    generate_assessment_cards,
    evaluate_assessment_responses,
    generate_structured_lesson_plan,
    generate_lesson_plan_from_assessment_responses,  # Optimized combined function
    evaluate_card_response,
    generate_final_summary,
    generate_image_async,
    generate_images_parallel,  # Parallel image generation
)
from core.models import (
    AssessmentCard, AssessmentResult, LessonPlan, LessonCard
)


# ---------------------------------------------------------------------------
# Loading Spinner Widget
# ---------------------------------------------------------------------------

class LoadingSpinner(ttk.Frame):
    """A simple animated loading spinner widget for Tkinter."""
    
    def __init__(self, parent, text: str = "Loading...") -> None:
        super().__init__(parent)
        
        self.text = text
        self.spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.spinner_index = 0
        self.is_running = False
        self._after_id = None
        
        # Use a larger, more visible font with light blue on dark background
        self.label = ttk.Label(
            self, 
            text=f"{self.spinner_chars[0]} {text}", 
            font=("Helvetica", 14),
            foreground="#7bb3ff"  # Light blue color for loading text on dark background
        )
        self.label.pack(pady=20)
        
        # Start hidden by default
        self.grid_remove()
    
    def start(self) -> None:
        """Start the spinner animation."""
        if self.is_running:
            return
        
        self.is_running = True
        self.grid()
        self._animate()
    
    def stop(self) -> None:
        """Stop the spinner animation."""
        self.is_running = False
        if self._after_id:
            self.after_cancel(self._after_id)
            self._after_id = None
        self.grid_remove()
    
    def _animate(self) -> None:
        """Animate the spinner by rotating through characters."""
        if not self.is_running:
            return
        
        char = self.spinner_chars[self.spinner_index]
        self.label.configure(text=f"{char} {self.text}")
        self.spinner_index = (self.spinner_index + 1) % len(self.spinner_chars)
        
        self._after_id = self.after(100, self._animate)  # Update every 100ms


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class LanguageBuddyApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()

        self.title("GAIT Language Buddy")
        
        # Modern window sizing - compact but readable
        window_width = 850
        window_height = 750
        
        # Center window on screen
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        center_x = int(screen_width/2 - window_width/2)
        center_y = int(screen_height/2 - window_height/2)
        
        self.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")
        self.min_app_width = 750
        self.min_app_height = 650
        self.minsize(self.min_app_width, self.min_app_height)  # Minimum size to ensure usability
        
        # Configure style for dark theme
        self.configure(bg="#1e1e1e")  # Dark background
        
        # Configure ttk style for dark theme
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#1e1e1e')
        style.configure('TLabel', background='#1e1e1e', foreground='#e0e0e0', font=('Helvetica', 15))
        style.configure('TButton', background='#2d2d2d', foreground='#e0e0e0', font=('Helvetica', 14))
        style.map('TButton', background=[('active', '#3d3d3d')])
        # Style for selected multiple choice buttons (depressed/radio button style)
        style.configure('Selected.TButton', background='#4a6fa5', foreground='#ffffff')
        style.map('Selected.TButton', background=[('active', '#5a7fb5'), ('pressed', '#3a5f95')])
        style.configure(
            'TCombobox',
            background='#2d2d2d',
            foreground='#ffffff',
            fieldbackground='#3d3d3d',
            borderwidth=2,
            relief='solid',
            arrowcolor='#ffffff',
        )
        style.map('TCombobox', fieldbackground=[('readonly', '#3d3d3d')], 
                 background=[('readonly', '#2d2d2d')])
        style.configure('TText', background='#2d2d2d', foreground='#e0e0e0')
        style.configure('TScrollbar', background='#2d2d2d', troughcolor='#1e1e1e')

        # State
        self.selected_language: Optional[str] = None
        self.assessment_cards: List[AssessmentCard] = []
        self.assessment_stage: int = 0  # 0-3 (0 = not started, 1-3 = current stage)
        self.assessment_responses: List[Dict[str, Any]] = []
        self.assessment_result: Optional[AssessmentResult] = None
        self.lesson_plan: Optional[LessonPlan] = None
        self.lesson_index: int = 0
        self.final_summary: Optional[Dict[str, Any]] = None
        self.pending_assessment_images: Set[str] = set()
        
        # Image cache
        self.image_cache: Dict[str, str] = {}  # image_prompt -> image_path
        self.image_photos: Dict[str, ImageTk.PhotoImage] = {}  # image_path -> PhotoImage

        # Container for "cards"
        container = ttk.Frame(self)
        container.pack(fill="both", expand=True)
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        self.cards = {}

        for CardClass in (IntroCard, AssessmentCardView, LessonCardView, SummaryCard):
            card = CardClass(parent=container, controller=self)
            self.cards[CardClass.__name__] = card
            card.grid(row=0, column=0, sticky="nsew")

        self.show_card("IntroCard")

    def adjust_window_to_content(self, content_widget: ttk.Frame, padding: int = 120) -> None:
        """Adjust the application window height/width to fit the currently visible content."""
        try:
            self.update_idletasks()
            content_widget.update_idletasks()
            required_width = content_widget.winfo_reqwidth() + padding
            required_height = content_widget.winfo_reqheight() + padding
            width = max(self.min_app_width, required_width)
            height = max(self.min_app_height, required_height)

            # Only resize when there is a meaningful difference
            current_geom = self.geometry().split("+")[0]
            current_width, current_height = [int(val) for val in current_geom.split("x")]
            if abs(width - current_width) > 5 or abs(height - current_height) > 5:
                self.geometry(f"{int(width)}x{int(height)}")
        except Exception:
            # Fallback to current geometry; avoid crashing on resize attempts
            pass

    # Card management -------------------------------------------------------

    def show_card(self, name: str) -> None:
        card = self.cards[name]
        card.tkraise()

    # High-level flow methods ----------------------------------------------

    def start_session(self, language: str) -> None:
        """Triggered from Intro card when user picks a language."""
        self.selected_language = language
        self.assessment_cards = []
        self.assessment_stage = 0
        self.assessment_responses = []
        self.assessment_result = None
        self.lesson_plan = None
        self.lesson_index = 0
        self.final_summary = None

        # Show loading spinner and switch to assessment card
        assessment_card: AssessmentCardView = self.cards["AssessmentCardView"]
        assessment_card.show_loading("Generating assessment questions...")
        self.show_card("AssessmentCardView")

        # Run API calls in a background thread
        def generate_assessment_threaded():
            try:
                # Generate all assessment cards (fast text-only call)
                assessment_cards = generate_assessment_cards(language)

                # Show questions ASAP; images are fetched per-card on demand
                self.after(0, lambda: self._on_assessment_generated(assessment_cards))

            except Exception as e:
                # Show error in main thread
                self.after(0, lambda: self._on_assessment_error(str(e)))

        thread = threading.Thread(target=generate_assessment_threaded, daemon=True)
        thread.start()


    def _on_assessment_image_generated(self, image_prompt: str, image_path: Optional[str]) -> None:
        """Called when an assessment image is generated (in main thread)."""
        if image_path:
            self.image_cache[image_prompt] = image_path
            # Update UI if this card is currently displayed
            assessment_card: AssessmentCardView = self.cards["AssessmentCardView"]
            assessment_card.update_image_if_needed(image_prompt, image_path)
    
    def _on_lesson_image_generated(self, image_prompt: str, image_path: Optional[str]) -> None:
        """Called when a lesson image is generated (in main thread)."""
        if image_path:
            self.image_cache[image_prompt] = image_path
            # Update UI if this card is currently displayed
            lesson_card: LessonCardView = self.cards["LessonCardView"]
            lesson_card.update_image_if_needed(image_prompt, image_path)

    def _on_assessment_generated(self, assessment_cards: List[AssessmentCard]) -> None:
        """Called when assessment generation completes (in main thread)."""
        self.assessment_cards = assessment_cards
        self.assessment_stage = 1
        
        assessment_card: AssessmentCardView = self.cards["AssessmentCardView"]
        assessment_card.hide_loading()
        assessment_card.show_stage(1, assessment_cards[0])

    def _on_assessment_error(self, error: str) -> None:
        """Called when assessment generation fails (in main thread)."""
        assessment_card: AssessmentCardView = self.cards["AssessmentCardView"]
        assessment_card.hide_loading()
        messagebox.showerror("Assessment Error", f"Failed to generate assessment: {error}\n\nReturning to intro screen.")
        self.show_card("IntroCard")

    def request_assessment_image(self, image_prompt: Optional[str]) -> None:
        """Kick off generation for a specific assessment image if needed."""
        if not image_prompt:
            return
        if image_prompt in self.image_cache:
            return
        if image_prompt in self.pending_assessment_images:
            return

        self.pending_assessment_images.add(image_prompt)

        def _on_image_ready(path: Optional[str]) -> None:
            def _finish() -> None:
                self.pending_assessment_images.discard(image_prompt)
                self._on_assessment_image_generated(image_prompt, path)
            self.after(0, _finish)

        generate_image_async(image_prompt, _on_image_ready)

    def submit_assessment_response(self, stage: int, response: str, answer_index: Optional[int] = None) -> None:
        """Called by AssessmentCardView when user submits an assessment response."""
        if stage < 1 or stage > 3 or stage > len(self.assessment_cards):
            return
        
        # Store response (serialize card to dict, not LessonCard object)
        card = self.assessment_cards[stage - 1].card
        self.assessment_responses.append({
            "stage": stage,
            "response": response,
            "answer_index": answer_index,
            "card_type": card.type,
            "question": card.question or card.word or "",
            "image_prompt": card.image_prompt,
            "instruction": card.instruction,
            "correct_answer": card.correct_answer,
            "options": card.options,
            "correct_index": card.correct_index,
        })
        
        # Move to next stage or evaluate
        if stage < 3:
            self.assessment_stage = stage + 1
            assessment_card: AssessmentCardView = self.cards["AssessmentCardView"]
            assessment_card.show_stage(stage + 1, self.assessment_cards[stage])
        else:
            # All assessments done - evaluate and generate lesson plan
            assessment_card: AssessmentCardView = self.cards["AssessmentCardView"]
            assessment_card.show_loading("Evaluating your responses and generating personalized lessons...")
            
            def evaluate_and_generate_lesson():
                import sys
                try:
                    # OPTIMIZED: Combine assessment evaluation + lesson generation into ONE API call
                    print("[UI] Starting lesson generation...", file=sys.stdout, flush=True)
                    assessment_result, lesson_plan = generate_lesson_plan_from_assessment_responses(
                        self.assessment_responses,
                        self.selected_language or "Spanish"
                    )
                    print("[UI] Lesson plan generated, switching to lesson view...", file=sys.stdout, flush=True)
                    
                    # Show first lesson immediately - don't wait for images
                    self.after(0, lambda: self._on_lesson_plan_generated(assessment_result, lesson_plan))
                    
                    # Start ALL image generation in background (don't block UI)
                    image_prompts_to_generate = [
                        card.image_prompt 
                        for card in lesson_plan.cards 
                        if card.image_prompt and card.image_prompt not in self.image_cache
                    ]
                    
                    # Generate all images in parallel using optimized function
                    if image_prompts_to_generate:
                        print(f"[UI] Starting background generation of {len(image_prompts_to_generate)} images...", file=sys.stdout, flush=True)
                        generate_images_parallel(
                            image_prompts_to_generate,
                            lambda prompt, path: self._on_lesson_image_generated(prompt, path)
                        )
                except Exception as e:
                    print(f"[UI] Error generating lessons: {e}", file=sys.stderr, flush=True)
                    self.after(0, lambda: self._on_lesson_generation_error(str(e)))
            
            thread = threading.Thread(target=evaluate_and_generate_lesson, daemon=True)
            thread.start()

    def _on_lesson_image_generated(self, image_prompt: str, image_path: Optional[str]) -> None:
        """Called when a lesson image is generated (in main thread)."""
        if image_path:
            self.image_cache[image_prompt] = image_path
            # Update UI if this card is currently displayed
            lesson_card: LessonCardView = self.cards["LessonCardView"]
            lesson_card.update_image_if_needed(image_prompt, image_path)

    def _on_lesson_plan_generated(self, assessment_result: AssessmentResult, lesson_plan: LessonPlan) -> None:
        """Called when lesson plan generation completes (in main thread)."""
        import sys
        print("[UI] Lesson plan received, switching to lesson view...", file=sys.stdout, flush=True)
        self.assessment_result = assessment_result
        self.lesson_plan = lesson_plan
        self.lesson_index = 0
        
        # Hide assessment loading
        assessment_card: AssessmentCardView = self.cards["AssessmentCardView"]
        assessment_card.hide_loading()
        
        # Switch to lesson card view and show first card
        lesson_card: LessonCardView = self.cards["LessonCardView"]
        print("[UI] Showing first lesson card...", file=sys.stdout, flush=True)
        self.show_card("LessonCardView")
        lesson_card.hide_loading()
        lesson_card.show_card_index(0)

    def _on_lesson_generation_error(self, error: str) -> None:
        """Called when lesson generation fails (in main thread)."""
        assessment_card: AssessmentCardView = self.cards["AssessmentCardView"]
        assessment_card.hide_loading()
        messagebox.showerror("Lesson Generation Error", f"Failed to generate lessons: {error}\n\nReturning to intro screen.")
        self.show_card("IntroCard")

    def submit_lesson_card_response(self, response: str, answer_index: Optional[int] = None) -> None:
        """Submit response for current lesson card and get feedback."""
        if not self.lesson_plan or self.lesson_index >= len(self.lesson_plan.cards):
            return
        
        card = self.lesson_plan.cards[self.lesson_index]
        card.user_response = response
        card.user_answer_index = answer_index
        
        lesson_card: LessonCardView = self.cards["LessonCardView"]
        
        # For text-based responses, show loading (API call needed)
        # For multiple choice, instant evaluation
        needs_api_call = card.type in ("text_question", "image_question", "fill_in_blank")
        
        if needs_api_call:
            lesson_card.show_feedback_loading()
        
        def evaluate_response():
            try:
                evaluation = evaluate_card_response(
                    card,
                    response,
                    answer_index,
                    self.selected_language or "Spanish"
                )
                
                card.is_correct = evaluation["is_correct"]
                card.card_score = evaluation["card_score"]
                card.feedback = evaluation["feedback"]
                if evaluation.get("alternatives"):
                    card.alternatives = evaluation["alternatives"]
                if evaluation.get("vocabulary_expansion"):
                    card.vocabulary_expansion = evaluation["vocabulary_expansion"]
                
                self.after(0, lambda: lesson_card.show_feedback(evaluation))
            except Exception as e:
                self.after(0, lambda: lesson_card.show_feedback_error(str(e)))
        
        if needs_api_call:
            # Run API call in background thread
            thread = threading.Thread(target=evaluate_response, daemon=True)
            thread.start()
        else:
            # Instant evaluation for multiple choice
            evaluation = evaluate_card_response(
                card,
                response,
                answer_index,
                self.selected_language or "Spanish"
            )
            
            card.is_correct = evaluation["is_correct"]
            card.card_score = evaluation["card_score"]
            card.feedback = evaluation["feedback"]
            if evaluation.get("alternatives"):
                card.alternatives = evaluation["alternatives"]
            if evaluation.get("vocabulary_expansion"):
                card.vocabulary_expansion = evaluation["vocabulary_expansion"]
            
            lesson_card.show_feedback(evaluation)

    def next_lesson_card(self) -> None:
        """Move to the next lesson card or to summary when done."""
        if not self.lesson_plan:
            return

        self.lesson_index += 1
        if self.lesson_index >= len(self.lesson_plan.cards):
            # Done -> generate final summary and go to summary card
            summary_card: SummaryCard = self.cards["SummaryCard"]
            summary_card.show_loading("Generating final summary...")
            self.show_card("SummaryCard")
            
            def generate_summary():
                try:
                    total_score = sum(card.card_score for card in self.lesson_plan.cards)
                    average_score = total_score // len(self.lesson_plan.cards) if self.lesson_plan.cards else 0
                    self.lesson_plan.total_score = average_score
                    
                    summary = generate_final_summary(
                        self.lesson_plan,
                        self.assessment_result or AssessmentResult(),
                        self.selected_language or "Spanish"
                    )
                    self.final_summary = summary
                    self.after(0, lambda: summary_card.set_summary(summary, self.lesson_plan))
                except Exception as e:
                    self.after(0, lambda: summary_card.show_error(str(e)))
            
            thread = threading.Thread(target=generate_summary, daemon=True)
            thread.start()
        else:
            lesson_card: LessonCardView = self.cards["LessonCardView"]
            lesson_card.show_card_index(self.lesson_index)

    def restart(self) -> None:
        """Restart the whole experience from the intro."""
        self.selected_language = None
        self.assessment_cards = []
        self.assessment_stage = 0
        self.assessment_responses = []
        self.assessment_result = None
        self.lesson_plan = None
        self.lesson_index = 0
        self.final_summary = None
        self.image_cache.clear()
        self.image_photos.clear()

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

        # Center content in window - add side spacers
        self.columnconfigure(0, weight=1)  # Left spacer
        self.columnconfigure(1, weight=0)  # Center column - no expansion, sized by content
        self.columnconfigure(2, weight=1)  # Right spacer
        self.rowconfigure(0, weight=1)  # Top spacer
        self.rowconfigure(2, weight=1)  # Bottom spacer
        
        # Content wrapper - sized by content, centered by spacers
        content_frame = ttk.Frame(self)
        content_frame.grid(row=1, column=1, sticky="ns")
        self.content = content_frame
        content_frame.columnconfigure(0, weight=1)
        content_frame.rowconfigure(0, weight=1)
        content_frame.rowconfigure(5, weight=1)

        title = ttk.Label(
            self.content,
            text="GAIT Language Buddy",
            font=("Helvetica", 28, "bold"),
            anchor="center",
            foreground="#ffffff",  # White text on dark background
        )
        title.grid(row=1, column=0, pady=(40, 20), padx=20, sticky="ew")

        desc_text = (
            "Welcome! This tool helps you learn and practice another language with personalized lessons.\n\n"
            "You will:\n"
            "  1. Choose a target language.\n"
            "  2. Complete a 3-stage assessment to determine your proficiency level.\n"
            "  3. Work through 10 personalized lesson cards with various question types.\n"
            "  4. Receive immediate feedback on each answer with vocabulary expansion.\n\n"
            "At the end, you'll see a comprehensive summary with scores and study suggestions."
        )
        desc = ttk.Label(
            self.content, 
            text=desc_text, 
            wraplength=650, 
            justify="left",
            font=("Helvetica", 14),
            foreground="#d0d0d0",  # Light gray text on dark background
        )
        desc.grid(row=2, column=0, padx=80, pady=(0, 30), sticky="ew")
        self.desc_label = desc

        lang_frame = ttk.Frame(self.content)
        lang_frame.grid(row=3, column=0, padx=40, pady=(0, 20))

        ttk.Label(lang_frame, text="Choose your target language:", font=("Helvetica", 14)).grid(
            row=0, column=0, sticky="w", padx=(0, 10)
        )

        self.language_var = tk.StringVar()
        self.language_combo = ttk.Combobox(
            lang_frame,
            textvariable=self.language_var,
            values=SUPPORTED_LANGUAGES,
            state="readonly",
            width=25,
            font=("Helvetica", 13),
        )
        self.language_combo.grid(row=0, column=1, padx=(0, 0))
        
        # Style the combobox to be more visible
        style = ttk.Style()
        style.configure(
            'TCombobox',
            fieldbackground='#3d3d3d',
            background='#2d2d2d',
            foreground='#ffffff',
            borderwidth=2,
            relief='solid',
            arrowcolor='#ffffff',
        )

        start_button = ttk.Button(
            self.content,
            text="Start Session",
            command=self._on_start_clicked,
        )
        start_button.grid(row=4, column=0, pady=(20, 50))
        
        # Style the button to be more prominent
        style = ttk.Style()
        style.configure("Start.TButton", font=("Helvetica", 14, "bold"), padding=10)
        start_button.configure(style="Start.TButton")

        # Responsive wrapping
        self.content.bind("<Configure>", self._on_content_resize)

    def _on_start_clicked(self) -> None:
        language = self.language_var.get()
        if not language:
            messagebox.showinfo("Choose a language", "Please select a target language to continue.")
            return

        # Start the session in the controller
        self.controller.start_session(language)

    def reset(self) -> None:
        self.language_var.set("")

    def _on_content_resize(self, event: tk.Event) -> None:
        """Adjust intro text wrapping when the window size changes."""
        if event.widget is not self.content:
            return
        available = max(300, event.width - 160)
        self.desc_label.configure(wraplength=available)


# ---------------------------------------------------------------------------
# Assessment Card View (3-stage initial assessment)
# ---------------------------------------------------------------------------

class AssessmentCardView(ttk.Frame):
    """Displays the 3-stage initial language assessment."""
    
    def __init__(self, parent, controller: LanguageBuddyApp) -> None:
        super().__init__(parent)
        self.controller = controller
        self.current_card: Optional[AssessmentCard] = None

        # Center content in window - add side spacers
        self.columnconfigure(0, weight=1)  # Left spacer
        self.columnconfigure(1, weight=0)  # Center column - no expansion, sized by content
        self.columnconfigure(2, weight=1)  # Right spacer
        self.rowconfigure(0, weight=1)  # Top spacer
        self.rowconfigure(2, weight=1)  # Bottom spacer
        
        # Content wrapper - sized by content, centered by spacers
        content_frame = ttk.Frame(self)
        content_frame.grid(row=1, column=1, sticky="ns")
        self.content = content_frame
        content_frame.columnconfigure(0, weight=1)
        content_frame.rowconfigure(0, weight=1)
        content_frame.rowconfigure(7, weight=1)

        self.title_label = ttk.Label(
            self.content,
            text="Language Assessment",
            font=("Helvetica", 28, "bold"),
            anchor="center",
            foreground="#ffffff",  # White text on dark background
        )
        self.title_label.grid(row=1, column=0, pady=(30, 15), sticky="ew")

        self.stage_label = ttk.Label(
            self.content,
            text="Stage 1 of 3",
            font=("Helvetica", 14),
            foreground="#7bb3ff",  # Light blue text on dark background
        )
        self.stage_label.grid(row=2, column=0, pady=(0, 10))

        # Loading spinner (starts hidden)
        self.loading_spinner = LoadingSpinner(self.content, text="Generating assessment...")
        self.loading_spinner.grid(row=3, column=0, pady=(20, 20))

        self.pending_instruction_text: str = ""
        self.waiting_for_image: bool = False

        # Instruction label (left-justified)
        self.instruction_label = ttk.Label(
            self.content,
            text="",
            wraplength=900,
            justify="left",
            font=("Helvetica", 14),
            foreground="#7bb3ff",  # Light blue text on dark background
        )
        self.instruction_label.grid(row=4, column=0, padx=50, pady=(0, 10), sticky="w")

        # Image loading status labels
        self.image_status_label = ttk.Label(
            self.content,
            text="",
            wraplength=900,
            justify="left",
            font=("Helvetica", 13),
            foreground="#7bb3ff",
        )
        self.image_status_label.grid(row=5, column=0, padx=50, sticky="w")
        self.image_status_label.grid_remove()

        self.image_status_detail_label = ttk.Label(
            self.content,
            text="",
            wraplength=900,
            justify="left",
            font=("Helvetica", 12, "italic"),
            foreground="#9bc6ff",
        )
        self.image_status_detail_label.grid(row=6, column=0, padx=50, pady=(0, 10), sticky="w")
        self.image_status_detail_label.grid_remove()

        # Image area (for image-based questions)
        self.image_label = ttk.Label(self.content)
        self.image_label.grid(row=7, column=0, padx=20, pady=(0, 10))
        self.image_label.grid_remove()

        # Question label
        self.question_label = ttk.Label(
            self.content,
            text="",
            font=("Helvetica", 24, "bold"),
            wraplength=900,
            justify="center",
            foreground="#ffffff",  # White text on dark background
        )
        self.question_label.grid(row=8, column=0, padx=50, pady=(15, 8), sticky="ew")

        # Answer area (dynamic based on card type)
        self.answer_frame = ttk.Frame(self.content)
        self.answer_frame.grid(row=9, column=0, padx=50, pady=(0, 15), sticky="nsew")
        
        # Text input (for text questions)
        self.text_widget: Optional[tk.Text] = None
        self.text_scrollbar: Optional[ttk.Scrollbar] = None

        submit_button = ttk.Button(
            self.content,
            text="Submit Answer",
            command=self._on_submit_clicked,
        )
        self.submit_button = submit_button
        submit_button.grid(row=10, column=0, pady=(10, 20))

        # Responsive wrapping labels
        self._wrap_targets = [
            self.instruction_label,
            self.image_status_label,
            self.image_status_detail_label,
            self.question_label,
        ]
        self.content.bind("<Configure>", self._update_wraplengths)

    # ------------------------------------------------------------------
    # Helper methods for instruction/status text
    # ------------------------------------------------------------------

    def _set_instruction_text(self, text: str) -> None:
        self.instruction_label.configure(text=text)

    def _show_image_status(self, primary: str, detail: Optional[str] = None) -> None:
        self.image_status_label.configure(text=primary)
        self.image_status_label.grid()
        if detail:
            self.image_status_detail_label.configure(text=detail)
            self.image_status_detail_label.grid()
        else:
            self.image_status_detail_label.grid_remove()

    def _clear_image_status(self) -> None:
        self.image_status_label.grid_remove()
        self.image_status_detail_label.grid_remove()

    def _update_wraplengths(self, event: Optional[tk.Event] = None) -> None:
        """Keep instruction/question text wrapping responsive."""
        if event and event.widget is not self.content:
            return
        width = self.content.winfo_width() or self.winfo_width()
        wrap = max(250, width - 120)
        for label in self._wrap_targets:
            label.configure(wraplength=wrap)

    def show_loading(self, message: str = "Loading...") -> None:
        """Show the loading spinner."""
        self.loading_spinner.text = message
        self.loading_spinner.label.configure(text=f"{self.loading_spinner.spinner_chars[0]} {message}")
        self.loading_spinner.start()
        self.instruction_label.grid_remove()
        self.image_status_label.grid_remove()
        self.image_status_detail_label.grid_remove()
        self.image_label.grid_remove()
        self.question_label.grid_remove()
        self.answer_frame.grid_remove()
        self.submit_button.grid_remove()
        self.stage_label.grid_remove()
        self.title_label.grid_remove()

    def hide_loading(self) -> None:
        """Hide the loading spinner."""
        self.loading_spinner.stop()
        self.instruction_label.grid()
        if self.image_status_label.cget("text"):
            self.image_status_label.grid()
        if self.image_status_detail_label.cget("text"):
            self.image_status_detail_label.grid()
        self.question_label.grid()
        self.answer_frame.grid()
        self.submit_button.grid()
        self.stage_label.grid()
        self.title_label.grid()

    def show_stage(self, stage: int, assessment_card: AssessmentCard) -> None:
        """Display a specific assessment stage."""
        self.current_card = assessment_card
        card = assessment_card.card

        self.stage_label.configure(text=f"Stage {stage} of 3")

        # Instruction text we want once everything is ready
        instruction_text = card.instruction or "Please answer the following question:"
        self.pending_instruction_text = instruction_text

        self._set_instruction_text(instruction_text)
        self._clear_image_status()

        # Set question now (we may temporarily hide it if waiting for image)
        self.question_label.configure(text=card.question or "")

        # Clear previous answer widgets
        self._clear_answer_widgets()

        # Render answer area (assessment now uses text input only)
        self._render_text_input()

        # By default, show question + answers + submit
        self.question_label.grid()
        self.answer_frame.grid()
        self.submit_button.grid()

        # Handle image loading
        if card.image_prompt:
            image_path = self.controller.image_cache.get(card.image_prompt)
            if image_path:
                # Image already available
                self.waiting_for_image = False
                self._display_image(image_path)
            else:
                # We are waiting for the image: show a *text* status, not a second spinner
                self.waiting_for_image = True
                self._set_instruction_text("Loading image for this question...")
                self._show_image_status("Loading image...", "This may take a few seconds.")
                self.image_label.grid_remove()
                # Hide Q&A until image arrives so the user doesn't answer blind
                self.question_label.grid_remove()
                self.answer_frame.grid_remove()
                self.submit_button.grid_remove()
                self.controller.request_assessment_image(card.image_prompt)
        else:
            self.waiting_for_image = False
            self.image_label.grid_remove()

        self.controller.adjust_window_to_content(self.content)


    def _render_text_input(self) -> None:
        """Render text input area."""
        text_frame = ttk.Frame(self.answer_frame)
        text_frame.grid(row=0, column=0, sticky="nsew")
        self.answer_frame.rowconfigure(0, weight=1)
        self.answer_frame.columnconfigure(0, weight=1)
        
        self.text_widget = tk.Text(
            text_frame, 
            height=8, 
            wrap="word",
            bg="#2d2d2d",  # Dark background
            fg="#e0e0e0",  # Light text
            insertbackground="#e0e0e0",  # Light cursor
            selectbackground="#3d3d3d",  # Dark selection background
            selectforeground="#ffffff",  # White selected text
        )
        self.text_widget.grid(row=0, column=0, sticky="nsew", pady=(5, 0))
        text_frame.rowconfigure(0, weight=1)
        text_frame.columnconfigure(0, weight=1)
        
        self.text_scrollbar = ttk.Scrollbar(
            text_frame,
            orient="vertical",
            command=self.text_widget.yview,
        )
        self.text_scrollbar.grid(row=0, column=1, sticky="ns")
        self.text_widget.configure(yscrollcommand=self.text_scrollbar.set)

    def _clear_answer_widgets(self) -> None:
        """Clear all answer widgets."""
        # Destroy text widget
        if self.text_widget:
            self.text_widget.destroy()
            self.text_scrollbar.destroy()
            self.text_widget = None
            self.text_scrollbar = None


    def _display_image(self, image_path: str) -> None:
        """Display an image from path and restore normal layout."""
        try:
            img = Image.open(image_path)
            img.thumbnail((600, 400))
            photo = ImageTk.PhotoImage(img)
            # Keep reference to prevent GC
            self.controller.current_photo = photo
            self.image_label.configure(image=photo, text="")
            self.image_label.grid()

        except Exception:
            self.image_label.configure(text="[Could not display image]", image="")
            self.image_label.grid()

        # Once the image is ready, restore the normal instruction + Q&A
        self._set_instruction_text(self.pending_instruction_text or "Please answer the following question.")
        self._clear_image_status()
        self.question_label.grid()
        self.answer_frame.grid()
        self.submit_button.grid()
        self.controller.adjust_window_to_content(self.content)


    def update_image_if_needed(self, image_prompt: str, image_path: str) -> None:
        """Update image if it matches the current card's image prompt."""
        if (
            self.current_card
            and self.current_card.card.image_prompt == image_prompt
            and self.waiting_for_image
        ):
            self.waiting_for_image = False
            self._display_image(image_path)


    def _on_submit_clicked(self) -> None:
        """Handle submit button click."""
        if not self.current_card:
            return
        
        card = self.current_card.card
        response = ""
        answer_index = None
        
        if self.text_widget:
            response = self.text_widget.get("1.0", tk.END).strip()
            if not response:
                messagebox.showinfo("Enter an answer", "Please enter your answer before submitting.")
                return
        else:
            messagebox.showinfo("Enter an answer", "Please enter your answer before submitting.")
            return
        
        self.controller.submit_assessment_response(
            self.current_card.stage,
            response,
            answer_index
        )


# ---------------------------------------------------------------------------
# Lesson card (10 prompts, one at a time)
# ---------------------------------------------------------------------------

class LessonCardView(ttk.Frame):
    def __init__(self, parent, controller: LanguageBuddyApp) -> None:
        super().__init__(parent)
        self.controller = controller

        # Center content in window - add side spacers
        self.columnconfigure(0, weight=1)  # Left spacer
        self.columnconfigure(1, weight=0)  # Center column - no expansion, sized by content
        self.columnconfigure(2, weight=1)  # Right spacer
        self.rowconfigure(0, weight=1)  # Top spacer
        self.rowconfigure(2, weight=1)  # Bottom spacer
        
        # Content wrapper - sized by content, centered by spacers
        content_frame = ttk.Frame(self)
        content_frame.grid(row=1, column=1, sticky="nsew", pady=(10, 20))
        self.content = content_frame
        content_frame.columnconfigure(0, weight=1)
        content_frame.rowconfigure(0, weight=1)
        content_frame.rowconfigure(7, weight=1)

        self.title_label = ttk.Label(
            self.content,
            text="Lesson Prompts",
            font=("Helvetica", 24, "bold"),
        )
        self.title_label.grid(row=1, column=0, pady=(20, 10))

        self.subtitle_label = ttk.Label(
            self.content,
            text="Work through the prompts one by one.",
            wraplength=900,
            justify="left",
            font=("Helvetica", 14),
            foreground="#d0d0d0",  # Light gray text on dark background
        )
        self.subtitle_label.grid(row=2, column=0, padx=50, pady=(0, 15), sticky="w")

        # Loading spinner
        self.loading_spinner = LoadingSpinner(self.content, text="Evaluating and generating lesson...")
        self.loading_spinner.grid(row=3, column=0, pady=(40, 40))

        self.progress_label = ttk.Label(
            self.content, 
            text="",
            foreground="#7bb3ff",  # Light blue text on dark background
        )
        self.progress_label.grid(row=3, column=0, pady=(0, 10))
        
        # Image area (for image-based questions)
        self.image_label = ttk.Label(self.content)
        self.image_label.grid(row=3, column=0, padx=20, pady=(0, 10))
        self.image_label.grid_remove()
        
        # Question label
        self.question_label = ttk.Label(
            self.content,
            text="",
            font=("Helvetica", 24, "bold"),
            wraplength=900,
            justify="center",
            foreground="#ffffff",  # White text on dark background
        )
        self.question_label.grid(row=4, column=0, padx=50, pady=(15, 8), sticky="ew")
        
        # Answer area (dynamic based on card type)
        self.answer_frame = ttk.Frame(self.content)
        self.answer_frame.grid(row=5, column=0, padx=20, pady=(10, 10), sticky="nsew")
        
        # Multiple choice buttons (will be created dynamically)
        self.mc_buttons: List[ttk.Button] = []
        self.selected_mc_index: Optional[int] = None
        
        # Text input (for text questions)
        self.text_widget: Optional[tk.Text] = None
        self.text_scrollbar: Optional[ttk.Scrollbar] = None

        self.next_button = ttk.Button(
            self.content, text="Submit Answer", command=self._on_next_clicked
        )
        self.next_button.grid(row=6, column=0, pady=(10, 20))
        self.submit_button = self.next_button

        # Feedback card (full-width, scrollable)
        self.feedback_card = ttk.Frame(self.content)
        self.feedback_card.grid(row=7, column=0, padx=20, pady=(5, 20), sticky="nsew")
        self.content.rowconfigure(7, weight=1)
        self.feedback_card.columnconfigure(0, weight=1)
        self.feedback_card.rowconfigure(1, weight=1)
        self.feedback_card.grid_remove()

        self.feedback_title = ttk.Label(
            self.feedback_card,
            text="Feedback",
            font=("Helvetica", 20, "bold"),
            justify="left",
            anchor="w",
        )
        self.feedback_title.grid(row=0, column=0, pady=(4, 6), sticky="ew")

        self.feedback_canvas = tk.Canvas(
            self.feedback_card,
            highlightthickness=0,
            background="#1e1e1e",
        )
        self.feedback_canvas.grid(row=1, column=0, sticky="nsew")
        self.feedback_scrollbar = ttk.Scrollbar(
            self.feedback_card, orient="vertical", command=self.feedback_canvas.yview
        )
        self.feedback_scrollbar.grid(row=1, column=1, sticky="ns")
        self.feedback_canvas.configure(yscrollcommand=self.feedback_scrollbar.set)

        self.feedback_body = ttk.Frame(self.feedback_canvas)
        self.feedback_body.columnconfigure(0, weight=1)
        self.feedback_body_window = self.feedback_canvas.create_window(
            (0, 0), window=self.feedback_body, anchor="nw"
        )

        def _sync_scroll_region(_: tk.Event) -> None:
            self.feedback_canvas.configure(scrollregion=self.feedback_canvas.bbox("all"))

        self.feedback_body.bind("<Configure>", _sync_scroll_region)
        self.feedback_canvas.bind("<Configure>", self._on_feedback_canvas_resize)

        self.feedback_label = ttk.Label(
            self.feedback_body,
            text="",
            wraplength=900,
            justify="left",
            anchor="w",
            foreground="#d0d0d0",
        )
        self.feedback_label.grid(row=0, column=0, pady=(0, 8), sticky="ew")

        self.vocab_expansion_label = ttk.Label(
            self.feedback_body,
            text="",
            wraplength=900,
            justify="left",
            anchor="w",
            font=("Helvetica", 14, "bold"),
            foreground="#7bb3ff",
        )
        self.vocab_expansion_label.grid(row=1, column=0, pady=(0, 6), sticky="ew")

        self.continue_button = ttk.Button(
            self.feedback_card,
            text="Continue",
            command=self._on_continue_after_feedback,
        )
        self.continue_button.grid(row=2, column=0, pady=(10, 5), sticky="e")

        self.feedback_spinner = LoadingSpinner(
            self.content, text="Evaluating your answer..."
        )
        self.feedback_spinner.grid(row=7, column=0, pady=(10, 30))
        self.feedback_spinner.grid_remove()

        # Responsive wrapping for question/feedback text
        self.content.bind("<Configure>", self._update_wraplengths)

    def _reset_feedback_state(self) -> None:
        """Hide feedback card/spinner and reset labels."""
        self.feedback_spinner.stop()
        self.feedback_spinner.grid_remove()
        self.feedback_card.grid_remove()
        self.feedback_title.configure(text="Feedback", foreground="#ffffff")
        self.feedback_label.configure(text="")
        self.vocab_expansion_label.configure(text="")
        self.vocab_expansion_label.grid_remove()
        self.continue_button.configure(text="Continue", command=self._on_continue_after_feedback)
        if hasattr(self, "feedback_canvas"):
            self.feedback_canvas.yview_moveto(0)

    def show_loading(self, message: str = "Loading...") -> None:
        """Show the loading spinner."""
        self.loading_spinner.text = message
        self.loading_spinner.label.configure(text=f"{self.loading_spinner.spinner_chars[0]} {message}")
        self.loading_spinner.start()
        self.progress_label.grid_remove()
        self.image_label.grid_remove()
        self.question_label.grid_remove()
        self.answer_frame.grid_remove()
        self.next_button.grid_remove()
        self.title_label.grid_remove()
        self.subtitle_label.grid_remove()
        self._reset_feedback_state()

    def hide_loading(self) -> None:
        """Hide the loading spinner."""
        self.loading_spinner.stop()
        self.title_label.grid()
        self.subtitle_label.grid()

    def _update_wraplengths(self, event: Optional[tk.Event] = None) -> None:
        """Adjust wraplengths dynamically based on available width."""
        if event and event.widget is not self.content:
            return
        width = self.content.winfo_width() or self.winfo_width()
        primary_wrap = max(320, width - 80)
        card_width = self.feedback_card.winfo_width() or width
        feedback_wrap = max(260, card_width - 60)
        self.subtitle_label.configure(wraplength=primary_wrap)
        self.question_label.configure(wraplength=primary_wrap)
        self.feedback_label.configure(wraplength=feedback_wrap)
        self.vocab_expansion_label.configure(wraplength=feedback_wrap)

    def _on_feedback_canvas_resize(self, event: tk.Event) -> None:
        """Ensure the feedback body fills the available width."""
        if hasattr(self, "feedback_body_window"):
            self.feedback_canvas.itemconfigure(self.feedback_body_window, width=event.width)
        # Recompute wrap lengths with the new width
        self._update_wraplengths()

    def show_card_index(self, index: int) -> None:
        """Show a specific lesson card by index."""
        plan = self.controller.lesson_plan
        if not plan or not plan.cards:
            return

        total = len(plan.cards)
        index = max(0, min(index, total - 1))
        self.controller.lesson_index = index
        card = plan.cards[index]
        
        self.progress_label.configure(text=f"Card {index + 1} of {total}")
        self._reset_feedback_state()
        self.next_button.configure(state="normal")
        self.next_button.grid()
        self._render_card(card)
        
        # Reset submit button for new card
        if index == total - 1:
            self.next_button.configure(text="Finish Lesson", command=self._on_next_clicked)
        else:
            self.next_button.configure(text="Submit Answer", command=self._on_next_clicked)
        self.submit_button = self.next_button

    def _render_card(self, card: LessonCard) -> None:
        """Render a structured lesson card."""
        # Clear previous widgets
        self._clear_card_widgets()
        
        # Show instruction if available
        if card.instruction:
            self.subtitle_label.configure(text=card.instruction)
            self.subtitle_label.grid()
        else:
            self.subtitle_label.grid_remove()
        
        # Show question
        question_text = card.question or card.word or ""
        self.question_label.configure(text=question_text)
        
        # Show image if available
        if card.image_prompt:
            image_path = self.controller.image_cache.get(card.image_prompt)
            if image_path:
                self._display_image(image_path)
            else:
                # Simple text placeholder while image loads
                self.image_label.configure(text="Loading image...", image="")
                self.image_label.grid()
        else:
            self.image_label.grid_remove() 

        # Render answer area based on card type
        if card.type == "multiple_choice":
            self._render_multiple_choice(card)
        elif card.type == "vocabulary":
            self._render_vocabulary(card)
        else:
            self._render_text_input(card)
    
    def _render_multiple_choice(self, card: LessonCard) -> None:
        """Render multiple choice options."""
        if not card.options:
            # Fallback: if the model failed to provide options, fall back to text input
            self._render_text_input(card)
            return
        
        # Clear text widget if it exists
        if hasattr(self, 'text_widget') and self.text_widget:
            self.text_widget.grid_remove()
        if hasattr(self, 'text_scrollbar') and self.text_scrollbar:
            self.text_scrollbar.grid_remove()
        
        # Reset selection state
        self.selected_mc_index = None
        
        # Create buttons for each option
        self.mc_buttons = []
        for i, option in enumerate(card.options):
            btn = ttk.Button(
                self.answer_frame,
                text=f"{chr(65+i)}. {option}",  # A, B, C, D
                command=lambda idx=i: self._on_mc_selected(idx),
            )
            btn.grid(row=i, column=0, sticky="ew", pady=(4, 4), padx=20)
            self.mc_buttons.append(btn)
        self.answer_frame.grid()
        self.answer_frame.columnconfigure(0, weight=1)

    def _set_instruction_text(self, text: str) -> None:
        self.instruction_label.configure(text=text)

    def _show_image_status(self, primary: str, detail: Optional[str] = None) -> None:
        self.image_status_label.configure(text=primary)
        self.image_status_label.grid()
        if detail:
            self.image_status_detail_label.configure(text=detail)
            self.image_status_detail_label.grid()
        else:
            self.image_status_detail_label.grid_remove()

    def _clear_image_status(self) -> None:
        self.image_status_label.grid_remove()
        self.image_status_detail_label.grid_remove()
    
    def _render_vocabulary(self, card: LessonCard) -> None:
        """Render vocabulary card (read-only display)."""
        vocab_text = f"Word: {card.word}\n"
        if card.translation:
            vocab_text += f"Translation: {card.translation}\n"
        if card.example:
            vocab_text += f"Example: {card.example}\n"
        if card.related_words:
            vocab_text += f"Related words: {', '.join(card.related_words)}"
        
        if not hasattr(self, 'vocab_label'):
            self.vocab_label = ttk.Label(self.answer_frame, text="", wraplength=600, justify="center")
            self.vocab_label.grid(row=0, column=0, padx=20, pady=10)
        self.vocab_label.configure(text=vocab_text)
        self.answer_frame.grid()
    
    def _render_text_input(self, card: LessonCard) -> None:
        """Render text input area."""
        # Clear multiple choice buttons if they exist
        if hasattr(self, 'mc_buttons'):
            for btn in self.mc_buttons:
                btn.destroy()
            self.mc_buttons = []
        
        if not hasattr(self, 'text_widget') or not self.text_widget:
            text_frame = ttk.Frame(self.answer_frame)
            text_frame.grid(row=0, column=0, sticky="nsew")
            self.answer_frame.rowconfigure(0, weight=1)
            self.answer_frame.columnconfigure(0, weight=1)
            
            self.text_widget = tk.Text(
                text_frame, 
                height=6, 
                wrap="word",
                bg="#2d2d2d",  # Dark background
                fg="#e0e0e0",  # Light text
                insertbackground="#e0e0e0",  # Light cursor
                selectbackground="#3d3d3d",  # Dark selection background
                selectforeground="#ffffff",  # White selected text
            )
            self.text_widget.grid(row=0, column=0, sticky="nsew", pady=(5, 0))
            text_frame.rowconfigure(0, weight=1)
            text_frame.columnconfigure(0, weight=1)
            
            self.text_scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.text_widget.yview)
            self.text_scrollbar.grid(row=0, column=1, sticky="ns")
            self.text_widget.configure(yscrollcommand=self.text_scrollbar.set)
        
        self.text_widget.delete("1.0", tk.END)
        self.text_widget.grid()
        self.answer_frame.grid()
    
    def _clear_card_widgets(self) -> None:
        """Clear all card rendering widgets."""
        self.answer_frame.grid_remove()
        if hasattr(self, 'mc_buttons'):
            for btn in self.mc_buttons:
                btn.destroy()
            self.mc_buttons = []
        if hasattr(self, 'text_widget') and self.text_widget:
            self.text_widget.grid_remove()
        if hasattr(self, 'vocab_label'):
            self.vocab_label.grid_remove()
    
    def _on_mc_selected(self, index: int) -> None:
        """Handle multiple choice selection - buttons stay depressed (radio button style)."""
        self.selected_mc_index = index
        # Update button styles to show selection - selected button stays depressed
        for i, btn in enumerate(self.mc_buttons):
            if i == index:
                btn.configure(style="Selected.TButton")
            else:
                btn.configure(style="TButton")
    
    def _display_image(self, image_path: str) -> None:
        """Display an image from path."""
        try:
            img = Image.open(image_path)
            img.thumbnail((600, 400))
            photo = ImageTk.PhotoImage(img)
            if not hasattr(self.controller, 'current_lesson_photo'):
                self.controller.current_lesson_photo = photo
            self.controller.current_lesson_photo = photo
            if not hasattr(self, 'image_label'):
                self.image_label = ttk.Label(self.content)
                self.image_label.grid(row=6, column=0, padx=20, pady=(0, 10))
            self.image_label.configure(image=photo, text="")
            self.image_label.grid()
            self.controller.adjust_window_to_content(self.content)
        except Exception:
            if not hasattr(self, 'image_label'):
                self.image_label = ttk.Label(self.content)
                self.image_label.grid(row=6, column=0, padx=20, pady=(0, 10))
            self.image_label.configure(text="[Could not display image]", image="")
            self.image_label.grid()
            self.controller.adjust_window_to_content(self.content)
    
    
    def update_image_if_needed(self, image_prompt: str, image_path: str) -> None:
        """Update image if it matches the current card's image prompt."""
        plan = self.controller.lesson_plan
        if plan and self.controller.lesson_index < len(plan.cards):
            card = plan.cards[self.controller.lesson_index]
            if card.image_prompt == image_prompt:
                self._display_image(image_path)


    def show_feedback_loading(self) -> None:
        """Show loading while evaluating feedback."""
        self.answer_frame.grid_remove()
        self.next_button.grid_remove()
        self.feedback_card.grid_remove()
        self.feedback_spinner.label.configure(text="⏳ Evaluating your answer...")
        self.feedback_spinner.grid()
        self.feedback_spinner.start()
    
    def show_feedback(self, evaluation: Dict[str, Any]) -> None:
        """Show feedback after card evaluation."""
        self.feedback_spinner.stop()
        self.feedback_spinner.grid_remove()
        self.answer_frame.grid_remove()
        self.next_button.grid_remove()
        
        feedback_text = evaluation.get("feedback", "")
        correct_answer = evaluation.get("correct_answer", "")
        alternatives = evaluation.get("alternatives", [])
        vocab_expansion = evaluation.get("vocabulary_expansion", [])
        
        feedback_display = "✓ Correct!" if evaluation.get("is_correct") else "✗ Incorrect"
        if feedback_text:
            feedback_display += f"\n\n{feedback_text}"
        if correct_answer:
            feedback_display += f"\n\nCorrect answer: {correct_answer}"
        if alternatives:
            feedback_display += f"\nAlternatives: {', '.join(alternatives)}"
        
        self.feedback_title.configure(text="Feedback", foreground="#ffffff")
        self.feedback_label.configure(text=feedback_display)
        self.feedback_label.grid()
        
        if vocab_expansion:
            vocab_text = "Vocabulary to learn: " + ", ".join(vocab_expansion)
            self.vocab_expansion_label.configure(text=vocab_text)
            self.vocab_expansion_label.grid()
        else:
            self.vocab_expansion_label.grid_remove()
        
        self.feedback_card.grid()
        self.continue_button.configure(text="Continue to next prompt", command=self._on_continue_after_feedback)
        self.continue_button.grid()
    
    def show_feedback_error(self, error: str) -> None:
        """Show error if feedback evaluation fails."""
        self.feedback_spinner.stop()
        self.feedback_spinner.grid_remove()
        self.answer_frame.grid_remove()
        self.next_button.grid_remove()
        self.feedback_title.configure(text="Evaluation error", foreground="#ff6666")
        self.feedback_label.configure(text=f"Failed to evaluate answer: {error}")
        self.feedback_label.grid()
        self.feedback_card.grid()
        self.continue_button.configure(text="Continue", command=self._on_continue_after_feedback)
        self.continue_button.grid()
    
    def _on_continue_after_feedback(self) -> None:
        """Continue to next card after viewing feedback."""
        self.controller.next_lesson_card()
    
    def _on_next_clicked(self) -> None:
        """Handle submit/next button click."""
        plan = self.controller.lesson_plan
        if not plan or self.controller.lesson_index >= len(plan.cards):
            return
        
        card = plan.cards[self.controller.lesson_index]
        
        # Get user response
        response = ""
        answer_index = None
        
        if card.type == "multiple_choice":
            if not hasattr(self, 'selected_mc_index') or self.selected_mc_index is None:
                messagebox.showinfo("Select an answer", "Please select an option.")
                return
            answer_index = self.selected_mc_index
            response = card.options[answer_index] if answer_index < len(card.options) else ""
        elif card.type == "vocabulary":
            # Vocabulary cards don't need submission, just continue
            self.controller.next_lesson_card()
            return
        else:
            if not hasattr(self, 'text_widget') or not self.text_widget:
                messagebox.showinfo("Enter an answer", "Please enter your answer.")
                return
            response = self.text_widget.get("1.0", tk.END).strip()
            if not response:
                messagebox.showinfo("Enter an answer", "Please enter your answer before submitting.")
                return
        
        # Submit response
        self.controller.submit_lesson_card_response(response, answer_index)
        # Store submit button reference
        self.submit_button = self.next_button


# ---------------------------------------------------------------------------
# Summary card
# ---------------------------------------------------------------------------

class SummaryCard(ttk.Frame):
    def __init__(self, parent, controller: LanguageBuddyApp) -> None:
        super().__init__(parent)
        self.controller = controller

        # Center content in window - add side spacers
        self.columnconfigure(0, weight=1)  # Left spacer
        self.columnconfigure(2, weight=1)  # Right spacer
        self.rowconfigure(0, weight=1)  # Top spacer
        self.rowconfigure(2, weight=1)  # Bottom spacer
        
        # Content wrapper - max width, centered
        content_frame = ttk.Frame(self)
        content_frame.grid(row=1, column=1, sticky="nsew")
        self.content = content_frame
        content_frame.columnconfigure(0, weight=1)
        content_frame.rowconfigure(0, weight=1)
        content_frame.rowconfigure(4, weight=1)

        title = ttk.Label(
            self.content,
            text="Session Summary",
            font=("Helvetica", 28, "bold"),
            foreground="#ffffff",  # White text on dark background
        )
        title.grid(row=1, column=0, pady=(30, 15))

        # Loading spinner
        self.loading_spinner = LoadingSpinner(self.content, text="Generating summary...")
        self.loading_spinner.grid(row=2, column=0, pady=(40, 40))

        self.summary_label = ttk.Label(
            self.content, 
            text="", 
            wraplength=900, 
            justify="center",
            font=("Helvetica", 14),
            foreground="#d0d0d0",  # Light gray text on dark background
        )
        self.summary_label.grid(row=2, column=0, padx=50, pady=(0, 30), sticky="ew")
        self.content.bind("<Configure>", self._on_summary_resize)

        button_frame = ttk.Frame(self.content)
        button_frame.grid(row=3, column=0, pady=(10, 30))

        again_button = ttk.Button(
            button_frame,
            text="Try another assessment",
            command=self.controller.restart,
        )
        again_button.grid(row=0, column=0, padx=10)

        exit_button = ttk.Button(
            button_frame,
            text="Exit",
            command=self.controller.destroy,
        )
        exit_button.grid(row=0, column=1, padx=10)

    def _on_summary_resize(self, event: tk.Event) -> None:
        """Adjust wraplength for summary when window resizes."""
        if event.widget is not self.content:
            return
        wrap = max(320, event.width - 120)
        self.summary_label.configure(wraplength=wrap)
    
    def show_loading(self, message: str = "Loading...") -> None:
        """Show loading spinner."""
        self.loading_spinner.text = message
        self.loading_spinner.label.configure(text=f"{self.loading_spinner.spinner_chars[0]} {message}")
        self.loading_spinner.start()
        self.summary_label.grid_remove()
    
    def hide_loading(self) -> None:
        """Hide loading spinner."""
        self.loading_spinner.stop()
        self.summary_label.grid()
    
    def show_error(self, error: str) -> None:
        """Show error message."""
        self.hide_loading()
        messagebox.showerror("Error", f"Failed to generate summary: {error}")

    def set_summary(self, summary: Dict[str, Any], lesson_plan: Optional[LessonPlan] = None) -> None:
        """Set summary from final summary dict."""
        self.hide_loading()
        
        if not summary:
            text = "No summary is available for this session."
        else:
            overall_score = summary.get("overall_score", 0)
            proficiency_improvement = summary.get("proficiency_improvement", "")
            strengths = summary.get("strengths", []) or []
            areas_to_improve = summary.get("areas_to_improve", []) or []
            study_suggestions = summary.get("study_suggestions", []) or []
            
            # Add lesson plan total score if available
            if lesson_plan:
                overall_score = lesson_plan.total_score

            text_lines = [
                f"Overall Score: {overall_score}/100",
                "",
                f"Progress: {proficiency_improvement}",
                "",
                "Strengths:",
                *[f"  • {s}" for s in (strengths or ["Keep up the good work!"])],
                "",
                "Areas to Improve:",
                *[f"  • {a}" for a in (areas_to_improve or ["Continue practicing"])],
                "",
                "Study Suggestions:",
                *[f"  • {s}" for s in (study_suggestions or ["Practice regularly", "Review vocabulary daily"])],
                "",
                "Would you like to try another assessment?",
            ]
            text = "\n".join(text_lines)

        self.summary_label.configure(text=text)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = LanguageBuddyApp()
    app.mainloop()

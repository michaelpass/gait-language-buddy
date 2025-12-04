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
import os

from typing import Optional, List, Dict, Any, Set, Callable

from PIL import Image, ImageTk

# Audio playback support
try:
    import pygame
    pygame.mixer.init()
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Note: pygame not installed. Audio playback will be disabled.")
    print("Install with: pip install pygame")

# Audio recording support for speaking exercises
RECORDING_AVAILABLE = False
RECORDING_ERROR = None
try:
    import sounddevice as sd
    import soundfile as sf
    import tempfile
    # Test that we can actually access audio devices
    devices = sd.query_devices()
    input_devices = [d for d in devices if d['max_input_channels'] > 0]
    if len(input_devices) == 0:
        RECORDING_ERROR = "No microphone found. Check System Preferences → Privacy → Microphone"
        print(f"Note: {RECORDING_ERROR}")
    else:
        RECORDING_AVAILABLE = True
        print(f"✓ Recording available: {len(input_devices)} microphone(s) found")
except ImportError as e:
    RECORDING_ERROR = f"Missing packages: {e}. Install with: pip install sounddevice soundfile"
    print(f"Note: {RECORDING_ERROR}")
except Exception as e:
    RECORDING_ERROR = f"Audio device error: {e}. On macOS, try: brew install portaudio"
    print(f"Note: {RECORDING_ERROR}")

from core.logger import logger
from core.api import (
    SUPPORTED_LANGUAGES,
    is_api_available,  # Check if API key is loaded
    generate_assessment_cards,
    evaluate_assessment_responses,
    generate_structured_lesson_plan,
    generate_lesson_plan_from_assessment_responses,  # Optimized combined function
    generate_teaching_content,  # Teaching content generation
    evaluate_card_response,
    generate_final_summary,
    generate_image_async,
    generate_images_parallel,  # Parallel image generation
    generate_speech_async,  # TTS generation
    transcribe_audio_async,  # STT transcription
)
from core.database import (
    initialize_database,
    get_db,
    VocabularyItem,
    SessionRecord,
    LanguageProfile,
)
from core.models import (
    AssessmentCard, AssessmentResult, LessonPlan, LessonCard,
    TeachingCard, TeachingPlan, SessionStats
)

# Log application startup
logger.banner("GAIT Language Buddy - Starting Application")


# ---------------------------------------------------------------------------
# Scrollable Frame Widget (for cards that may exceed window height)
# ---------------------------------------------------------------------------

class ScrollableFrame(ttk.Frame):
    """
    A frame that provides vertical scrolling for its content.
    Scrollbar auto-hides when content fits within the viewport.
    Content is centered both horizontally and vertically when it fits.
    
    Usage:
        scrollable = ScrollableFrame(parent)
        scrollable.pack(fill="both", expand=True)
        
        # Add widgets to scrollable.content instead of scrollable directly
        ttk.Label(scrollable.content, text="Hello").pack()
    """
    
    def __init__(self, parent, **kwargs) -> None:
        super().__init__(parent, **kwargs)
        
        # Track scrollbar visibility and layout state
        self._scrollbar_visible = False
        self._last_canvas_height = 0
        
        # Create canvas and scrollbar
        self.canvas = tk.Canvas(
            self,
            highlightthickness=0,
            background="#1e1e1e",
        )
        self.scrollbar = ttk.Scrollbar(
            self,
            orient="vertical",
            command=self.canvas.yview,
        )
        
        # Create an outer frame for vertical centering
        self._outer_frame = ttk.Frame(self.canvas)
        
        # Create the content frame inside the outer frame
        self.content = ttk.Frame(self._outer_frame)
        self.content.pack(expand=True)
        
        # Create window in canvas for outer frame
        self.content_window = self.canvas.create_window(
            (0, 0),
            window=self._outer_frame,
            anchor="nw",
        )
        
        # Configure scrollbar
        self.canvas.configure(yscrollcommand=self._on_scroll_set)
        
        # Layout - canvas fills space, scrollbar hidden initially
        self.canvas.pack(side="left", fill="both", expand=True)
        # Don't pack scrollbar initially - it will show when needed
        
        # Bind events
        self.content.bind("<Configure>", self._on_content_configure)
        self._outer_frame.bind("<Configure>", self._on_outer_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        
        # Bind mousewheel scrolling
        self._bind_mousewheel()
    
    def _on_scroll_set(self, first: str, last: str) -> None:
        """Handle scrollbar position updates - show/hide scrollbar dynamically."""
        first_f = float(first)
        last_f = float(last)
        
        # If the entire content is visible (first=0.0 and last=1.0), hide scrollbar
        needs_scrollbar = not (first_f <= 0.0 and last_f >= 1.0)
        
        if needs_scrollbar and not self._scrollbar_visible:
            self.scrollbar.pack(side="right", fill="y")
            self._scrollbar_visible = True
        elif not needs_scrollbar and self._scrollbar_visible:
            self.scrollbar.pack_forget()
            self._scrollbar_visible = False
        
        # Update scrollbar position
        self.scrollbar.set(first, last)
    
    def _on_content_configure(self, event: tk.Event) -> None:
        """Update layout when content changes."""
        self.after_idle(self._update_layout)
    
    def _on_outer_configure(self, event: tk.Event) -> None:
        """Update scroll region when outer frame changes."""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def _on_canvas_configure(self, event: tk.Event) -> None:
        """Update layout when canvas is resized."""
        self._update_layout()
    
    def _update_layout(self) -> None:
        """Update content position and scroll region for proper centering."""
        try:
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                return
            
            # Get content dimensions
            self.content.update_idletasks()
            content_height = self.content.winfo_reqheight()
            
            # Determine if scrolling is needed
            needs_scrolling = content_height > canvas_height
            
            # Set outer frame dimensions
            # Width always matches canvas for horizontal centering of children
            # Height depends on whether we need scrolling
            if needs_scrolling:
                # Content overflows - outer frame matches content height
                outer_height = content_height
            else:
                # Content fits - outer frame matches canvas for vertical centering
                outer_height = canvas_height
            
            # Update the canvas window size
            self.canvas.itemconfigure(
                self.content_window, 
                width=canvas_width,
                height=outer_height
            )
            
            # Update scroll region
            self.canvas.configure(scrollregion=(0, 0, canvas_width, outer_height))
            
            # Check scrollbar visibility
            self._check_scrollbar_visibility()
            
        except Exception:
            pass
    
    def _check_scrollbar_visibility(self) -> None:
        """Check if scrollbar should be visible based on content vs canvas size."""
        try:
            canvas_height = self.canvas.winfo_height()
            self.content.update_idletasks()
            content_height = self.content.winfo_reqheight()
            
            needs_scrollbar = content_height > canvas_height
            
            if needs_scrollbar and not self._scrollbar_visible:
                self.scrollbar.pack(side="right", fill="y")
                self._scrollbar_visible = True
            elif not needs_scrollbar and self._scrollbar_visible:
                self.scrollbar.pack_forget()
                self._scrollbar_visible = False
        except Exception:
            pass
    
    def _bind_mousewheel(self) -> None:
        """Bind mousewheel events for scrolling."""
        # Bind to canvas and all children
        self.canvas.bind("<Enter>", self._bind_wheel_events)
        self.canvas.bind("<Leave>", self._unbind_wheel_events)
    
    def _bind_wheel_events(self, event: tk.Event) -> None:
        """Bind mousewheel when mouse enters."""
        # Windows and macOS have different mousewheel events
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        # Linux uses Button-4 and Button-5
        self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux)
    
    def _unbind_wheel_events(self, event: tk.Event) -> None:
        """Unbind mousewheel when mouse leaves."""
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")
    
    def _on_mousewheel(self, event: tk.Event) -> None:
        """Handle mousewheel on Windows/macOS."""
        # Only scroll if scrollbar is visible (content overflows)
        if not self._scrollbar_visible:
            return
        # event.delta is positive for scroll up, negative for scroll down
        # Windows: delta is typically 120 or -120
        # macOS: delta can be smaller values
        if event.delta:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    def _on_mousewheel_linux(self, event: tk.Event) -> None:
        """Handle mousewheel on Linux."""
        # Only scroll if scrollbar is visible (content overflows)
        if not self._scrollbar_visible:
            return
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")
    
    def scroll_to_top(self) -> None:
        """Scroll content to top and re-center."""
        self.canvas.yview_moveto(0)
        # Re-center content after scrolling
        self.after_idle(self._update_layout)


# ---------------------------------------------------------------------------
# Responsive Image Widget (resizes with window)
# ---------------------------------------------------------------------------

class ResponsiveImage(ttk.Frame):
    """
    An image widget that resizes dynamically with the window.
    Targets approximately 1/3 of window height while maintaining aspect ratio.
    """
    
    # Minimum dimensions to ensure visibility
    MIN_WIDTH = 150
    MIN_HEIGHT = 100
    # Maximum dimensions to prevent excessive size
    MAX_WIDTH = 800
    MAX_HEIGHT = 600
    # Target fraction of window height
    TARGET_HEIGHT_FRACTION = 0.33
    
    def __init__(self, parent, **kwargs) -> None:
        super().__init__(parent, **kwargs)
        
        self._original_image: Optional[Image.Image] = None
        self._image_path: Optional[str] = None
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._last_window_height: int = 0
        
        # Label to display the image
        self.image_label = ttk.Label(self)
        self.image_label.pack(fill="both", expand=True)
        
        # Bind resize event to self and schedule binding to root window
        self.bind("<Configure>", self._on_resize)
        self.after(100, self._bind_to_root_window)
    
    def _bind_to_root_window(self) -> None:
        """Bind to root window configure event to catch window resizes."""
        root = self._get_root_window()
        if root:
            root.bind("<Configure>", self._on_window_resize, add="+")
    
    def _get_root_window(self) -> Optional[tk.Tk]:
        """Get the root Tk window."""
        widget = self
        while widget.master:
            widget = widget.master
        return widget if isinstance(widget, tk.Tk) else None
    
    def set_image(self, image_path: str) -> bool:
        """
        Load and display an image from the given path.
        Returns True if successful, False otherwise.
        """
        try:
            self._image_path = image_path
            self._original_image = Image.open(image_path)
            self._update_display()
            return True
        except Exception:
            self._original_image = None
            self._image_path = None
            self.image_label.configure(text="[Could not display image]", image="")
            return False
    
    def clear_image(self) -> None:
        """Clear the current image."""
        self._original_image = None
        self._image_path = None
        self._photo = None
        self.image_label.configure(text="", image="")
    
    def set_placeholder(self, text: str = "Loading image...") -> None:
        """Show placeholder text instead of image."""
        self._original_image = None
        self._photo = None
        self.image_label.configure(text=text, image="")
    
    def _on_resize(self, event: tk.Event) -> None:
        """Handle widget resize - update image to fit new size."""
        if self._original_image:
            self._check_and_update()
    
    def _on_window_resize(self, event: tk.Event) -> None:
        """Handle window resize - update image if visible."""
        if self._original_image and self.winfo_viewable():
            self._check_and_update()
    
    def _check_and_update(self) -> None:
        """Check if window height changed significantly and update if needed."""
        root = self._get_root_window()
        if root:
            window_height = root.winfo_height()
            # Only redraw if window height changed by more than 20px
            if abs(window_height - self._last_window_height) > 20:
                self._last_window_height = window_height
                self._update_display()
    
    def _update_display(self) -> None:
        """Update the displayed image to fit ~1/3 of window height."""
        if not self._original_image:
            return
        
        # Get window dimensions
        root = self._get_root_window()
        if root:
            window_height = root.winfo_height()
            window_width = root.winfo_width()
        else:
            window_height = 750  # Default fallback
            window_width = 850
        
        # Target height is 1/3 of window height
        target_height = int(window_height * self.TARGET_HEIGHT_FRACTION)
        # Target width based on available space (with padding)
        target_width = int(window_width * 0.7)
        
        # Apply min/max constraints
        target_height = max(self.MIN_HEIGHT, min(target_height, self.MAX_HEIGHT))
        target_width = max(self.MIN_WIDTH, min(target_width, self.MAX_WIDTH))
        
        # Calculate size maintaining aspect ratio
        orig_width, orig_height = self._original_image.size
        
        # Scale to fit within target dimensions
        width_ratio = target_width / orig_width
        height_ratio = target_height / orig_height
        ratio = min(width_ratio, height_ratio)
        
        new_width = max(self.MIN_WIDTH, int(orig_width * ratio))
        new_height = max(self.MIN_HEIGHT, int(orig_height * ratio))
        
        # Resize image using high-quality resampling
        resized = self._original_image.copy()
        resized.thumbnail((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create PhotoImage and update label
        self._photo = ImageTk.PhotoImage(resized)
        self.image_label.configure(image=self._photo, text="")
        
        # Store current window height for change detection
        self._last_window_height = window_height


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
# Audio Player Widget
# ---------------------------------------------------------------------------

class AudioPlayer(ttk.Frame):
    """
    A widget for playing audio files with a single play/pause button.
    Uses pygame for cross-platform audio playback.
    """
    
    def __init__(self, parent, on_skip: Optional[Callable[[], None]] = None, **kwargs) -> None:
        super().__init__(parent, **kwargs)
        
        self._audio_path: Optional[str] = None
        self._is_playing: bool = False
        self._is_loaded: bool = False
        self._playback_finished: bool = False
        self._on_skip = on_skip
        
        # Create control frame
        self.control_frame = ttk.Frame(self)
        self.control_frame.pack(pady=10)
        
        # Single play/pause/replay button
        self.play_button = ttk.Button(
            self.control_frame,
            text="▶ Play Audio",
            command=self._on_play_click,
            width=18,
        )
        self.play_button.pack(side="left", padx=5)
        
        # Skip button - for users who can't listen to audio
        self.skip_button = ttk.Button(
            self.control_frame,
            text="⏭ Skip Audio",
            command=self._on_skip_click,
            width=12,
        )
        self.skip_button.pack(side="left", padx=5)
        
        # Status label
        self.status_label = ttk.Label(
            self,
            text="",
            font=("Helvetica", 11),
            foreground="#7bb3ff",
            anchor="center",
        )
        self.status_label.pack(pady=(5, 0))
        
        # Skip info label
        self.skip_info_label = ttk.Label(
            self,
            text="Can't listen? Skip this exercise →",
            font=("Helvetica", 10),
            foreground="#888888",
            anchor="center",
        )
        self.skip_info_label.pack(pady=(2, 0))
        
        # Loading indicator
        self.loading_label = ttk.Label(
            self,
            text="⏳ Generating audio...",
            font=("Helvetica", 12),
            foreground="#7bb3ff",
            anchor="center",
        )
        self.loading_label.pack(pady=10)
        
        # Initially show loading, hide controls
        self._show_loading()
    
    def _on_skip_click(self) -> None:
        """Handle skip button click."""
        logger.ui("User skipped audio exercise")
        self.stop()
        if self._on_skip:
            self._on_skip()
    
    def _show_loading(self) -> None:
        """Show loading state."""
        self.loading_label.pack(pady=10)
        self.control_frame.pack_forget()
        self.status_label.pack_forget()
        self.skip_info_label.pack_forget()
    
    def _show_controls(self) -> None:
        """Show playback controls."""
        self.loading_label.pack_forget()
        self.control_frame.pack(pady=10)
        self.status_label.pack(pady=(5, 0))
        self.skip_info_label.pack(pady=(2, 0))
    
    def set_audio(self, audio_path: str) -> bool:
        """
        Load an audio file for playback.
        Returns True if successful.
        """
        if not AUDIO_AVAILABLE:
            self.status_label.configure(text="Audio playback not available")
            self._show_controls()
            return False
        
        if not audio_path or not os.path.exists(audio_path):
            self.status_label.configure(text="Audio file not found")
            self._show_controls()
            return False
        
        try:
            self._audio_path = audio_path
            pygame.mixer.music.load(audio_path)
            self._is_loaded = True
            self._is_playing = False
            self._playback_finished = False
            self.play_button.configure(text="▶ Play Audio")
            self.status_label.configure(text="Click to listen")
            self._show_controls()
            logger.ui(f"Audio loaded: {audio_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            self.status_label.configure(text="Failed to load audio")
            self._show_controls()
            return False
    
    def set_loading(self, message: str = "Generating audio...") -> None:
        """Show loading state with custom message."""
        self.loading_label.configure(text=f"⏳ {message}")
        self._show_loading()
    
    def _on_play_click(self) -> None:
        """Handle play/pause/replay button click."""
        if not AUDIO_AVAILABLE or not self._is_loaded:
            return
        
        if self._is_playing:
            # Pause
            pygame.mixer.music.pause()
            self._is_playing = False
            self.play_button.configure(text="▶ Resume")
            self.status_label.configure(text="Paused")
        elif self._playback_finished:
            # Replay from beginning
            pygame.mixer.music.play()
            self._is_playing = True
            self._playback_finished = False
            self.play_button.configure(text="⏸ Pause")
            self.status_label.configure(text="Playing...")
            self._check_playback_status()
        else:
            # Play or unpause
            if pygame.mixer.music.get_pos() == -1:
                # Not started yet, play from beginning
                pygame.mixer.music.play()
            else:
                # Was paused, unpause
                pygame.mixer.music.unpause()
            self._is_playing = True
            self.play_button.configure(text="⏸ Pause")
            self.status_label.configure(text="Playing...")
            
            # Schedule check for when playback ends
            self._check_playback_status()
    
    def _check_playback_status(self) -> None:
        """Check if audio is still playing and update UI."""
        if not self._is_playing:
            return
        
        if not pygame.mixer.music.get_busy():
            # Playback finished
            self._is_playing = False
            self._playback_finished = True
            self.play_button.configure(text="▶ Play Again")
            self.status_label.configure(text="Finished - click to replay")
        else:
            # Still playing, check again later
            self.after(200, self._check_playback_status)
    
    def stop(self) -> None:
        """Stop playback."""
        if AUDIO_AVAILABLE and self._is_loaded:
            pygame.mixer.music.stop()
            self._is_playing = False
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.stop()
        if self._audio_path and os.path.exists(self._audio_path):
            try:
                # Unload the music first
                if AUDIO_AVAILABLE:
                    pygame.mixer.music.unload()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Speech Recorder Widget
# ---------------------------------------------------------------------------

class SpeechRecorder(ttk.Frame):
    """
    A widget for recording user speech for speaking exercises.
    Records audio and provides callback with the recorded file path.
    Uses click-to-start/click-to-stop for better reliability.
    """
    
    SAMPLE_RATE = 16000  # Whisper prefers 16kHz
    CHANNELS = 1  # Mono
    
    def __init__(self, parent, on_recording_complete: Optional[Callable[[str], None]] = None, 
                 on_skip: Optional[Callable[[], None]] = None, **kwargs) -> None:
        super().__init__(parent, **kwargs)
        
        self._is_recording: bool = False
        self._recording_data: List = []
        self._recording_path: Optional[str] = None
        self._on_complete = on_recording_complete
        self._on_skip = on_skip
        self._record_thread: Optional[threading.Thread] = None
        
        # Prompt label (what to say)
        self.prompt_label = ttk.Label(
            self,
            text="",
            font=("Helvetica", 16, "bold"),
            foreground="#ffffff",
            wraplength=500,
            justify="center",
            anchor="center",
        )
        self.prompt_label.pack(pady=(10, 15))
        
        # Button frame to hold record and skip buttons
        self.button_frame = ttk.Frame(self)
        self.button_frame.pack(pady=10)
        
        # Record button - use tk.Button for better event handling
        # Using click-to-toggle instead of hold-to-record for reliability
        self.record_button = tk.Button(
            self.button_frame,
            text="🎤 Click to Start Recording",
            font=("Helvetica", 13, "bold"),
            bg="#1a1a1a",  # Very dark gray background
            fg="#e0e0e0",  # Light gray text for contrast
            activebackground="#5a9bd4",  # Light blue when pressed
            activeforeground="#000000",  # Black text when pressed
            highlightbackground="#1a1a1a",
            highlightcolor="#5a9bd4",
            relief="raised",
            borderwidth=3,
            padx=20,
            pady=10,
            cursor="hand2",
            command=self._toggle_recording,
        )
        self.record_button.pack(side="left", padx=5)
        
        # Skip button - for users who can't speak/record
        self.skip_button = ttk.Button(
            self.button_frame,
            text="⏭ Skip Speaking",
            command=self._on_skip_click,
            width=14,
        )
        self.skip_button.pack(side="left", padx=5)
        
        # Status label
        self.status_label = ttk.Label(
            self,
            text="Click the button to start recording",
            font=("Helvetica", 11),
            foreground="#7bb3ff",
            anchor="center",
        )
        self.status_label.pack(pady=(5, 5))
        
        # Skip info label
        self.skip_info_label = ttk.Label(
            self,
            text="Can't speak right now? Skip this exercise →",
            font=("Helvetica", 10),
            foreground="#888888",
            anchor="center",
        )
        self.skip_info_label.pack(pady=(0, 5))
        
        # Transcription result (shown after recording)
        self.transcription_label = ttk.Label(
            self,
            text="",
            font=("Helvetica", 13),
            foreground="#9bc6ff",
            wraplength=500,
            justify="center",
            anchor="center",
        )
        self.transcription_label.pack(pady=(5, 0))
        self.transcription_label.pack_forget()
        
        # Check if recording is available
        if not RECORDING_AVAILABLE:
            self.record_button.configure(state="disabled")
            error_msg = RECORDING_ERROR or "Recording not available. Install: pip install sounddevice soundfile"
            self.status_label.configure(
                text=error_msg,
                foreground="#ff6b6b"
            )
            # Also hide the skip info since we're showing an error
            self.skip_info_label.configure(text="Skip this exercise instead →")
    
    def _on_skip_click(self) -> None:
        """Handle skip button click."""
        logger.ui("User skipped speaking exercise")
        # Stop any ongoing recording
        if self._is_recording:
            self._is_recording = False
        if self._on_skip:
            self._on_skip()
    
    def _toggle_recording(self) -> None:
        """Toggle recording on/off with button click."""
        if self._is_recording:
            self._stop_recording()
        else:
            self._start_recording()
    
    def set_prompt(self, prompt_text: str) -> None:
        """Set the phrase the user should say."""
        self.prompt_label.configure(text=f'"{prompt_text}"')
        self.transcription_label.pack_forget()
        self.status_label.configure(text="Click the button to start recording", foreground="#7bb3ff")
        self.record_button.configure(
            text="🎤 Click to Start Recording",
            bg="#1a1a1a",  # Very dark gray
            fg="#e0e0e0",  # Light text
        )
    
    def _start_recording(self) -> None:
        """Start recording audio."""
        if not RECORDING_AVAILABLE or self._is_recording:
            return
        
        logger.ui("Starting speech recording...")
        self._is_recording = True
        self._recording_data = []
        self.record_button.configure(
            text="🔴 Click to Stop Recording",
            bg="#8B0000",  # Dark red background
        )
        self.status_label.configure(text="Recording... Click button when done", foreground="#ff6b6b")
        
        # Start recording in a thread
        def record_audio():
            try:
                # Record until stopped
                with sd.InputStream(samplerate=self.SAMPLE_RATE, channels=self.CHANNELS, 
                                   callback=self._audio_callback):
                    while self._is_recording:
                        sd.sleep(100)
            except Exception as e:
                logger.error(f"Recording error: {e}")
                self.after(0, lambda: self.status_label.configure(
                    text=f"Recording error: {e}", foreground="#ff6b6b"
                ))
        
        self._record_thread = threading.Thread(target=record_audio, daemon=True)
        self._record_thread.start()
    
    def _audio_callback(self, indata, frames, time, status):
        """Callback for audio input stream."""
        if status:
            logger.warning(f"Audio status: {status}")
        self._recording_data.append(indata.copy())
    
    def _stop_recording(self) -> None:
        """Stop recording and save the audio."""
        if not self._is_recording:
            return
        
        logger.ui("Stopping speech recording...")
        self._is_recording = False
        self.record_button.configure(
            text="🎤 Click to Start Recording",
            bg="#1a1a1a",  # Very dark gray
            fg="#e0e0e0",  # Light text
        )
        self.status_label.configure(text="Processing...", foreground="#7bb3ff")
        
        # Save recording in a thread
        def save_and_callback():
            try:
                if not self._recording_data:
                    self.after(0, lambda: self._on_no_audio())
                    return
                
                import numpy as np
                # Concatenate all recorded chunks
                audio_data = np.concatenate(self._recording_data, axis=0)
                
                # Check if there's actually audio data
                if len(audio_data) < self.SAMPLE_RATE * 0.5:  # Less than 0.5 seconds
                    self.after(0, lambda: self._on_no_audio())
                    return
                
                # Save to temp file
                fd, path = tempfile.mkstemp(suffix=".wav", prefix="gait_recording_")
                os.close(fd)
                sf.write(path, audio_data, self.SAMPLE_RATE)
                
                self._recording_path = path
                duration = len(audio_data) / self.SAMPLE_RATE
                logger.ui(f"Recording saved: {path} ({duration:.1f}s)")
                
                self.after(0, lambda: self.status_label.configure(
                    text=f"Recording complete ({duration:.1f}s)! Transcribing...", foreground="#7bb3ff"
                ))
                
                # Call the completion callback
                if self._on_complete:
                    self._on_complete(path)
                    
            except Exception as e:
                logger.error(f"Save recording error: {e}")
                self.after(0, lambda: self.status_label.configure(
                    text=f"Error saving: {e}", foreground="#ff6b6b"
                ))
        
        threading.Thread(target=save_and_callback, daemon=True).start()
    
    def _on_no_audio(self) -> None:
        """Called when no audio was recorded."""
        self.status_label.configure(
            text="No audio recorded. Click the button and speak clearly.", 
            foreground="#ff6b6b"
        )
        self.record_button.configure(
            text="🎤 Click to Start Recording",
            bg="#1a1a1a",
            fg="#e0e0e0",
        )
    
    def show_transcription(self, text: str) -> None:
        """Show the transcription result."""
        self.transcription_label.configure(text=f'You said: "{text}"')
        self.transcription_label.pack(pady=(5, 0))
        self.status_label.configure(text="Transcription complete", foreground="#7bb3ff")
    
    def show_error(self, message: str) -> None:
        """Show an error message."""
        self.status_label.configure(text=message, foreground="#ff6b6b")
    
    def get_recording_path(self) -> Optional[str]:
        """Get the path to the recorded audio file."""
        return self._recording_path
    
    def reset(self) -> None:
        """Reset the recorder for a new recording."""
        self._is_recording = False
        self._recording_data = []
        self._recording_path = None
        self.transcription_label.pack_forget()
        self.status_label.configure(text="Click the button to start recording", foreground="#7bb3ff")
        self.record_button.configure(
            text="🎤 Click to Start Recording",
            bg="#1a1a1a",
            fg="#e0e0e0",
        )


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class LanguageBuddyApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        logger.ui("Initializing LanguageBuddyApp window...")

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
        # Allow smaller windows - scrolling will handle overflow
        self.min_app_width = 400
        self.min_app_height = 300
        self.minsize(self.min_app_width, self.min_app_height)
        
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
        self.teaching_plan: Optional[TeachingPlan] = None  # Teaching content before quiz
        self.lesson_plan: Optional[LessonPlan] = None  # Quiz cards
        self.lesson_index: int = 0
        self.final_summary: Optional[Dict[str, Any]] = None
        self.pending_assessment_images: Set[str] = set()
        self.session_stats: Optional[SessionStats] = None  # For tracking changes
        
        # Image cache
        self.image_cache: Dict[str, str] = {}  # image_prompt -> image_path
        self.image_photos: Dict[str, ImageTk.PhotoImage] = {}  # image_path -> PhotoImage
        
        # Database and session tracking
        self._init_database()
        self.current_session: Optional[SessionRecord] = None
        self.session_vocabulary: List[str] = []  # Words encountered this session

        # Container for "cards"
        container = ttk.Frame(self)
        container.pack(fill="both", expand=True)
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        self.cards = {}

        logger.ui("Creating card views...")
        for CardClass in (IntroCard, AssessmentCardView, AssessmentResultsCard, TeachingCardView, LessonCardView, SummaryCard):
            card = CardClass(parent=container, controller=self)
            self.cards[CardClass.__name__] = card
            card.grid(row=0, column=0, sticky="nsew")
            logger.debug(f"  Created: {CardClass.__name__}")

        logger.ui("Application initialized successfully")
        self.show_card("IntroCard")
    
    def _init_database(self) -> None:
        """Initialize database connection."""
        logger.task_start("database_initialization")
        self.database_connected = False
        self.database_error: Optional[str] = None
        
        success = initialize_database()
        if success:
            logger.task_complete("database_initialization")
            # Load or create default user
            db = get_db()
            user = db.get_or_create_user()
            if user:
                logger.success(f"Database user loaded: {user.user_id}")
                self.database_connected = True
            else:
                logger.warning("Database connected but failed to load user")
                self.database_error = "Connected but failed to load user profile"
        else:
            logger.warning("Database not connected. Progress will not be saved.")
            # Check for specific error reasons
            import os
            creds_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
            if not creds_path:
                self.database_error = "FIREBASE_CREDENTIALS_PATH not set in .env"
            elif not os.path.exists(creds_path):
                self.database_error = f"Credentials file not found: {creds_path}"
            else:
                self.database_error = "Failed to connect to Firebase"
    
    def _track_vocabulary_from_card(self, card: LessonCard, evaluation: Dict[str, Any]) -> None:
        """Track vocabulary encountered in a lesson card."""
        db = get_db()
        language = self.selected_language or "Spanish"
        
        # Get words to track from card
        words_to_track = []
        
        # Main word from the card (for vocabulary cards)
        if card.word:
            words_to_track.append({
                "word": card.word,
                "translation": card.translation or "",
                "example": card.example or "",
            })
        
        # Correct answer might be a vocabulary word
        if card.correct_answer and card.type in ("image_question", "fill_in_blank"):
            words_to_track.append({
                "word": card.correct_answer,
                "translation": "",  # We don't always have translation
                "example": card.question or "",
            })
        
        # Vocabulary expansion words
        for word in (evaluation.get("vocabulary_expansion") or card.vocabulary_expansion or []):
            if word:
                words_to_track.append({
                    "word": word,
                    "translation": "",
                    "example": "",
                })
        
        # Track each word
        for word_data in words_to_track:
            word = word_data["word"]
            if not word or len(word) < 2:
                continue
            
            # Track in session
            if word not in self.session_vocabulary:
                self.session_vocabulary.append(word)
            
            # Get or create vocabulary item
            vocab_item = db.get_vocabulary_item(word, language)
            if not vocab_item:
                vocab_item = VocabularyItem(
                    word=word,
                    translation=word_data.get("translation", ""),
                    language=language,
                )
                if self.current_session:
                    self.current_session.new_vocabulary.append(word)
            
            # Record encounter
            vocab_item.record_encounter(
                correct=card.is_correct or False,
                card_type=card.type,
                example=word_data.get("example", ""),
            )
            
            # Save to database
            db.save_vocabulary_item(vocab_item)
        
        logger.debug(f"Tracked {len(words_to_track)} vocabulary words from card")
    
    def _save_session_to_database(self) -> None:
        """Save session record and update language profile."""
        if not self.current_session:
            return
        
        from datetime import datetime, timezone
        
        db = get_db()
        language = self.selected_language or "Spanish"
        
        # Finalize session record
        self.current_session.ended_at = datetime.now(timezone.utc).isoformat()
        
        # Calculate session stats
        if self.lesson_plan:
            self.current_session.cards_completed = len(self.lesson_plan.cards)
            self.current_session.cards_correct = sum(
                1 for card in self.lesson_plan.cards if card.is_correct
            )
            self.current_session.total_score = self.lesson_plan.total_score
        
        self.current_session.vocabulary_practiced = self.session_vocabulary
        
        # Calculate duration
        try:
            start = datetime.fromisoformat(self.current_session.started_at.replace('Z', '+00:00'))
            end = datetime.fromisoformat(self.current_session.ended_at.replace('Z', '+00:00'))
            self.current_session.duration_minutes = int((end - start).total_seconds() / 60)
        except Exception:
            pass
        
        # Set proficiency at end
        if self.assessment_result:
            self.current_session.proficiency_at_end = self.assessment_result.proficiency
        
        # Save session
        db.save_session(self.current_session)
        logger.success(f"Session saved: {self.current_session.session_id}")
        
        # Update language profile
        lang_profile = db.get_language_profile(language)
        if self.assessment_result:
            lang_profile.overall_proficiency = self.assessment_result.proficiency
            lang_profile.vocabulary_level = self.assessment_result.vocabulary_level
            lang_profile.grammar_level = self.assessment_result.grammar_level
            lang_profile.fluency_score = self.assessment_result.fluency_score
            lang_profile.strengths = self.assessment_result.strengths
            lang_profile.weaknesses = self.assessment_result.weaknesses
            lang_profile.recommendations = self.assessment_result.recommendations
            lang_profile.last_assessment_date = datetime.now(timezone.utc).isoformat()
        
        lang_profile.total_sessions += 1
        lang_profile.total_cards_completed += self.current_session.cards_completed
        lang_profile.total_vocabulary_learned = len(db.get_all_vocabulary(language))
        lang_profile.total_time_minutes += self.current_session.duration_minutes
        
        # Update streak
        lang_profile.last_practice_date = datetime.now(timezone.utc).isoformat()
        # Simple streak logic - would need more sophistication for real tracking
        lang_profile.current_streak_days = min(lang_profile.current_streak_days + 1, 365)
        lang_profile.longest_streak_days = max(
            lang_profile.longest_streak_days, 
            lang_profile.current_streak_days
        )
        
        db.update_language_profile(lang_profile)
        logger.success(f"Language profile updated: {lang_profile.overall_proficiency}")

    # Card management -------------------------------------------------------

    def show_card(self, name: str) -> None:
        logger.ui_transition("current_card", name)
        card = self.cards[name]
        card.tkraise()

    # High-level flow methods ----------------------------------------------

    def start_session(self, language: str) -> None:
        """Triggered from Intro card when user picks a language."""
        logger.separator(f"Starting New Session - {language}")
        logger.ui(f"User selected language: {language}")
        
        self.selected_language = language
        self.assessment_cards = []
        self.assessment_stage = 0
        self.assessment_responses = []
        self.assessment_result = None
        self.lesson_plan = None
        self.lesson_index = 0
        self.final_summary = None
        self.session_vocabulary = []
        
        # Create session record
        import uuid
        from datetime import datetime, timezone
        session_id = str(uuid.uuid4())[:8]
        self.current_session = SessionRecord(
            session_id=session_id,
            started_at=datetime.now(timezone.utc).isoformat(),
            language=language,
            session_type="lesson",
        )
        
        # Check for existing language profile
        db = get_db()
        lang_profile = db.get_language_profile(language)
        
        if lang_profile and lang_profile.total_sessions > 0:
            # Returning learner - skip assessment and go directly to lessons
            self.current_session.proficiency_at_start = lang_profile.overall_proficiency
            logger.ui(f"Returning learner: {lang_profile.overall_proficiency} level, {lang_profile.total_sessions} sessions")
            logger.ui("Skipping assessment, generating lessons based on existing profile...")
            
            # Create assessment result from existing profile
            self.assessment_result = AssessmentResult(
                proficiency=lang_profile.overall_proficiency,
                vocabulary_level=lang_profile.vocabulary_level,
                grammar_level=lang_profile.grammar_level,
                fluency_score=lang_profile.fluency_score,
                strengths=lang_profile.strengths,
                weaknesses=lang_profile.weaknesses,
                recommendations=lang_profile.recommendations,
            )
            
            # Go directly to lesson generation
            self._generate_lessons_for_returning_learner()
        else:
            # New learner - run full assessment
            logger.ui("New learner - starting comprehensive assessment...")
            self.current_session.session_type = "assessment"
            
            # Show loading spinner and switch to assessment card
            assessment_card: AssessmentCardView = self.cards["AssessmentCardView"]
            assessment_card.show_loading("Generating assessment questions (10-12 questions)...")
            self.show_card("AssessmentCardView")

            # Run API calls in a background thread
            def generate_assessment_threaded():
                logger.task_start("generate_assessment_threaded")
                try:
                    # Generate all assessment cards (comprehensive 10-12 question assessment)
                    assessment_cards = generate_assessment_cards(language)
                    logger.task_complete("generate_assessment_threaded")

                    # Show questions ASAP; images are fetched per-card on demand
                    self.after(0, lambda: self._on_assessment_generated(assessment_cards))

                except Exception as e:
                    logger.task_error("generate_assessment_threaded", str(e))
                    # Show error in main thread
                    self.after(0, lambda: self._on_assessment_error(str(e)))

            thread = threading.Thread(target=generate_assessment_threaded, daemon=True)
            thread.start()
            logger.task("Spawned background thread for assessment generation")
    
    def _generate_lessons_for_returning_learner(self) -> None:
        """Generate teaching and quiz content for a returning learner (skip assessment)."""
        # Show loading on teaching card
        teaching_card: TeachingCardView = self.cards["TeachingCardView"]
        teaching_card.show_loading(f"Preparing lessons for {self.assessment_result.proficiency} level...")
        self.show_card("TeachingCardView")
        
        # Store learner context for later use
        self._learner_context = None
        
        # Track batches for progressive loading
        self._teaching_batches: List[TeachingCard] = []
        self._first_batch_shown = False
        
        def on_batch_ready(cards: List[TeachingCard], batch_num: int, total_batches: int) -> None:
            """Called when each batch of teaching cards is ready."""
            self._teaching_batches.extend(cards)
            
            # Update loading message with progress
            progress_msg = f"Generating lessons... ({batch_num}/{total_batches} batches)"
            self.after(0, lambda: teaching_card.update_loading_progress(
                progress_msg, batch_num, total_batches, len(self._teaching_batches)
            ))
            
            # Show first batch immediately (don't wait for all batches)
            if batch_num == 1 and not self._first_batch_shown:
                self._first_batch_shown = True
                # Create partial teaching plan with first batch
                self.teaching_plan = TeachingPlan(
                    cards=list(self._teaching_batches),
                    proficiency_target=self.assessment_result.proficiency,
                    theme="Loading...",
                    new_words_count=sum(1 for c in self._teaching_batches if c.is_new),
                    review_words_count=sum(1 for c in self._teaching_batches if c.is_review),
                )
                self.after(0, self._on_first_teaching_batch_ready)
            else:
                # Update existing teaching plan with new cards
                self.after(0, lambda: self._on_teaching_batch_added(cards))
        
        def generate_teaching_content_threaded():
            """Generate teaching content in batches for progressive loading."""
            logger.task_start("generate_teaching_content")
            try:
                # Get learner context from database
                db = get_db()
                if db.is_connected():
                    self._learner_context = db.generate_llm_context_string(
                        self.selected_language or "Spanish"
                    )
                    logger.debug(f"Loaded learner context: {len(self._learner_context)} chars")
                
                # Generate teaching content in batches with callback
                logger.ui("Generating teaching content in batches...")
                teaching_plan = generate_teaching_content(
                    language=self.selected_language or "Spanish",
                    proficiency=self.assessment_result.proficiency,
                    learner_context=self._learner_context,
                    num_new_words=12,
                    num_review_words=5,
                    on_batch_ready=on_batch_ready,  # Progressive loading callback
                )
                
                self.teaching_plan = teaching_plan
                logger.task_complete("generate_teaching_content")
                logger.ui(f"Teaching content ready: {len(teaching_plan.cards)} cards")
                
                # Finalize teaching cards display
                self.after(0, self._on_teaching_content_ready)
                
                # Start quiz generation in background
                self.after(0, self._start_quiz_generation_background)
                
            except Exception as e:
                logger.task_error("generate_teaching_content", str(e))
                self.after(0, lambda: teaching_card.show_loading(f"Error: {str(e)}"))
        
        thread = threading.Thread(target=generate_teaching_content_threaded, daemon=True)
        thread.start()
        logger.task("Spawned background thread for teaching content generation")
    
    def _on_first_teaching_batch_ready(self) -> None:
        """Called when the first batch of teaching cards is ready - show immediately."""
        if not self.teaching_plan or not self.teaching_plan.cards:
            logger.error("No teaching cards in first batch")
            return
        
        logger.ui(f"First batch ready: {len(self.teaching_plan.cards)} cards - showing immediately!")
        self._capture_initial_stats()
        
        # Show teaching cards immediately (more batches may still be loading)
        teaching_card: TeachingCardView = self.cards["TeachingCardView"]
        teaching_card.set_teaching_plan(self.teaching_plan, more_loading=True)
    
    def _on_teaching_batch_added(self, new_cards: List[TeachingCard]) -> None:
        """Called when additional teaching cards are added from a new batch."""
        if not self.teaching_plan:
            return
        
        # Update existing plan with new cards
        self.teaching_plan.cards.extend(new_cards)
        self.teaching_plan.new_words_count = sum(1 for c in self.teaching_plan.cards if c.is_new)
        self.teaching_plan.review_words_count = sum(1 for c in self.teaching_plan.cards if c.is_review)
        
        logger.ui(f"Added {len(new_cards)} cards - total now: {len(self.teaching_plan.cards)}")
        
        # Update UI
        teaching_card: TeachingCardView = self.cards["TeachingCardView"]
        teaching_card.update_card_count(len(self.teaching_plan.cards))
        
        # Start generating images/audio for new cards
        teaching_card.add_cards_to_generation(new_cards)
    
    def _on_teaching_content_ready(self) -> None:
        """Called when all teaching content is ready."""
        if not self.teaching_plan:
            logger.error("Teaching plan not available")
            return
        
        logger.ui(f"All teaching content ready: {len(self.teaching_plan.cards)} cards, theme: {self.teaching_plan.theme}")
        
        # Mark loading as complete with final theme
        teaching_card: TeachingCardView = self.cards["TeachingCardView"]
        teaching_card.mark_batches_complete(final_theme=self.teaching_plan.theme)
    
    def _start_quiz_generation_background(self) -> None:
        """Start generating quiz content in background thread."""
        if not self.teaching_plan:
            logger.error("Cannot generate quiz without teaching plan")
            return
        
        def generate_quiz_threaded():
            logger.task_start("generate_quiz_content")
            try:
                logger.ui("Generating quiz cards based on teaching content...")
                lesson_plan = generate_structured_lesson_plan(
                    self.assessment_result,
                    self.selected_language or "Spanish",
                    learner_context=self._learner_context,
                    teaching_plan=self.teaching_plan  # Quiz tests what was taught!
                )
                
                self.lesson_plan = lesson_plan
                logger.task_complete("generate_quiz_content")
                logger.ui(f"Quiz ready: {len(lesson_plan.cards)} cards")
                
                # Start image generation for quiz cards
                quiz_image_prompts = [
                    card.image_prompt for card in lesson_plan.cards 
                    if card.image_prompt and card.image_prompt not in self.image_cache
                ]
                if quiz_image_prompts:
                    logger.ui(f"Starting background generation of {len(quiz_image_prompts)} quiz images...")
                    generate_images_parallel(
                        list(set(quiz_image_prompts)),
                        lambda prompt, path: self._on_any_image_generated(prompt, path)
                    )
                
            except Exception as e:
                logger.task_error("generate_quiz_content", str(e))
                logger.error(f"Quiz generation failed: {e}")
        
        thread = threading.Thread(target=generate_quiz_threaded, daemon=True)
        thread.start()
        logger.task("Spawned background thread for quiz generation")
    
    
    def _on_returning_learner_content_ready(self) -> None:
        """Legacy: Called when content is ready. Now handled by _on_teaching_content_ready."""
        self._on_teaching_content_ready()


    def _on_assessment_image_generated(self, image_prompt: str, image_path: Optional[str]) -> None:
        """Called when an assessment image is generated (in main thread)."""
        if image_path:
            logger.ui(f"Assessment image ready, caching...")
            self.image_cache[image_prompt] = image_path
            # Update UI if this card is currently displayed
            assessment_card: AssessmentCardView = self.cards["AssessmentCardView"]
            assessment_card.update_image_if_needed(image_prompt, image_path)
        else:
            logger.warning("Assessment image generation returned None")
    
    def _on_lesson_image_generated(self, image_prompt: str, image_path: Optional[str]) -> None:
        """Called when a lesson image is generated (in main thread)."""
        if image_path:
            logger.ui(f"Lesson image ready, caching...")
            self.image_cache[image_prompt] = image_path
            # Update UI if this card is currently displayed
            lesson_card: LessonCardView = self.cards["LessonCardView"]
            lesson_card.update_image_if_needed(image_prompt, image_path)
        else:
            logger.warning("Lesson image generation returned None")

    def _on_assessment_generated(self, assessment_cards: List[AssessmentCard]) -> None:
        """Called when assessment generation completes (in main thread)."""
        logger.ui(f"Assessment cards received: {len(assessment_cards)} cards")
        self.assessment_cards = assessment_cards
        self.assessment_stage = 1
        
        # Start generating ALL assessment images in parallel
        self._start_parallel_assessment_image_generation()
        
        assessment_card: AssessmentCardView = self.cards["AssessmentCardView"]
        assessment_card.total_stages = len(assessment_cards)  # Set total for progress display
        assessment_card.hide_loading()
        logger.ui(f"Showing assessment question 1 of {len(assessment_cards)}")
        assessment_card.show_stage(1, assessment_cards[0])
    
    def _start_parallel_assessment_image_generation(self) -> None:
        """Start generating all assessment images in parallel background threads."""
        if not self.assessment_cards:
            return
        
        # Collect all image prompts that need generation
        # AssessmentCard wraps a LessonCard in its .card attribute
        image_prompts_to_generate = []
        for i, assessment_card in enumerate(self.assessment_cards):
            image_prompt = assessment_card.card.image_prompt
            if image_prompt and image_prompt not in self.image_cache:
                if image_prompt not in self.pending_assessment_images:
                    image_prompts_to_generate.append(image_prompt)  # Just the prompt string
                    self.pending_assessment_images.add(image_prompt)
        
        if not image_prompts_to_generate:
            logger.debug("No assessment images to generate (all cached or none needed)")
            return
        
        logger.ui(f"Starting parallel generation of {len(image_prompts_to_generate)} assessment images...")
        
        # Use the parallel image generation function (expects List[str])
        generate_images_parallel(
            image_prompts_to_generate,
            lambda prompt, path: self._on_assessment_image_parallel_done(prompt, path)
        )
    
    def _on_assessment_image_parallel_done(self, image_prompt: str, image_path: Optional[str]) -> None:
        """Callback when a parallel assessment image generation completes."""
        def update_ui():
            self.pending_assessment_images.discard(image_prompt)
            self._on_assessment_image_generated(image_prompt, image_path)
        self.after(0, update_ui)

    def _on_assessment_error(self, error: str) -> None:
        """Called when assessment generation fails (in main thread)."""
        logger.error(f"Assessment generation error: {error}")
        assessment_card: AssessmentCardView = self.cards["AssessmentCardView"]
        assessment_card.hide_loading()
        messagebox.showerror("Assessment Error", f"Failed to generate assessment: {error}\n\nReturning to intro screen.")
        self.show_card("IntroCard")

    def request_assessment_image(self, image_prompt: Optional[str]) -> None:
        """Kick off generation for a specific assessment image if needed."""
        if not image_prompt:
            logger.debug("No image prompt provided, skipping image request")
            return
        if image_prompt in self.image_cache:
            logger.debug("Image already in cache, skipping request")
            return
        if image_prompt in self.pending_assessment_images:
            logger.debug("Image already pending, skipping duplicate request")
            return

        logger.ui("Requesting assessment image generation...")
        self.pending_assessment_images.add(image_prompt)

        def _on_image_ready(path: Optional[str]) -> None:
            def _finish() -> None:
                self.pending_assessment_images.discard(image_prompt)
                self._on_assessment_image_generated(image_prompt, path)
            self.after(0, _finish)

        generate_image_async(image_prompt, _on_image_ready)

    def submit_assessment_response(self, stage: int, response: str, answer_index: Optional[int] = None) -> None:
        """Called by AssessmentCardView when user submits an assessment response."""
        logger.ui(f"Assessment response submitted for question {stage} of {len(self.assessment_cards)}")
        
        total_stages = len(self.assessment_cards)
        if stage < 1 or stage > total_stages:
            logger.warning(f"Invalid stage number: {stage}")
            return
        
        # Store response (serialize card to dict, not LessonCard object)
        card = self.assessment_cards[stage - 1].card
        response_data = {
            "stage": stage,
            "response": response,  # THE USER'S ACTUAL ANSWER
            "answer_index": answer_index,
            "card_type": card.type,
            "question": card.question or card.word or "",
            "image_prompt": card.image_prompt,
            "instruction": card.instruction,
            "correct_answer": card.correct_answer,
            "options": card.options,
            "correct_index": card.correct_index,
        }
        self.assessment_responses.append(response_data)
        
        # Log what we're storing
        logger.debug(f"Stored assessment response for question {stage}:")
        logger.debug(f"  Question: {response_data['question'][:60] if response_data['question'] else 'N/A'}...")
        logger.debug(f"  User's answer: '{response}'")
        logger.debug(f"  Correct answer: {response_data['correct_answer'][:60] if response_data['correct_answer'] else 'N/A'}...")
        logger.debug(f"  Total responses collected: {len(self.assessment_responses)}")
        
        # Move to next stage or evaluate
        if stage < total_stages:
            self.assessment_stage = stage + 1
            logger.ui(f"Moving to assessment stage {stage + 1}")
            assessment_card: AssessmentCardView = self.cards["AssessmentCardView"]
            assessment_card.show_stage(stage + 1, self.assessment_cards[stage])
        else:
            # All assessments done - evaluate responses first, then generate content
            logger.ui("All assessments complete, evaluating responses...")
            assessment_card: AssessmentCardView = self.cards["AssessmentCardView"]
            assessment_card.show_loading("Evaluating your responses...")
            
            # Store learner context for later use
            self._learner_context = None
            
            def evaluate_assessment_threaded():
                """Step 1: Evaluate assessment (fast), show results immediately."""
                logger.task_start("evaluate_assessment")
                try:
                    # Get learner context from database
                    db = get_db()
                    if db.is_connected():
                        self._learner_context = db.generate_llm_context_string(
                            self.selected_language or "Spanish"
                        )
                        logger.debug(f"Loaded learner context: {len(self._learner_context)} chars")
                    
                    # Evaluate assessment responses (one API call)
                    logger.ui("Evaluating assessment responses...")
                    assessment_result = evaluate_assessment_responses(
                        self.selected_language or "Spanish",
                        self.assessment_responses
                    )
                    
                    self.assessment_result = assessment_result
                    logger.task_complete("evaluate_assessment")
                    logger.ui(f"Assessment evaluated: {assessment_result.proficiency} level")
                    
                    # Show assessment results immediately
                    self.after(0, self._on_assessment_evaluated)
                    
                    # Start teaching content generation in background
                    self.after(0, lambda: self._start_teaching_generation_background(assessment_result))
                    
                except Exception as e:
                    logger.task_error("evaluate_assessment", str(e))
                    self.after(0, lambda: self._on_lesson_generation_error(str(e)))
            
            thread = threading.Thread(target=evaluate_assessment_threaded, daemon=True)
            thread.start()
            logger.task("Spawned background thread for assessment evaluation")
    
    def _on_assessment_evaluated(self) -> None:
        """Called when assessment is evaluated - show results immediately."""
        if not self.assessment_result:
            logger.error("Assessment result not available")
            return
        
        logger.ui(f"Showing assessment results: {self.assessment_result.proficiency}")
        
        # Capture initial stats
        self._capture_initial_stats()
        
        # Show assessment results card
        results_card: AssessmentResultsCard = self.cards["AssessmentResultsCard"]
        results_card.show_results(self.assessment_result, self.selected_language or "Spanish")
        self.show_card("AssessmentResultsCard")
    
    def _start_teaching_generation_background(self, assessment_result: AssessmentResult) -> None:
        """Start generating teaching content in background thread with batched loading."""
        # Track batches for progressive loading
        self._teaching_batches = []
        self._first_batch_shown = False
        
        def on_batch_ready(cards: List[TeachingCard], batch_num: int, total_batches: int) -> None:
            """Called when each batch of teaching cards is ready."""
            self._teaching_batches.extend(cards)
            logger.ui(f"New learner batch {batch_num}/{total_batches}: {len(cards)} cards")
        
        def generate_teaching_threaded():
            logger.task_start("generate_teaching_for_new_learner")
            try:
                logger.ui("Generating teaching content in batches...")
                teaching_plan = generate_teaching_content(
                    language=self.selected_language or "Spanish",
                    proficiency=assessment_result.proficiency,
                    learner_context=self._learner_context,
                    num_new_words=12,
                    num_review_words=5,
                    on_batch_ready=on_batch_ready,  # Progressive loading
                )
                
                self.teaching_plan = teaching_plan
                logger.task_complete("generate_teaching_for_new_learner")
                logger.ui(f"Teaching content ready: {len(teaching_plan.cards)} cards")
                
                # Start quiz generation based on teaching content
                self.after(0, lambda: self._start_quiz_generation_background())
                
            except Exception as e:
                logger.task_error("generate_teaching_for_new_learner", str(e))
                logger.error(f"Teaching generation failed: {e}")
        
        thread = threading.Thread(target=generate_teaching_threaded, daemon=True)
        thread.start()
        logger.task("Spawned background thread for teaching content generation")

    def _on_lesson_image_generated(self, image_prompt: str, image_path: Optional[str]) -> None:
        """Called when a lesson image is generated (in main thread)."""
        if image_path:
            logger.ui("Lesson image ready, updating cache...")
            self.image_cache[image_prompt] = image_path
            # Update UI if this card is currently displayed
            lesson_card: LessonCardView = self.cards["LessonCardView"]
            lesson_card.update_image_if_needed(image_prompt, image_path)
    
    def _on_any_image_generated(self, image_prompt: str, image_path: Optional[str]) -> None:
        """Called when any image (teaching or quiz) is generated."""
        if not image_path:
            return
        
        def update_ui():
            logger.debug(f"Image ready: {image_prompt[:50]}...")
            self.image_cache[image_prompt] = image_path
            
            # Update teaching cards if any use this image
            if self.teaching_plan:
                for card in self.teaching_plan.cards:
                    if card.image_prompt == image_prompt:
                        card.image_path = image_path
            
            # Update lesson cards if any use this image
            if self.lesson_plan:
                for card in self.lesson_plan.cards:
                    if card.image_prompt == image_prompt:
                        card.image_path = image_path
            
            # Notify currently visible view to update
            lesson_card: LessonCardView = self.cards["LessonCardView"]
            lesson_card.update_image_if_needed(image_prompt, image_path)
        
        self.after(0, update_ui)

    def _capture_initial_stats(self) -> None:
        """Capture initial stats for comparing before/after in summary."""
        db = get_db()
        language = self.selected_language or "Spanish"
        
        profile = db.get_language_profile(language) if db.is_connected() else None
        
        self.session_stats = SessionStats(
            words_learned_before=profile.total_vocabulary_learned if profile else 0,
            fluency_before=profile.fluency_score if profile else 0,
            proficiency_before=profile.overall_proficiency if profile else "A1",
            streak_before=profile.current_streak_days if profile else 0,
        )
        logger.debug(f"Initial stats captured: {self.session_stats.words_learned_before} words, "
                    f"fluency={self.session_stats.fluency_before}")
    
    def proceed_to_lessons(self) -> None:
        """Called from AssessmentResultsCard to proceed to teaching phase."""
        logger.ui("Proceeding to learning phase...")
        
        # Show teaching card view
        teaching_card: TeachingCardView = self.cards["TeachingCardView"]
        self.show_card("TeachingCardView")
        
        # Check if we have teaching content
        if self.teaching_plan and len(self.teaching_plan.cards) > 0:
            # Show teaching content
            logger.ui(f"Showing teaching content: {len(self.teaching_plan.cards)} cards")
            teaching_card.set_teaching_plan(self.teaching_plan)
        else:
            # Teaching content still generating - show loading
            logger.ui("Teaching content still generating, showing loading...")
            teaching_card.show_loading("Preparing your personalized lessons...")
            # Check periodically if content is ready
            self._wait_for_teaching_content()
    
    def _wait_for_teaching_content(self) -> None:
        """Poll for teaching content to be ready."""
        if self.teaching_plan and len(self.teaching_plan.cards) > 0:
            logger.ui("Teaching content now ready!")
            teaching_card: TeachingCardView = self.cards["TeachingCardView"]
            teaching_card.set_teaching_plan(self.teaching_plan)
        else:
            # Check again in 500ms
            self.after(500, self._wait_for_teaching_content)
    
    def proceed_to_quiz(self) -> None:
        """Called from TeachingCardView or directly to proceed to quiz cards."""
        logger.ui("Proceeding to quiz...")
        
        # Switch to lesson card view
        lesson_card: LessonCardView = self.cards["LessonCardView"]
        self.show_card("LessonCardView")
        
        if self.lesson_plan and len(self.lesson_plan.cards) > 0:
            # Quiz ready - show first card
            logger.ui(f"Showing quiz: {len(self.lesson_plan.cards)} cards")
            lesson_card.hide_loading()
            lesson_card.show_card_index(0)
        else:
            # Quiz still generating - show loading
            logger.ui("Quiz content still generating, showing loading...")
            lesson_card.show_loading("Preparing your quiz...")
            # Check periodically if content is ready
            self._wait_for_quiz_content()
    
    def _wait_for_quiz_content(self) -> None:
        """Poll for quiz content to be ready."""
        if self.lesson_plan and len(self.lesson_plan.cards) > 0:
            logger.ui("Quiz content now ready!")
            lesson_card: LessonCardView = self.cards["LessonCardView"]
            lesson_card.hide_loading()
            lesson_card.show_card_index(0)
        else:
            # Check again in 500ms
            self.after(500, self._wait_for_quiz_content)

    def _on_lesson_generation_error(self, error: str) -> None:
        """Called when lesson generation fails (in main thread)."""
        logger.error(f"Lesson generation error: {error}")
        assessment_card: AssessmentCardView = self.cards["AssessmentCardView"]
        assessment_card.hide_loading()
        messagebox.showerror("Lesson Generation Error", f"Failed to generate lessons: {error}\n\nReturning to intro screen.")
        self.show_card("IntroCard")

    def submit_lesson_card_response(self, response: str, answer_index: Optional[int] = None) -> None:
        """Submit response for current lesson card and get feedback."""
        logger.ui(f"Lesson card response submitted (card {self.lesson_index + 1})")
        
        if not self.lesson_plan or self.lesson_index >= len(self.lesson_plan.cards):
            logger.warning("No lesson plan or invalid card index")
            return
        
        card = self.lesson_plan.cards[self.lesson_index]
        card.user_response = response
        card.user_answer_index = answer_index
        
        logger.debug(f"Card type: {card.type}, response length: {len(response)}")
        
        lesson_card: LessonCardView = self.cards["LessonCardView"]
        
        # For text-based responses, show loading (API call needed)
        # For multiple choice, instant evaluation
        needs_api_call = card.type in ("text_question", "image_question", "fill_in_blank")
        
        if needs_api_call:
            logger.ui("Showing feedback loading (API call needed)...")
            lesson_card.show_feedback_loading()
        else:
            logger.debug("Instant evaluation (no API call needed)")
        
        def evaluate_response():
            logger.task_start(f"evaluate_response_card_{self.lesson_index + 1}")
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
                
                # Track vocabulary in database
                self._track_vocabulary_from_card(card, evaluation)
                
                logger.task_complete(f"evaluate_response_card_{self.lesson_index + 1}")
                logger.debug(f"Evaluation: correct={card.is_correct}, score={card.card_score}")
                self.after(0, lambda: lesson_card.show_feedback(evaluation))
            except Exception as e:
                logger.task_error(f"evaluate_response_card_{self.lesson_index + 1}", str(e))
                self.after(0, lambda: lesson_card.show_feedback_error(str(e)))
        
        if needs_api_call:
            # Run API call in background thread
            thread = threading.Thread(target=evaluate_response, daemon=True)
            thread.start()
            logger.task("Spawned background thread for response evaluation")
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
            
            logger.debug(f"Instant evaluation: correct={card.is_correct}, score={card.card_score}")
            lesson_card.show_feedback(evaluation)

    def next_lesson_card(self) -> None:
        """Move to the next lesson card or to summary when done."""
        if not self.lesson_plan:
            logger.warning("next_lesson_card called but no lesson plan")
            return

        self.lesson_index += 1
        logger.ui(f"Moving to lesson card {self.lesson_index + 1}")
        
        if self.lesson_index >= len(self.lesson_plan.cards):
            # Done -> generate final summary and go to summary card
            logger.ui("All lessons complete, generating final summary...")
            summary_card: SummaryCard = self.cards["SummaryCard"]
            summary_card.show_loading("Generating final summary...")
            self.show_card("SummaryCard")
            
            def generate_summary():
                logger.task_start("generate_summary")
                try:
                    total_score = sum(card.card_score for card in self.lesson_plan.cards)
                    average_score = total_score // len(self.lesson_plan.cards) if self.lesson_plan.cards else 0
                    self.lesson_plan.total_score = average_score
                    logger.debug(f"Calculated scores: total={total_score}, average={average_score}")
                    
                    summary = generate_final_summary(
                        self.lesson_plan,
                        self.assessment_result or AssessmentResult(),
                        self.selected_language or "Spanish"
                    )
                    self.final_summary = summary
                    
                    # Save session to database
                    self._save_session_to_database()
                    
                    logger.task_complete("generate_summary")
                    self.after(0, lambda: summary_card.set_summary(summary, self.lesson_plan))
                except Exception as e:
                    logger.task_error("generate_summary", str(e))
                    self.after(0, lambda: summary_card.show_error(str(e)))
            
            thread = threading.Thread(target=generate_summary, daemon=True)
            thread.start()
            logger.task("Spawned background thread for summary generation")
        else:
            lesson_card: LessonCardView = self.cards["LessonCardView"]
            lesson_card.show_card_index(self.lesson_index)

    def restart(self) -> None:
        """Restart the whole experience from the intro."""
        logger.separator("Restarting Application")
        logger.ui("User requested restart, clearing all state...")
        
        self.selected_language = None
        self.assessment_cards = []
        self.assessment_stage = 0
        self.assessment_responses = []
        self.assessment_result = None
        self.teaching_plan = None
        self.lesson_plan = None
        self.lesson_index = 0
        self.final_summary = None
        self.session_stats = None
        self.image_cache.clear()
        self.image_photos.clear()
        
        logger.debug("State cleared, showing intro card")

        intro: IntroCard = self.cards["IntroCard"]
        intro.reset()
        self.show_card("IntroCard")
        logger.success("Application restarted")


# ---------------------------------------------------------------------------
# Intro card
# ---------------------------------------------------------------------------

class IntroCard(ttk.Frame):
    def __init__(self, parent, controller: LanguageBuddyApp) -> None:
        super().__init__(parent)
        self.controller = controller

        # Use scrollable frame for content
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        
        self.scrollable = ScrollableFrame(self)
        self.scrollable.grid(row=0, column=0, sticky="nsew")
        self.content = self.scrollable.content
        
        # Configure content to center widgets
        self.content.columnconfigure(0, weight=1)

        title = ttk.Label(
            self.content,
            text="GAIT Language Buddy",
            font=("Helvetica", 28, "bold"),
            anchor="center",
            foreground="#ffffff",  # White text on dark background
        )
        title.grid(row=0, column=0, pady=(40, 20), padx=20)

        desc_text = (
            "Welcome! This AI-powered tool helps you learn languages with personalized lessons.\n\n"
            "How it works:\n"
            "  1. Choose a language — you can learn multiple simultaneously!\n"
            "  2. New learners take a comprehensive assessment (10-12 questions)\n"
            "     to determine your starting proficiency level.\n"
            "  3. 📚 Learning Phase — Study new vocabulary, grammar, and phrases\n"
            "     with images, audio pronunciation, and conjugation tables.\n"
            "  4. 📝 Quiz Phase — Test what you learned with multiple choice,\n"
            "     fill-in-the-blank, listening, and speaking exercises.\n"
            "  5. 📊 Get feedback and track your progress over time.\n\n"
            "Returning learners skip the assessment and jump straight into learning!\n"
            "Your vocabulary, progress, and streaks are saved automatically."
        )
        desc = ttk.Label(
            self.content, 
            text=desc_text, 
            wraplength=650, 
            justify="center",
            anchor="center",
            font=("Helvetica", 14),
            foreground="#d0d0d0",  # Light gray text on dark background
        )
        desc.grid(row=1, column=0, padx=40, pady=(0, 20))
        self.desc_label = desc
        
        # Language selection for new/continue
        lang_frame = ttk.Frame(self.content)
        lang_frame.grid(row=2, column=0, padx=40, pady=(0, 20))

        ttk.Label(lang_frame, text="Select language:", font=("Helvetica", 14)).grid(
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
        self.language_combo.bind("<<ComboboxSelected>>", self._on_language_selected)
        
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
        
        # Selected language stats panel (shown when a language with progress is selected)
        self.selected_stats_frame = ttk.Frame(self.content)
        self.selected_stats_frame.grid(row=3, column=0, pady=(15, 10))
        self.selected_stats_frame.grid_remove()  # Hidden by default
        
        # Stats labels (will be populated when language is selected)
        self.selected_lang_title = ttk.Label(
            self.selected_stats_frame,
            text="",
            font=("Helvetica", 16, "bold"),
            foreground="#ffffff",
            anchor="center",
        )
        self.selected_lang_title.grid(row=0, column=0, columnspan=2, pady=(0, 8))
        
        self.selected_level_label = ttk.Label(
            self.selected_stats_frame,
            text="",
            font=("Helvetica", 13),
            foreground="#7bb3ff",
        )
        self.selected_level_label.grid(row=1, column=0, columnspan=2, pady=(0, 4))
        
        self.selected_stats_label = ttk.Label(
            self.selected_stats_frame,
            text="",
            font=("Helvetica", 12),
            foreground="#9bc6ff",
        )
        self.selected_stats_label.grid(row=2, column=0, columnspan=2, pady=(0, 4))
        
        self.selected_vocab_label = ttk.Label(
            self.selected_stats_frame,
            text="",
            font=("Helvetica", 12),
            foreground="#9bc6ff",
        )
        self.selected_vocab_label.grid(row=3, column=0, columnspan=2, pady=(0, 4))
        
        self.selected_streak_label = ttk.Label(
            self.selected_stats_frame,
            text="",
            font=("Helvetica", 12),
            foreground="#ffb347",
        )
        self.selected_streak_label.grid(row=4, column=0, columnspan=2, pady=(0, 8))
        
        # Reset button (only shown when there's data to reset)
        self.reset_language_button = ttk.Button(
            self.selected_stats_frame,
            text="🗑 Reset Progress",
            command=self._on_reset_language_clicked,
            width=16,
        )
        self.reset_language_button.grid(row=5, column=0, columnspan=2, pady=(5, 0))
        
        # Style reset button with warning color
        style.configure("Reset.TButton", font=("Helvetica", 11))
        self.reset_language_button.configure(style="Reset.TButton")

        # API key warning (shown only if API isn't available)
        self.api_warning_label = ttk.Label(
            self.content,
            text="⚠ OpenAI API key not found!\n\nPlease add your API key to a .env file:\nOPENAI_API_KEY=sk-...\n\nThe application cannot function without a valid API key.",
            font=("Helvetica", 13),
            foreground="#ff6b6b",  # Red warning color
            justify="center",
            anchor="center",
            wraplength=500,
        )
        self.api_warning_label.grid(row=4, column=0, pady=(10, 10))
        
        # Check API availability and show/hide warning
        self.api_available = is_api_available()
        if self.api_available:
            self.api_warning_label.grid_remove()
        
        # Database status label
        self.db_status_label = ttk.Label(
            self.content,
            text="",
            font=("Helvetica", 11),
            justify="center",
            anchor="center",
            wraplength=500,
        )
        self.db_status_label.grid(row=5, column=0, pady=(5, 5))
        self._update_database_status()
        
        self.start_button = ttk.Button(
            self.content,
            text="Start Session" if self.api_available else "API Key Required",
            command=self._on_start_clicked,
            state="normal" if self.api_available else "disabled",
        )
        self.start_button.grid(row=6, column=0, pady=(20, 50))
        
        # Style the button to be more prominent
        style = ttk.Style()
        style.configure("Start.TButton", font=("Helvetica", 14, "bold"), padding=10)
        self.start_button.configure(style="Start.TButton")

        # Responsive wrapping - bind to scrollable canvas for width changes
        self.scrollable.canvas.bind("<Configure>", self._on_content_resize)
    
    def _update_database_status(self) -> None:
        """Update the database status label based on connection state."""
        if self.controller.database_connected:
            self.db_status_label.configure(
                text="✓ Database connected - Progress will be saved",
                foreground="#69db7c"  # Green
            )
        else:
            error_msg = self.controller.database_error or "Unknown error"
            self.db_status_label.configure(
                text=f"⚠ Database not connected - {error_msg}\nProgress will NOT be saved between sessions.",
                foreground="#ffa94d"  # Orange warning
            )
    
    def _get_language_emoji(self, language: str) -> str:
        """Get flag emoji for language."""
        emojis = {
            "Spanish": "🇪🇸",
            "French": "🇫🇷",
            "German": "🇩🇪",
            "Japanese": "🇯🇵",
            "Chinese": "🇨🇳",
            "Arabic": "🇸🇦",
        }
        return emojis.get(language, "🌍")
    
    def _get_level_color(self, level: str) -> str:
        """Get color for proficiency level."""
        colors = {
            "A1": "#ff6b6b",  # Red - Beginner
            "A2": "#ffa94d",  # Orange
            "B1": "#ffd43b",  # Yellow
            "B2": "#69db7c",  # Green
            "C1": "#4dabf7",  # Blue
            "C2": "#da77f2",  # Purple - Mastery
        }
        return colors.get(level, "#ffffff")
    
    def _on_language_selected(self, event=None) -> None:
        """Update stats panel and button text based on selected language."""
        language = self.language_var.get()
        if not language:
            self.selected_stats_frame.grid_remove()
            return
        
        db = get_db()
        profile = db.get_language_profile(language) if db.is_connected() else None
        
        if profile and profile.total_sessions > 0:
            # Show stats panel with existing progress
            emoji = self._get_language_emoji(language)
            self.selected_lang_title.configure(text=f"{emoji} {language} Progress")
            
            level_color = self._get_level_color(profile.overall_proficiency)
            self.selected_level_label.configure(
                text=f"📊 Level: {profile.overall_proficiency} (Fluency: {profile.fluency_score}/100)",
                foreground=level_color
            )
            
            self.selected_stats_label.configure(
                text=f"📝 {profile.total_sessions} sessions  •  ⏱ {profile.total_time_minutes} min total"
            )
            
            self.selected_vocab_label.configure(
                text=f"📖 {profile.total_vocabulary_learned} words learned  •  🎯 {profile.total_cards_completed} cards completed"
            )
            
            if profile.current_streak_days > 0:
                self.selected_streak_label.configure(
                    text=f"🔥 {profile.current_streak_days} day streak (best: {profile.longest_streak_days})"
                )
                self.selected_streak_label.grid()
            else:
                self.selected_streak_label.grid_remove()
            
            self.selected_stats_frame.grid()
            self.start_button.configure(text=f"Continue {language}")
        else:
            # Hide stats panel for new language
            self.selected_stats_frame.grid_remove()
            self.start_button.configure(text=f"Start Learning {language}")
    
    def _on_reset_language_clicked(self) -> None:
        """Reset all progress for the selected language."""
        language = self.language_var.get()
        if not language:
            return
        
        # Get current stats for the confirmation message
        db = get_db()
        profile = db.get_language_profile(language) if db.is_connected() else None
        
        stats_text = ""
        if profile and profile.total_sessions > 0:
            stats_text = (
                f"\n\nCurrent progress:\n"
                f"• Level: {profile.overall_proficiency}\n"
                f"• {profile.total_sessions} sessions\n"
                f"• {profile.total_vocabulary_learned} words learned\n"
                f"• {profile.total_time_minutes} minutes practiced"
            )
        
        # Confirm deletion
        result = messagebox.askyesno(
            "Reset Progress",
            f"⚠️ Are you sure you want to reset ALL progress for {language}?\n\n"
            f"This will permanently delete:\n"
            f"• Your proficiency level\n"
            f"• All vocabulary data & strength ratings\n"
            f"• Session history\n"
            f"• Grammar patterns\n"
            f"• Streak data"
            f"{stats_text}\n\n"
            f"This action cannot be undone!",
            icon="warning"
        )
        
        if not result:
            return
        
        try:
            # Use the proper deletion method
            success = db.delete_language_progress(language)
            
            if success:
                logger.ui(f"Reset progress for {language}")
                messagebox.showinfo("Progress Reset", f"All progress for {language} has been reset.\n\nYou can start fresh!")
            else:
                messagebox.showerror("Error", "Failed to reset progress. Please try again.")
                return
            
            # Refresh the UI
            self.selected_stats_frame.grid_remove()
            self.start_button.configure(text=f"Start Learning {language}")
            
        except Exception as e:
            logger.error(f"Failed to reset language progress: {e}")
            messagebox.showerror("Error", f"Failed to reset progress: {e}")

    def _on_start_clicked(self) -> None:
        # Double-check API availability
        if not is_api_available():
            messagebox.showerror(
                "API Key Required", 
                "OpenAI API key not found.\n\nPlease add your API key to a .env file:\nOPENAI_API_KEY=sk-..."
            )
            return
        
        language = self.language_var.get()
        if not language:
            messagebox.showinfo("Choose a language", "Please select a target language to continue.")
            return

        # Start the session in the controller
        self.controller.start_session(language)

    def reset(self) -> None:
        """Reset the intro card state."""
        self.language_var.set("")
        self._on_language_selected()  # Reset button text and hide stats
        self._update_database_status()  # Refresh database status

    def _on_content_resize(self, event: tk.Event) -> None:
        """Adjust intro text wrapping when the window size changes."""
        available = max(200, event.width - 80)
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

        # Use scrollable frame for content
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        
        self.scrollable = ScrollableFrame(self)
        self.scrollable.grid(row=0, column=0, sticky="nsew")
        self.content = self.scrollable.content
        
        # Configure content to center widgets
        self.content.columnconfigure(0, weight=1)

        self.title_label = ttk.Label(
            self.content,
            text="Language Assessment",
            font=("Helvetica", 28, "bold"),
            anchor="center",
            foreground="#ffffff",  # White text on dark background
        )
        self.title_label.grid(row=0, column=0, pady=(30, 15))

        self.stage_label = ttk.Label(
            self.content,
            text="Question 1 of 10",
            font=("Helvetica", 14),
            anchor="center",
            foreground="#7bb3ff",  # Light blue text on dark background
        )
        self.stage_label.grid(row=1, column=0, pady=(0, 10))
        
        self.total_stages = 10  # Will be updated when assessment cards are loaded

        # Loading spinner (starts hidden)
        self.loading_spinner = LoadingSpinner(self.content, text="Generating assessment...")
        self.loading_spinner.grid(row=2, column=0, pady=(20, 20))

        self.pending_instruction_text: str = ""
        self.waiting_for_image: bool = False

        # Instruction label (centered)
        self.instruction_label = ttk.Label(
            self.content,
            text="",
            wraplength=600,
            justify="center",
            anchor="center",
            font=("Helvetica", 14),
            foreground="#7bb3ff",  # Light blue text on dark background
        )
        self.instruction_label.grid(row=3, column=0, padx=30, pady=(0, 10))

        # Image loading status labels
        self.image_status_label = ttk.Label(
            self.content,
            text="",
            wraplength=600,
            justify="center",
            anchor="center",
            font=("Helvetica", 13),
            foreground="#7bb3ff",
        )
        self.image_status_label.grid(row=4, column=0, padx=30)
        self.image_status_label.grid_remove()

        self.image_status_detail_label = ttk.Label(
            self.content,
            text="",
            wraplength=600,
            justify="center",
            anchor="center",
            font=("Helvetica", 12, "italic"),
            foreground="#9bc6ff",
        )
        self.image_status_detail_label.grid(row=5, column=0, padx=30, pady=(0, 10))
        self.image_status_detail_label.grid_remove()

        # Image area (for image-based questions) - responsive sizing
        self.responsive_image = ResponsiveImage(self.content)
        self.responsive_image.grid(row=6, column=0, padx=20, pady=(0, 10))
        self.responsive_image.grid_remove()

        # Question label
        self.question_label = ttk.Label(
            self.content,
            text="",
            font=("Helvetica", 24, "bold"),
            wraplength=600,
            justify="center",
            anchor="center",
            foreground="#ffffff",  # White text on dark background
        )
        self.question_label.grid(row=7, column=0, padx=30, pady=(15, 8))

        # Answer area (dynamic based on card type)
        self.answer_frame = ttk.Frame(self.content)
        self.answer_frame.grid(row=8, column=0, padx=30, pady=(0, 15))
        
        # Text input (for text questions)
        self.text_widget: Optional[tk.Text] = None
        self.text_scrollbar: Optional[ttk.Scrollbar] = None

        submit_button = ttk.Button(
            self.content,
            text="Submit Answer",
            command=self._on_submit_clicked,
        )
        self.submit_button = submit_button
        submit_button.grid(row=9, column=0, pady=(10, 40))

        # Responsive wrapping labels
        self._wrap_targets = [
            self.instruction_label,
            self.image_status_label,
            self.image_status_detail_label,
            self.question_label,
        ]
        self.scrollable.canvas.bind("<Configure>", self._update_wraplengths)

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
        width = self.scrollable.canvas.winfo_width() or self.winfo_width()
        wrap = max(200, width - 60)
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
        self.responsive_image.grid_remove()
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

        self.stage_label.configure(text=f"Question {stage} of {self.total_stages}")

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
                self.responsive_image.grid_remove()
                # Hide Q&A until image arrives so the user doesn't answer blind
                self.question_label.grid_remove()
                self.answer_frame.grid_remove()
                self.submit_button.grid_remove()
                self.controller.request_assessment_image(card.image_prompt)
        else:
            self.waiting_for_image = False
            self.responsive_image.grid_remove()

        # Scroll to top when showing new stage
        self.scrollable.scroll_to_top()


    def _render_text_input(self) -> None:
        """Render text input area."""
        text_frame = ttk.Frame(self.answer_frame)
        text_frame.grid(row=0, column=0)
        
        self.text_widget = tk.Text(
            text_frame, 
            height=8,
            width=60,  # Fixed width for consistent centering
            wrap="word",
            bg="#2d2d2d",  # Dark background
            fg="#e0e0e0",  # Light text
            insertbackground="#e0e0e0",  # Light cursor
            selectbackground="#3d3d3d",  # Dark selection background
            selectforeground="#ffffff",  # White selected text
        )
        self.text_widget.grid(row=0, column=0, pady=(5, 0))
        
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
        success = self.responsive_image.set_image(image_path)
        self.responsive_image.grid()

        # Once the image is ready, restore the normal instruction + Q&A
        self._set_instruction_text(self.pending_instruction_text or "Please answer the following question.")
        self._clear_image_status()
        self.question_label.grid()
        self.answer_frame.grid()
        self.submit_button.grid()


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
# Teaching Card View (Educational content before quiz)
# ---------------------------------------------------------------------------

class TeachingCardView(ttk.Frame):
    """
    Displays teaching content (vocabulary, grammar, phrases) with prev/next navigation.
    This is shown BEFORE the quiz to teach new concepts.
    """
    
    def __init__(self, parent, controller: LanguageBuddyApp) -> None:
        super().__init__(parent)
        self.controller = controller
        self.teaching_plan: Optional[TeachingPlan] = None
        self.current_index: int = 0
        
        # Asset loading tracking
        self.total_images: int = 0
        self.loaded_images: int = 0
        self.total_audio: int = 0
        self.loaded_audio: int = 0
        self.more_batches_loading: bool = False
        
        # Use scrollable frame for content
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        
        self.scrollable = ScrollableFrame(self)
        self.scrollable.grid(row=0, column=0, sticky="nsew")
        self.content = self.scrollable.content
        
        # Configure content to center widgets
        self.content.columnconfigure(0, weight=1)
        
        # Loading spinner
        self.loading_spinner = LoadingSpinner(self.content, text="Generating lesson content...")
        self.loading_spinner.grid(row=0, column=0, pady=(100, 40))
        
        # Title area
        self.title_label = ttk.Label(
            self.content,
            text="📚 Learning Time",
            font=("Helvetica", 28, "bold"),
            foreground="#ffffff",
            anchor="center",
        )
        self.title_label.grid(row=1, column=0, pady=(20, 5))
        
        # Theme/subtitle
        self.theme_label = ttk.Label(
            self.content,
            text="",
            font=("Helvetica", 14),
            foreground="#7bb3ff",
            anchor="center",
        )
        self.theme_label.grid(row=2, column=0, pady=(0, 5))
        
        # Asset loading progress (images/audio)
        self.asset_progress_label = ttk.Label(
            self.content,
            text="",
            font=("Helvetica", 11),
            foreground="#69db7c",
            anchor="center",
        )
        self.asset_progress_label.grid(row=3, column=0, pady=(0, 5))
        self.asset_progress_label.grid_remove()
        
        # Progress indicator
        self.progress_label = ttk.Label(
            self.content,
            text="Card 1 of 10",
            font=("Helvetica", 12),
            foreground="#9bc6ff",
            anchor="center",
        )
        self.progress_label.grid(row=4, column=0, pady=(0, 15))
        
        # Card content frame
        self.card_frame = ttk.Frame(self.content)
        self.card_frame.grid(row=5, column=0, padx=30, pady=(10, 10), sticky="ew")
        self.card_frame.columnconfigure(0, weight=1)
        
        # Card title (New Word: der Apfel, Grammar: Articles, etc.)
        self.card_title_label = ttk.Label(
            self.card_frame,
            text="",
            font=("Helvetica", 22, "bold"),
            foreground="#ffd43b",  # Yellow highlight for card title
            anchor="center",
            wraplength=600,
        )
        self.card_title_label.grid(row=0, column=0, pady=(10, 10))
        
        # Image area (for vocabulary illustrations)
        self.responsive_image = ResponsiveImage(self.card_frame)
        self.responsive_image.grid(row=1, column=0, padx=20, pady=(5, 10))
        self.responsive_image.grid_remove()
        
        # Image loading placeholder
        self.image_loading_label = ttk.Label(
            self.card_frame,
            text="🖼️ Loading image...",
            font=("Helvetica", 14),
            foreground="#7bb3ff",
            anchor="center",
        )
        self.image_loading_label.grid(row=1, column=0, pady=(20, 20))
        self.image_loading_label.grid_remove()
        
        # Main word/phrase display
        self.word_label = ttk.Label(
            self.card_frame,
            text="",
            font=("Helvetica", 36, "bold"),
            foreground="#ffffff",
            anchor="center",
        )
        self.word_label.grid(row=2, column=0, pady=(10, 5))
        
        # Translation
        self.translation_label = ttk.Label(
            self.card_frame,
            text="",
            font=("Helvetica", 20),
            foreground="#69db7c",  # Green for translation
            anchor="center",
        )
        self.translation_label.grid(row=3, column=0, pady=(0, 10))
        
        # Part of speech / Gender info
        self.pos_label = ttk.Label(
            self.card_frame,
            text="",
            font=("Helvetica", 12, "italic"),
            foreground="#9bc6ff",
            anchor="center",
        )
        self.pos_label.grid(row=4, column=0, pady=(0, 10))
        
        # Audio player for pronunciation
        self.audio_player: Optional[AudioPlayer] = None
        self.audio_frame = ttk.Frame(self.card_frame)
        self.audio_frame.grid(row=5, column=0, pady=(5, 10))
        
        # Explanation text
        self.explanation_label = ttk.Label(
            self.card_frame,
            text="",
            font=("Helvetica", 14),
            foreground="#d0d0d0",
            wraplength=550,
            justify="center",
            anchor="center",
        )
        self.explanation_label.grid(row=6, column=0, pady=(10, 10), padx=20)
        
        # Example sentence frame
        self.example_frame = ttk.Frame(self.card_frame)
        self.example_frame.grid(row=7, column=0, pady=(10, 10))
        
        self.example_label = ttk.Label(
            self.example_frame,
            text="",
            font=("Helvetica", 16, "italic"),
            foreground="#ffffff",
            wraplength=550,
            justify="center",
            anchor="center",
        )
        self.example_label.grid(row=0, column=0, pady=(0, 5))
        
        self.example_translation_label = ttk.Label(
            self.example_frame,
            text="",
            font=("Helvetica", 13),
            foreground="#9bc6ff",
            wraplength=550,
            justify="center",
            anchor="center",
        )
        self.example_translation_label.grid(row=1, column=0)
        
        # Grammar examples frame (for grammar lessons)
        self.grammar_frame = ttk.Frame(self.card_frame)
        self.grammar_frame.grid(row=8, column=0, pady=(10, 10))
        self.grammar_frame.columnconfigure(0, weight=1)
        
        # Related words
        self.related_frame = ttk.Frame(self.card_frame)
        self.related_frame.grid(row=9, column=0, pady=(10, 10))
        
        self.related_label = ttk.Label(
            self.related_frame,
            text="",
            font=("Helvetica", 12),
            foreground="#7bb3ff",
            wraplength=500,
            justify="center",
            anchor="center",
        )
        self.related_label.grid(row=0, column=0)
        
        # Memory tip
        self.mnemonic_label = ttk.Label(
            self.card_frame,
            text="",
            font=("Helvetica", 12, "italic"),
            foreground="#ffa94d",  # Orange for tips
            wraplength=500,
            justify="center",
            anchor="center",
        )
        self.mnemonic_label.grid(row=10, column=0, pady=(10, 10))
        
        # Navigation buttons frame - MUST be after card_frame (row 5)
        self.nav_frame = ttk.Frame(self.content)
        self.nav_frame.grid(row=6, column=0, pady=(20, 10))
        
        self.prev_button = ttk.Button(
            self.nav_frame,
            text="← Previous",
            command=self._on_prev,
            width=15,
        )
        self.prev_button.grid(row=0, column=0, padx=10)
        
        self.next_button = ttk.Button(
            self.nav_frame,
            text="Next →",
            command=self._on_next,
            width=15,
        )
        self.next_button.grid(row=0, column=1, padx=10)
        
        # Skip to quiz button
        self.skip_button = ttk.Button(
            self.content,
            text="Skip to Quiz →",
            command=self._on_skip_to_quiz,
            width=20,
        )
        self.skip_button.grid(row=7, column=0, pady=(10, 30))
        
        # Initially hide content
        self._hide_content()
    
    def _hide_content(self) -> None:
        """Hide all content elements."""
        self.title_label.grid_remove()
        self.theme_label.grid_remove()
        self.progress_label.grid_remove()
        self.card_frame.grid_remove()
        self.nav_frame.grid_remove()
        self.skip_button.grid_remove()
    
    def _show_content(self) -> None:
        """Show all content elements."""
        self.title_label.grid()
        self.theme_label.grid()
        self.progress_label.grid()
        self.card_frame.grid()
        self.nav_frame.grid()
        self.skip_button.grid()
    
    def show_loading(self, message: str = "Generating lesson content...") -> None:
        """Show loading state."""
        self._hide_content()
        self.loading_spinner.text = message
        self.loading_spinner.label.configure(text=f"{self.loading_spinner.spinner_chars[0]} {message}")
        self.loading_spinner.grid()
        self.loading_spinner.start()
    
    def hide_loading(self) -> None:
        """Hide loading state."""
        self.loading_spinner.stop()
        self.loading_spinner.grid_remove()
    
    def set_teaching_plan(self, teaching_plan: TeachingPlan, more_loading: bool = False) -> None:
        """Set the teaching plan and display the first card.
        
        Args:
            teaching_plan: The teaching plan to display
            more_loading: If True, more batches are still loading
        """
        self.teaching_plan = teaching_plan
        self.current_index = 0
        self.more_batches_loading = more_loading
        
        self.hide_loading()
        self._show_content()
        
        # Update theme
        theme = teaching_plan.theme or "Language Learning"
        self.theme_label.configure(text=f"Theme: {theme}")
        
        # Start generating images for all cards
        self._start_image_generation()
        
        # Start generating audio for all cards
        self._start_audio_generation()
        
        # Show batch loading status if more batches coming
        if more_loading:
            self.asset_progress_label.configure(text="⏳ Loading more lessons...")
            self.asset_progress_label.grid()
        
        # Show first card
        self._render_current_card()
    
    def update_loading_progress(self, message: str, batch_num: int, total_batches: int, total_cards: int) -> None:
        """Update the loading spinner with batch progress."""
        pct = int((batch_num / total_batches) * 100)
        self.loading_spinner.text = f"{message}\n{total_cards} cards ready ({pct}%)"
        self.loading_spinner.label.configure(text=f"{self.loading_spinner.spinner_chars[self.loading_spinner.spinner_index]} {self.loading_spinner.text}")
    
    def update_card_count(self, total_cards: int) -> None:
        """Update the displayed card count when new batches arrive."""
        if self.teaching_plan:
            # Update progress label to show new total
            card = self.teaching_plan.cards[self.current_index] if self.current_index < len(self.teaching_plan.cards) else None
            if card:
                card_type_display = "New" if card.is_new else "Review"
                self.progress_label.configure(
                    text=f"Card {self.current_index + 1} of {total_cards} • {card_type_display}"
                )
    
    def add_cards_to_generation(self, new_cards: List[TeachingCard]) -> None:
        """Start generating images/audio for newly added cards."""
        if not new_cards:
            return
        
        # Add new images to generation queue
        new_image_prompts = [
            card.image_prompt for card in new_cards
            if card.image_prompt and not card.image_path
        ]
        if new_image_prompts:
            self.total_images += len(new_image_prompts)
            self._update_asset_progress()
            generate_images_parallel(
                list(set(new_image_prompts)),
                lambda prompt, path: self._on_image_generated(prompt, path)
            )
        
        # Add new audio to generation queue
        language = self.controller.selected_language or "Spanish"
        for card in new_cards:
            if card.audio_text and not card.audio_path:
                self.total_audio += 1
                self._update_asset_progress()
                
                card_index = self.teaching_plan.cards.index(card) if self.teaching_plan else -1
                def on_audio_done(path: Optional[str], idx: int = card_index) -> None:
                    def update():
                        self.loaded_audio += 1
                        self._update_asset_progress()
                        if path and self.teaching_plan and idx >= 0:
                            self.teaching_plan.cards[idx].audio_path = path
                            if self.current_index == idx:
                                self._update_audio_player()
                    self.after(0, update)
                
                generate_speech_async(card.audio_text, language, on_audio_done)
    
    def mark_batches_complete(self, final_theme: Optional[str] = None) -> None:
        """Mark that all teaching batches have been loaded."""
        self.more_batches_loading = False
        logger.ui("All teaching batches complete")
        
        # Update theme if provided
        if final_theme and self.teaching_plan:
            self.teaching_plan.theme = final_theme
            self.theme_label.configure(text=f"Theme: {final_theme}")
        
        # Update asset progress to show final state
        self._update_asset_progress()
    
    def _start_image_generation(self) -> None:
        """Start generating images for all cards that need them."""
        if not self.teaching_plan:
            return
        
        image_prompts = []
        for i, card in enumerate(self.teaching_plan.cards):
            if card.image_prompt and not card.image_path:
                image_prompts.append(card.image_prompt)
        
        # Track totals for progress display
        self.total_images = len(image_prompts)
        self.loaded_images = 0
        self._update_asset_progress()
        
        if image_prompts:
            logger.ui(f"Starting parallel generation of {len(image_prompts)} teaching images...")
            generate_images_parallel(
                image_prompts,
                lambda prompt, path: self._on_image_generated(prompt, path)
            )
    
    def _on_image_generated(self, image_prompt: str, image_path: Optional[str]) -> None:
        """Called when an image is generated."""
        if not self.teaching_plan:
            return
        
        def update_ui():
            # Update progress counter
            self.loaded_images += 1
            self._update_asset_progress()
            
            if not image_path:
                return
                
            # Update the card that has this image prompt
            for card in self.teaching_plan.cards:
                if card.image_prompt == image_prompt:
                    card.image_path = image_path
            
            # Re-render current card if it has this image
            if self.teaching_plan.cards[self.current_index].image_prompt == image_prompt:
                self._render_current_card()
        
        self.after(0, update_ui)
    
    def _start_audio_generation(self) -> None:
        """Start generating audio for all cards that need TTS."""
        if not self.teaching_plan:
            return
        
        language = self.controller.selected_language or "Spanish"
        
        # Count how many audio files need generation
        audio_cards = [card for card in self.teaching_plan.cards if card.audio_text and not card.audio_path]
        self.total_audio = len(audio_cards)
        self.loaded_audio = 0
        self._update_asset_progress()
        
        for i, card in enumerate(self.teaching_plan.cards):
            if card.audio_text and not card.audio_path:
                def on_audio_done(path: Optional[str], card_index: int = i) -> None:
                    def update():
                        # Update progress counter
                        self.loaded_audio += 1
                        self._update_asset_progress()
                        
                        if path and self.teaching_plan:
                            self.teaching_plan.cards[card_index].audio_path = path
                            # Re-render if current card
                            if self.current_index == card_index:
                                self._update_audio_player()
                    self.after(0, update)
                
                generate_speech_async(card.audio_text, language, on_audio_done)
    
    def _update_asset_progress(self) -> None:
        """Update the asset loading progress display."""
        parts = []
        
        if self.total_images > 0:
            if self.loaded_images < self.total_images:
                parts.append(f"🖼️ {self.loaded_images}/{self.total_images} images")
            else:
                parts.append(f"✓ {self.total_images} images")
        
        if self.total_audio > 0:
            if self.loaded_audio < self.total_audio:
                parts.append(f"🔊 {self.loaded_audio}/{self.total_audio} audio")
            else:
                parts.append(f"✓ {self.total_audio} audio")
        
        if parts:
            progress_text = " • ".join(parts)
            self.asset_progress_label.configure(text=progress_text)
            self.asset_progress_label.grid()
            
            # Change color when all done
            if self.loaded_images >= self.total_images and self.loaded_audio >= self.total_audio:
                self.asset_progress_label.configure(foreground="#69db7c")  # Green
            else:
                self.asset_progress_label.configure(foreground="#7bb3ff")  # Blue
        else:
            self.asset_progress_label.grid_remove()
    
    def _render_current_card(self) -> None:
        """Render the current teaching card."""
        if not self.teaching_plan or self.current_index >= len(self.teaching_plan.cards):
            return
        
        card = self.teaching_plan.cards[self.current_index]
        total = len(self.teaching_plan.cards)
        
        # Update progress
        card_type_display = "New" if card.is_new else "Review"
        self.progress_label.configure(
            text=f"Card {self.current_index + 1} of {total} • {card_type_display}"
        )
        
        # Update navigation buttons
        self.prev_button.configure(state="normal" if self.current_index > 0 else "disabled")
        
        if self.current_index >= total - 1:
            self.next_button.configure(text="Start Quiz →")
            # Hide skip button on last card (redundant with Start Quiz)
            self.skip_button.grid_remove()
        else:
            self.next_button.configure(text="Next →")
            # Show skip button on other cards
            self.skip_button.grid()
        
        # Clear grammar frame
        for widget in self.grammar_frame.winfo_children():
            widget.destroy()
        
        # Render based on card type
        if card.type == "vocabulary_intro" or card.type == "phrase_lesson" or card.type == "concept_review":
            self._render_vocabulary_card(card)
        elif card.type == "grammar_lesson":
            self._render_grammar_card(card)
        elif card.type == "conjugation_table":
            self._render_conjugation_card(card)
        else:
            self._render_vocabulary_card(card)  # Default
        
        # Scroll to top
        self.scrollable.scroll_to_top()
    
    def _render_vocabulary_card(self, card: TeachingCard) -> None:
        """Render a vocabulary/phrase card."""
        # Title
        self.card_title_label.configure(text=card.title)
        
        # Image - show placeholder if loading
        if card.image_path:
            self.responsive_image.set_image(card.image_path)
            self.responsive_image.grid()
            self.image_loading_label.grid_remove()
        elif card.image_prompt:
            # Image is being generated - show loading placeholder
            self.responsive_image.grid_remove()
            self.image_loading_label.configure(text="🖼️ Loading image...")
            self.image_loading_label.grid()
        else:
            self.responsive_image.grid_remove()
            self.image_loading_label.grid_remove()
        
        # Main word
        self.word_label.configure(text=card.word or "")
        self.word_label.grid() if card.word else self.word_label.grid_remove()
        
        # Translation
        self.translation_label.configure(text=card.translation or "")
        self.translation_label.grid() if card.translation else self.translation_label.grid_remove()
        
        # Part of speech / Gender
        pos_text = ""
        if card.part_of_speech:
            pos_text = card.part_of_speech
        if card.gender:
            pos_text += f" • {card.gender}"
        if card.pronunciation_hint:
            pos_text += f" • 🔊 {card.pronunciation_hint}"
        self.pos_label.configure(text=pos_text)
        self.pos_label.grid() if pos_text else self.pos_label.grid_remove()
        
        # Audio player
        self._update_audio_player()
        
        # Explanation
        self.explanation_label.configure(text=card.explanation or "")
        self.explanation_label.grid() if card.explanation else self.explanation_label.grid_remove()
        
        # Example sentence
        self.example_label.configure(text=f'"{card.example_sentence}"' if card.example_sentence else "")
        self.example_label.grid() if card.example_sentence else self.example_label.grid_remove()
        
        self.example_translation_label.configure(text=card.example_translation or "")
        self.example_translation_label.grid() if card.example_translation else self.example_translation_label.grid_remove()
        
        self.example_frame.grid() if card.example_sentence else self.example_frame.grid_remove()
        
        # Related words
        if card.related_words:
            related_text = "Related: " + " • ".join(card.related_words[:5])
            self.related_label.configure(text=related_text)
            self.related_frame.grid()
        else:
            self.related_frame.grid_remove()
        
        # Memory tip
        if card.mnemonic:
            self.mnemonic_label.configure(text=f"💡 Tip: {card.mnemonic}")
            self.mnemonic_label.grid()
        else:
            self.mnemonic_label.grid_remove()
        
        # Hide grammar frame
        self.grammar_frame.grid_remove()
    
    def _render_grammar_card(self, card: TeachingCard) -> None:
        """Render a grammar lesson card."""
        # Title
        self.card_title_label.configure(text=card.title)
        
        # Hide image for grammar cards
        self.responsive_image.grid_remove()
        
        # Grammar rule as main text
        self.word_label.configure(text=card.grammar_rule or "")
        self.word_label.grid() if card.grammar_rule else self.word_label.grid_remove()
        
        # Hide translation
        self.translation_label.grid_remove()
        self.pos_label.grid_remove()
        
        # Audio player
        self._update_audio_player()
        
        # Explanation
        self.explanation_label.configure(text=card.explanation or "")
        self.explanation_label.grid() if card.explanation else self.explanation_label.grid_remove()
        
        # Hide example frame
        self.example_frame.grid_remove()
        
        # Grammar examples
        if card.grammar_examples:
            self.grammar_frame.grid()
            
            examples_title = ttk.Label(
                self.grammar_frame,
                text="Examples:",
                font=("Helvetica", 14, "bold"),
                foreground="#7bb3ff",
            )
            examples_title.grid(row=0, column=0, pady=(5, 10), sticky="w")
            
            for i, example in enumerate(card.grammar_examples[:5]):
                target = example.get("target", "")
                english = example.get("english", "")
                
                example_frame = ttk.Frame(self.grammar_frame)
                example_frame.grid(row=i+1, column=0, pady=5, sticky="w", padx=20)
                
                target_label = ttk.Label(
                    example_frame,
                    text=target,
                    font=("Helvetica", 14),
                    foreground="#ffffff",
                )
                target_label.grid(row=0, column=0, sticky="w")
                
                english_label = ttk.Label(
                    example_frame,
                    text=f"→ {english}",
                    font=("Helvetica", 12),
                    foreground="#9bc6ff",
                )
                english_label.grid(row=1, column=0, sticky="w", padx=(10, 0))
        else:
            self.grammar_frame.grid_remove()
        
        # Common mistakes as mnemonic
        if card.common_mistakes:
            mistakes_text = "⚠️ Watch out: " + " | ".join(card.common_mistakes[:3])
            self.mnemonic_label.configure(text=mistakes_text)
            self.mnemonic_label.grid()
        else:
            self.mnemonic_label.grid_remove()
        
        # Related words
        self.related_frame.grid_remove()
    
    def _render_conjugation_card(self, card: TeachingCard) -> None:
        """Render a verb conjugation table card."""
        # Title
        self.card_title_label.configure(text=card.title or f"Conjugation: {card.infinitive}")
        
        # Hide standard vocab elements
        self.responsive_image.grid_remove()
        self.image_loading_label.grid_remove()
        
        # Show infinitive and translation
        infinitive_text = card.infinitive or card.word or ""
        self.word_label.configure(text=infinitive_text)
        self.word_label.grid() if infinitive_text else self.word_label.grid_remove()
        
        translation_text = card.translation or ""
        if card.verb_type:
            translation_text += f" ({card.verb_type})"
        self.translation_label.configure(text=translation_text)
        self.translation_label.grid() if translation_text else self.translation_label.grid_remove()
        
        # Show tense
        if card.tense:
            self.pos_label.configure(text=f"📖 {card.tense} Tense")
            self.pos_label.grid()
        else:
            self.pos_label.grid_remove()
        
        # Audio
        self._update_audio_player()
        
        # Explanation
        self.explanation_label.configure(text=card.explanation or "")
        self.explanation_label.grid() if card.explanation else self.explanation_label.grid_remove()
        
        # Hide example frame
        self.example_frame.grid_remove()
        
        # Build conjugation table in grammar_frame
        self.grammar_frame.grid()
        for widget in self.grammar_frame.winfo_children():
            widget.destroy()
        
        # Table title
        table_title = ttk.Label(
            self.grammar_frame,
            text="📋 Conjugation Table",
            font=("Helvetica", 16, "bold"),
            foreground="#ffd43b",
        )
        table_title.grid(row=0, column=0, columnspan=2, pady=(10, 15))
        
        # Create conjugation table
        if card.conjugations:
            row_idx = 1
            
            # Define the order of pronouns for different languages
            pronoun_order = ["ich", "du", "er/sie/es", "wir", "ihr", "sie/Sie",  # German
                           "je", "tu", "il/elle", "nous", "vous", "ils/elles",  # French
                           "yo", "tú", "él/ella", "nosotros", "vosotros", "ellos/ellas",  # Spanish
                           "I", "you", "he/she", "we", "you (pl)", "they"]  # Fallback
            
            # Sort conjugations by pronoun order if possible
            sorted_conjugations = []
            for pronoun in pronoun_order:
                if pronoun in card.conjugations:
                    sorted_conjugations.append((pronoun, card.conjugations[pronoun]))
            # Add any remaining conjugations not in the order
            for pronoun, form in card.conjugations.items():
                if (pronoun, form) not in sorted_conjugations:
                    sorted_conjugations.append((pronoun, form))
            
            for pronoun, conjugated_form in sorted_conjugations:
                # Pronoun column
                pronoun_label = ttk.Label(
                    self.grammar_frame,
                    text=pronoun,
                    font=("Helvetica", 14),
                    foreground="#9bc6ff",
                    width=12,
                    anchor="e",
                )
                pronoun_label.grid(row=row_idx, column=0, padx=(20, 10), pady=3, sticky="e")
                
                # Conjugated form column
                form_label = ttk.Label(
                    self.grammar_frame,
                    text=conjugated_form,
                    font=("Helvetica", 16, "bold"),
                    foreground="#ffffff",
                    anchor="w",
                )
                form_label.grid(row=row_idx, column=1, padx=(10, 20), pady=3, sticky="w")
                
                row_idx += 1
        
        # Add example sentences
        if card.conjugation_examples:
            examples_title = ttk.Label(
                self.grammar_frame,
                text="📝 Example Sentences",
                font=("Helvetica", 14, "bold"),
                foreground="#69db7c",
            )
            examples_title.grid(row=row_idx, column=0, columnspan=2, pady=(20, 10))
            row_idx += 1
            
            for example in card.conjugation_examples[:4]:
                sentence = example.get("sentence", "")
                translation = example.get("translation", "")
                
                if sentence:
                    sentence_label = ttk.Label(
                        self.grammar_frame,
                        text=f"• {sentence}",
                        font=("Helvetica", 13),
                        foreground="#ffffff",
                        wraplength=500,
                    )
                    sentence_label.grid(row=row_idx, column=0, columnspan=2, padx=30, pady=(5, 0), sticky="w")
                    row_idx += 1
                    
                    if translation:
                        trans_label = ttk.Label(
                            self.grammar_frame,
                            text=f"  → {translation}",
                            font=("Helvetica", 12),
                            foreground="#9bc6ff",
                            wraplength=500,
                        )
                        trans_label.grid(row=row_idx, column=0, columnspan=2, padx=40, pady=(0, 5), sticky="w")
                        row_idx += 1
        
        # Hide related/mnemonic for conjugation cards
        self.related_frame.grid_remove()
        self.mnemonic_label.grid_remove()
    
    def _update_audio_player(self) -> None:
        """Update or create audio player for current card."""
        if not self.teaching_plan:
            return
        
        card = self.teaching_plan.cards[self.current_index]
        
        # Clear existing audio player
        for widget in self.audio_frame.winfo_children():
            widget.destroy()
        
        if card.audio_path:
            self.audio_player = AudioPlayer(self.audio_frame)
            self.audio_player.pack(pady=5)
            self.audio_player.set_audio(card.audio_path)
            self.audio_frame.grid()
        elif card.audio_text:
            # Audio is being generated
            loading_label = ttk.Label(
                self.audio_frame,
                text="🔊 Generating audio...",
                font=("Helvetica", 11),
                foreground="#7bb3ff",
            )
            loading_label.pack(pady=5)
            self.audio_frame.grid()
        else:
            self.audio_frame.grid_remove()
    
    def _on_prev(self) -> None:
        """Go to previous card."""
        if self.current_index > 0:
            self.current_index -= 1
            self._render_current_card()
    
    def _on_next(self) -> None:
        """Go to next card or start quiz."""
        if not self.teaching_plan:
            return
        
        if self.current_index >= len(self.teaching_plan.cards) - 1:
            # Finished teaching, proceed to quiz
            self._on_skip_to_quiz()
        else:
            self.current_index += 1
            self._render_current_card()
    
    def _on_skip_to_quiz(self) -> None:
        """Skip to the quiz section."""
        logger.ui("User finished teaching section, proceeding to quiz...")
        self.controller.proceed_to_quiz()


# ---------------------------------------------------------------------------
# Lesson card (10 prompts, one at a time)
# ---------------------------------------------------------------------------

class LessonCardView(ttk.Frame):
    def __init__(self, parent, controller: LanguageBuddyApp) -> None:
        super().__init__(parent)
        self.controller = controller

        # Use scrollable frame for content
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        
        self.scrollable = ScrollableFrame(self)
        self.scrollable.grid(row=0, column=0, sticky="nsew")
        self.content = self.scrollable.content
        
        # Configure content to center widgets
        self.content.columnconfigure(0, weight=1)

        self.title_label = ttk.Label(
            self.content,
            text="Lesson Prompts",
            font=("Helvetica", 24, "bold"),
            anchor="center",
        )
        self.title_label.grid(row=0, column=0, pady=(20, 10))

        self.subtitle_label = ttk.Label(
            self.content,
            text="Work through the prompts one by one.",
            wraplength=600,
            justify="center",
            anchor="center",
            font=("Helvetica", 14),
            foreground="#d0d0d0",  # Light gray text on dark background
        )
        self.subtitle_label.grid(row=1, column=0, padx=30, pady=(0, 15))

        # Loading spinner
        self.loading_spinner = LoadingSpinner(self.content, text="Evaluating and generating lesson...")
        self.loading_spinner.grid(row=2, column=0, pady=(40, 40))

        self.progress_label = ttk.Label(
            self.content, 
            text="",
            anchor="center",
            foreground="#7bb3ff",  # Light blue text on dark background
        )
        self.progress_label.grid(row=2, column=0, pady=(0, 10))
        
        # Image area (for image-based questions) - responsive sizing
        self.responsive_image = ResponsiveImage(self.content)
        self.responsive_image.grid(row=3, column=0, padx=20, pady=(0, 10))
        self.responsive_image.grid_remove()
        
        # Question label
        self.question_label = ttk.Label(
            self.content,
            text="",
            font=("Helvetica", 24, "bold"),
            wraplength=600,
            justify="center",
            anchor="center",
            foreground="#ffffff",  # White text on dark background
        )
        self.question_label.grid(row=4, column=0, padx=30, pady=(15, 8))
        
        # Answer area (dynamic based on card type)
        self.answer_frame = ttk.Frame(self.content)
        self.answer_frame.grid(row=5, column=0, padx=20, pady=(10, 10))
        
        # Multiple choice buttons (will be created dynamically)
        self.mc_buttons: List[ttk.Button] = []
        self.selected_mc_index: Optional[int] = None
        
        # Text input (for text questions)
        self.text_widget: Optional[tk.Text] = None
        self.text_scrollbar: Optional[ttk.Scrollbar] = None
        
        # Audio player (for audio questions)
        self.audio_player: Optional[AudioPlayer] = None
        self.current_audio_path: Optional[str] = None
        
        # Speech recorder (for speaking exercises)
        self.speech_recorder: Optional[SpeechRecorder] = None
        self.current_transcription: Optional[str] = None

        self.next_button = ttk.Button(
            self.content, text="Submit Answer", command=self._on_next_clicked
        )
        self.next_button.grid(row=6, column=0, pady=(10, 20))
        self.submit_button = self.next_button

        # Feedback card
        self.feedback_card = ttk.Frame(self.content)
        self.feedback_card.grid(row=7, column=0, padx=20, pady=(5, 40))
        self.feedback_card.grid_remove()

        self.feedback_title = ttk.Label(
            self.feedback_card,
            text="Feedback",
            font=("Helvetica", 20, "bold"),
            justify="center",
            anchor="center",
        )
        self.feedback_title.grid(row=0, column=0, pady=(4, 6))

        # Simplified feedback body (no nested canvas - outer ScrollableFrame handles scrolling)
        self.feedback_body = ttk.Frame(self.feedback_card)
        self.feedback_body.grid(row=1, column=0)

        self.feedback_label = ttk.Label(
            self.feedback_body,
            text="",
            wraplength=600,
            justify="center",
            anchor="center",
            foreground="#d0d0d0",
        )
        self.feedback_label.grid(row=0, column=0, pady=(0, 8))

        self.vocab_expansion_label = ttk.Label(
            self.feedback_body,
            text="",
            wraplength=600,
            justify="center",
            anchor="center",
            font=("Helvetica", 14, "bold"),
            foreground="#7bb3ff",
        )
        self.vocab_expansion_label.grid(row=1, column=0, pady=(0, 6))

        self.continue_button = ttk.Button(
            self.feedback_card,
            text="Continue",
            command=self._on_continue_after_feedback,
        )
        self.continue_button.grid(row=2, column=0, pady=(10, 5))

        self.feedback_spinner = LoadingSpinner(
            self.content, text="Evaluating your answer..."
        )
        self.feedback_spinner.grid(row=7, column=0, pady=(10, 30))
        self.feedback_spinner.grid_remove()

        # Responsive wrapping for question/feedback text
        self.scrollable.canvas.bind("<Configure>", self._update_wraplengths)

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

    def show_loading(self, message: str = "Loading...") -> None:
        """Show the loading spinner."""
        self.loading_spinner.text = message
        self.loading_spinner.label.configure(text=f"{self.loading_spinner.spinner_chars[0]} {message}")
        self.loading_spinner.start()
        self.progress_label.grid_remove()
        self.responsive_image.grid_remove()
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
        width = self.scrollable.canvas.winfo_width() or self.winfo_width()
        wrap = max(200, width - 60)
        self.subtitle_label.configure(wraplength=wrap)
        self.question_label.configure(wraplength=wrap)
        self.feedback_label.configure(wraplength=wrap)
        self.vocab_expansion_label.configure(wraplength=wrap)

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
        
        # Scroll to top when showing new card
        self.scrollable.scroll_to_top()

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
                self.responsive_image.set_placeholder("Loading image...")
                self.responsive_image.grid()
        else:
            self.responsive_image.grid_remove() 

        # Render answer area based on card type
        if card.type == "multiple_choice":
            self._render_multiple_choice(card)
        elif card.type == "vocabulary":
            self._render_vocabulary(card)
        elif card.type in ("audio_transcription", "audio_comprehension"):
            self._render_audio_card(card)
        elif card.type == "speaking":
            self._render_speaking_card(card)
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
                width=50,  # Fixed width for consistent centering
            )
            btn.grid(row=i, column=0, pady=(4, 4), padx=20)
            self.mc_buttons.append(btn)
        self.answer_frame.grid()

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
        
        if not hasattr(self, 'vocab_label') or not self.vocab_label:
            self.vocab_label = ttk.Label(self.answer_frame, text="", wraplength=600, justify="center", anchor="center")
            self.vocab_label.grid(row=0, column=0, padx=20, pady=10)
        self.vocab_label.configure(text=vocab_text)
        self.answer_frame.grid()
    
    def _render_audio_card(self, card: LessonCard) -> None:
        """Render audio transcription or comprehension card."""
        # Clear multiple choice buttons if they exist
        if hasattr(self, 'mc_buttons'):
            for btn in self.mc_buttons:
                btn.destroy()
            self.mc_buttons = []
        
        # Hide image area for audio cards
        self.responsive_image.grid_remove()
        
        # Create or update audio player - place it in content frame (row 3, where image normally goes)
        if not self.audio_player:
            self.audio_player = AudioPlayer(
                self.content,
                on_skip=self._on_skip_audio
            )
        
        # Place audio player in the content area (between progress label and question)
        self.audio_player.grid(row=3, column=0, pady=(10, 10))
        
        # Check if we already have audio for this card
        if card.audio_path and os.path.exists(card.audio_path):
            # Audio already generated
            self.audio_player.set_audio(card.audio_path)
        else:
            # Need to generate audio
            self.audio_player.set_loading("Generating audio...")
            self._generate_audio_for_card(card)
        
        # Create text input for transcription/answer in answer_frame
        if not hasattr(self, 'text_widget') or not self.text_widget:
            text_frame = ttk.Frame(self.answer_frame)
            text_frame.grid(row=0, column=0, pady=(10, 0))
            
            self.text_widget = tk.Text(
                text_frame, 
                height=4,
                width=60,
                wrap="word",
                bg="#2d2d2d",
                fg="#e0e0e0",
                insertbackground="#e0e0e0",
                selectbackground="#3d3d3d",
                selectforeground="#ffffff",
            )
            self.text_widget.grid(row=0, column=0, pady=(5, 0))
            
            self.text_scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.text_widget.yview)
            self.text_scrollbar.grid(row=0, column=1, sticky="ns")
            self.text_widget.configure(yscrollcommand=self.text_scrollbar.set)
        else:
            # Update existing text widget
            self.text_widget.delete("1.0", tk.END)
            self.text_widget.grid()
            if self.text_scrollbar:
                self.text_scrollbar.grid()
        
        self.answer_frame.grid()
    
    def _render_speaking_card(self, card: LessonCard) -> None:
        """Render speaking exercise card with recording interface."""
        # Clear multiple choice buttons if they exist
        if hasattr(self, 'mc_buttons'):
            for btn in self.mc_buttons:
                btn.destroy()
            self.mc_buttons = []
        
        # Hide image and audio areas
        self.responsive_image.grid_remove()
        if self.audio_player:
            self.audio_player.grid_remove()
        
        # Reset transcription
        self.current_transcription = None
        
        # Create speech recorder if needed
        if not self.speech_recorder:
            self.speech_recorder = SpeechRecorder(
                self.content,
                on_recording_complete=self._on_recording_complete,
                on_skip=self._on_skip_speaking
            )
        
        # Set the prompt and show the recorder
        speaking_prompt = card.speaking_prompt or card.correct_answer or ""
        self.speech_recorder.set_prompt(speaking_prompt)
        self.speech_recorder.grid(row=3, column=0, pady=(10, 10))
        
        # Hide the answer frame for speaking cards (recorder handles input)
        self.answer_frame.grid_remove()
    
    def _on_recording_complete(self, audio_path: str) -> None:
        """Handle completed recording - transcribe the audio."""
        language = self.controller.selected_language or "Spanish"
        
        def on_transcription_complete(transcription: Optional[str]) -> None:
            """Callback when transcription is complete."""
            def update_ui():
                if transcription:
                    self.current_transcription = transcription
                    if self.speech_recorder:
                        self.speech_recorder.show_transcription(transcription)
                    logger.ui(f"Transcription: {transcription}")
                else:
                    if self.speech_recorder:
                        self.speech_recorder.show_error("Could not transcribe audio. Try again.")
                    logger.error("Transcription failed")
            
            self.after(0, update_ui)
        
        # Transcribe in background
        transcribe_audio_async(audio_path, language, on_transcription_complete)
    
    def _on_skip_audio(self) -> None:
        """Handle skipping an audio exercise (TTS listening)."""
        logger.ui("Skipping audio exercise")
        
        plan = self.controller.lesson_plan
        if not plan or self.controller.lesson_index >= len(plan.cards):
            return
        
        card = plan.cards[self.controller.lesson_index]
        
        # Mark as skipped with partial credit
        card.is_correct = False
        card.card_score = 0
        card.user_response = "[SKIPPED - Audio exercise]"
        card.feedback = "Exercise skipped. Audio exercises help develop listening skills - try them when you can!"
        
        # Show brief feedback then move to next card
        self._show_skip_feedback("Audio exercise skipped", card)
    
    def _on_skip_speaking(self) -> None:
        """Handle skipping a speaking exercise (STT recording)."""
        logger.ui("Skipping speaking exercise")
        
        plan = self.controller.lesson_plan
        if not plan or self.controller.lesson_index >= len(plan.cards):
            return
        
        card = plan.cards[self.controller.lesson_index]
        
        # Mark as skipped with partial credit
        card.is_correct = False
        card.card_score = 0
        card.user_response = "[SKIPPED - Speaking exercise]"
        card.feedback = "Exercise skipped. Speaking practice helps with pronunciation - try it when you can!"
        
        # Show brief feedback then move to next card
        self._show_skip_feedback("Speaking exercise skipped", card)
    
    def _show_skip_feedback(self, message: str, card: LessonCard) -> None:
        """Show brief feedback for skipped exercises and move to next card."""
        # Create a simple feedback display
        feedback_data = {
            "is_correct": False,
            "card_score": 0,
            "feedback": card.feedback,
            "correct_answer": card.correct_answer or card.speaking_prompt or "",
            "alternatives": [],
            "vocabulary_expansion": card.vocabulary_expansion or [],
        }
        
        # Show the feedback
        self.show_feedback(feedback_data)
    
    def _generate_audio_for_card(self, card: LessonCard) -> None:
        """Generate TTS audio for a card asynchronously."""
        if not card.audio_text:
            logger.warning("Card has no audio_text, cannot generate audio")
            self.audio_player.status_label.configure(text="No audio text available")
            return
        
        language = self.controller.selected_language or "Spanish"
        
        def on_audio_complete(audio_path: Optional[str]) -> None:
            """Callback when audio generation completes."""
            def update_ui():
                if audio_path:
                    card.audio_path = audio_path
                    self.current_audio_path = audio_path
                    self.audio_player.set_audio(audio_path)
                    logger.ui(f"Audio ready for playback: {audio_path}")
                else:
                    self.audio_player.status_label.configure(text="Failed to generate audio")
                    logger.error("Audio generation failed")
            
            # Update UI on main thread
            self.after(0, update_ui)
        
        # Generate audio in background
        generate_speech_async(card.audio_text, language, on_audio_complete)
    
    def _render_text_input(self, card: LessonCard) -> None:
        """Render text input area."""
        # Clear multiple choice buttons if they exist
        if hasattr(self, 'mc_buttons'):
            for btn in self.mc_buttons:
                btn.destroy()
            self.mc_buttons = []
        
        if not hasattr(self, 'text_widget') or not self.text_widget:
            text_frame = ttk.Frame(self.answer_frame)
            text_frame.grid(row=0, column=0)
            
            self.text_widget = tk.Text(
                text_frame, 
                height=6,
                width=60,  # Fixed width for consistent centering
                wrap="word",
                bg="#2d2d2d",  # Dark background
                fg="#e0e0e0",  # Light text
                insertbackground="#e0e0e0",  # Light cursor
                selectbackground="#3d3d3d",  # Dark selection background
                selectforeground="#ffffff",  # White selected text
            )
            self.text_widget.grid(row=0, column=0, pady=(5, 0))
            
            self.text_scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.text_widget.yview)
            self.text_scrollbar.grid(row=0, column=1, sticky="ns")
            self.text_widget.configure(yscrollcommand=self.text_scrollbar.set)
        
        self.text_widget.delete("1.0", tk.END)
        self.text_widget.grid()
        self.answer_frame.grid()
    
    def _clear_card_widgets(self) -> None:
        """Clear all card rendering widgets."""
        self.answer_frame.grid_remove()
        
        # Destroy ALL children of answer_frame to prevent layout issues
        for child in self.answer_frame.winfo_children():
            child.destroy()
        
        # Reset widget references
        self.mc_buttons = []
        self.text_widget = None
        self.text_scrollbar = None
        if hasattr(self, 'vocab_label'):
            self.vocab_label = None
        
        # Clean up audio player (now in content frame)
        if self.audio_player:
            self.audio_player.stop()
            self.audio_player.grid_remove()
        
        # Clean up speech recorder
        if self.speech_recorder:
            self.speech_recorder.reset()
            self.speech_recorder.grid_remove()
        self.current_transcription = None
    
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
        self.responsive_image.set_image(image_path)
        self.responsive_image.grid()
    
    
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
        elif card.type == "speaking":
            # Speaking cards use the transcription
            if not self.current_transcription:
                messagebox.showinfo("Record your speech", "Please record yourself saying the phrase first.")
                return
            response = self.current_transcription
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
# Assessment Results Card
# ---------------------------------------------------------------------------

class AssessmentResultsCard(ttk.Frame):
    """Displays the assessment results with detailed feedback before moving to lessons."""
    
    def __init__(self, parent, controller: "LanguageBuddyApp") -> None:
        super().__init__(parent)
        self.controller = controller
        
        # Use scrollable frame for content
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        
        self.scrollable = ScrollableFrame(self)
        self.scrollable.grid(row=0, column=0, sticky="nsew")
        self.content = self.scrollable.content
        
        # Configure content to center widgets
        self.content.columnconfigure(0, weight=1)
        
        # Title
        self.title_label = ttk.Label(
            self.content,
            text="📊 Assessment Complete",
            font=("Helvetica", 24, "bold"),
            foreground="#ffffff",
            anchor="center",
        )
        self.title_label.grid(row=0, column=0, pady=(30, 20), padx=20)
        
        # Results container
        self.results_frame = ttk.Frame(self.content)
        self.results_frame.grid(row=1, column=0, padx=40, pady=10, sticky="ew")
        self.results_frame.columnconfigure(0, weight=1)
        
        # Continue button
        self.continue_button = ttk.Button(
            self.content,
            text="Start Learning →",
            command=self._on_continue_clicked,
        )
        self.continue_button.grid(row=2, column=0, pady=(30, 40))
        
        # Style the button
        style = ttk.Style()
        style.configure("Continue.TButton", font=("Helvetica", 14, "bold"), padding=10)
        self.continue_button.configure(style="Continue.TButton")
    
    def show_results(self, assessment_result: AssessmentResult, language: str) -> None:
        """Display the assessment results with detailed feedback."""
        # Clear previous results
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        row = 0
        
        # === PROFICIENCY LEVEL SECTION ===
        level_frame = ttk.Frame(self.results_frame)
        level_frame.grid(row=row, column=0, pady=(0, 25), sticky="ew")
        level_frame.columnconfigure(0, weight=1)
        
        # Big proficiency display
        level_color = self._get_level_color(assessment_result.proficiency)
        level_label = ttk.Label(
            level_frame,
            text=assessment_result.proficiency,
            font=("Helvetica", 72, "bold"),
            foreground=level_color,
            anchor="center",
        )
        level_label.grid(row=0, column=0)
        
        level_name = self._get_level_name(assessment_result.proficiency)
        level_name_label = ttk.Label(
            level_frame,
            text=level_name,
            font=("Helvetica", 18),
            foreground="#d0d0d0",
            anchor="center",
        )
        level_name_label.grid(row=1, column=0, pady=(5, 0))
        
        # Language label
        lang_label = ttk.Label(
            level_frame,
            text=f"in {language}",
            font=("Helvetica", 14),
            foreground="#7bb3ff",
            anchor="center",
        )
        lang_label.grid(row=2, column=0, pady=(5, 0))
        
        row += 1
        
        # === SKILLS BREAKDOWN SECTION ===
        skills_section = self._create_section(
            self.results_frame, row, "📈 Skills Breakdown"
        )
        row += 1
        
        skills_frame = ttk.Frame(self.results_frame)
        skills_frame.grid(row=row, column=0, pady=(0, 20), padx=20, sticky="ew")
        skills_frame.columnconfigure((0, 1, 2), weight=1)
        
        # Vocabulary
        self._create_skill_card(skills_frame, 0, "📖 Vocabulary", 
                               assessment_result.vocabulary_level)
        # Grammar
        self._create_skill_card(skills_frame, 1, "✏️ Grammar", 
                               assessment_result.grammar_level)
        # Fluency
        fluency_level = self._score_to_level(assessment_result.fluency_score)
        self._create_skill_card(skills_frame, 2, "💬 Fluency", 
                               f"{assessment_result.fluency_score}/100")
        
        row += 1
        
        # === STRENGTHS SECTION ===
        if assessment_result.strengths:
            self._create_section(self.results_frame, row, "✅ Your Strengths")
            row += 1
            
            strengths_frame = ttk.Frame(self.results_frame)
            strengths_frame.grid(row=row, column=0, pady=(0, 20), padx=30, sticky="w")
            
            for i, strength in enumerate(assessment_result.strengths[:5]):
                strength_label = ttk.Label(
                    strengths_frame,
                    text=f"• {strength}",
                    font=("Helvetica", 13),
                    foreground="#69db7c",
                    wraplength=500,
                    justify="left",
                )
                strength_label.grid(row=i, column=0, sticky="w", pady=2)
            row += 1
        
        # === AREAS TO IMPROVE SECTION ===
        if assessment_result.weaknesses:
            self._create_section(self.results_frame, row, "🎯 Areas to Improve")
            row += 1
            
            weaknesses_frame = ttk.Frame(self.results_frame)
            weaknesses_frame.grid(row=row, column=0, pady=(0, 20), padx=30, sticky="w")
            
            for i, weakness in enumerate(assessment_result.weaknesses[:5]):
                weakness_label = ttk.Label(
                    weaknesses_frame,
                    text=f"• {weakness}",
                    font=("Helvetica", 13),
                    foreground="#ffa94d",
                    wraplength=500,
                    justify="left",
                )
                weakness_label.grid(row=i, column=0, sticky="w", pady=2)
            row += 1
        
        # === RECOMMENDATIONS SECTION ===
        if assessment_result.recommendations:
            self._create_section(self.results_frame, row, "💡 Recommendations")
            row += 1
            
            recs_frame = ttk.Frame(self.results_frame)
            recs_frame.grid(row=row, column=0, pady=(0, 20), padx=30, sticky="w")
            
            for i, rec in enumerate(assessment_result.recommendations[:5]):
                rec_label = ttk.Label(
                    recs_frame,
                    text=f"• {rec}",
                    font=("Helvetica", 13),
                    foreground="#9bc6ff",
                    wraplength=500,
                    justify="left",
                )
                rec_label.grid(row=i, column=0, sticky="w", pady=2)
            row += 1
        
        # === LEVEL EXPLANATION SECTION ===
        self._create_section(self.results_frame, row, "📚 What This Level Means")
        row += 1
        
        explanation = self._get_level_explanation(assessment_result.proficiency)
        explanation_label = ttk.Label(
            self.results_frame,
            text=explanation,
            font=("Helvetica", 12),
            foreground="#d0d0d0",
            wraplength=550,
            justify="left",
        )
        explanation_label.grid(row=row, column=0, pady=(0, 20), padx=30, sticky="w")
        
        # Scroll to top
        self.scrollable.canvas.yview_moveto(0)
    
    def _create_section(self, parent, row: int, title: str) -> ttk.Label:
        """Create a section header."""
        label = ttk.Label(
            parent,
            text=title,
            font=("Helvetica", 16, "bold"),
            foreground="#7bb3ff",
        )
        label.grid(row=row, column=0, pady=(15, 10), padx=20, sticky="w")
        return label
    
    def _create_skill_card(self, parent, col: int, skill_name: str, level: str) -> None:
        """Create a skill indicator card."""
        frame = ttk.Frame(parent)
        frame.grid(row=0, column=col, padx=10, pady=5)
        
        name_label = ttk.Label(
            frame,
            text=skill_name,
            font=("Helvetica", 12),
            foreground="#d0d0d0",
            anchor="center",
        )
        name_label.grid(row=0, column=0)
        
        # Determine color based on level
        if "/" in level:  # It's a score like "75/100"
            color = "#7bb3ff"
        else:
            color = self._get_level_color(level)
        
        level_label = ttk.Label(
            frame,
            text=level,
            font=("Helvetica", 20, "bold"),
            foreground=color,
            anchor="center",
        )
        level_label.grid(row=1, column=0, pady=(5, 0))
    
    def _get_level_color(self, level: str) -> str:
        """Get color for proficiency level."""
        colors = {
            "A1": "#ff6b6b",  # Red - Beginner
            "A2": "#ffa94d",  # Orange
            "B1": "#ffd43b",  # Yellow
            "B2": "#69db7c",  # Green
            "C1": "#4dabf7",  # Blue
            "C2": "#da77f2",  # Purple - Mastery
        }
        return colors.get(level, "#ffffff")
    
    def _get_level_name(self, level: str) -> str:
        """Get human-readable name for level."""
        names = {
            "A1": "Beginner",
            "A2": "Elementary",
            "B1": "Intermediate",
            "B2": "Upper Intermediate",
            "C1": "Advanced",
            "C2": "Proficient / Near-Native",
        }
        return names.get(level, "Unknown")
    
    def _score_to_level(self, score: int) -> str:
        """Convert fluency score to approximate level."""
        if score >= 90:
            return "C2"
        elif score >= 75:
            return "C1"
        elif score >= 60:
            return "B2"
        elif score >= 45:
            return "B1"
        elif score >= 30:
            return "A2"
        else:
            return "A1"
    
    def _get_level_explanation(self, level: str) -> str:
        """Get detailed explanation of what the level means."""
        explanations = {
            "A1": (
                "At A1 (Beginner), you can understand and use familiar everyday expressions "
                "and very basic phrases. You can introduce yourself and others, ask and answer "
                "questions about personal details such as where you live, people you know, "
                "and things you have. You can interact in a simple way provided the other "
                "person talks slowly and clearly."
            ),
            "A2": (
                "At A2 (Elementary), you can understand sentences and frequently used expressions "
                "related to areas of most immediate relevance (e.g., personal and family information, "
                "shopping, local geography, employment). You can communicate in simple and routine "
                "tasks requiring a direct exchange of information on familiar matters. You can "
                "describe aspects of your background and immediate environment."
            ),
            "B1": (
                "At B1 (Intermediate), you can understand the main points of clear standard input "
                "on familiar matters regularly encountered in work, school, leisure, etc. You can "
                "deal with most situations likely to arise while traveling. You can produce simple "
                "connected text on familiar topics. You can describe experiences, events, dreams, "
                "hopes, and ambitions and briefly give reasons for opinions and plans."
            ),
            "B2": (
                "At B2 (Upper Intermediate), you can understand the main ideas of complex text on "
                "both concrete and abstract topics, including technical discussions in your field. "
                "You can interact with a degree of fluency and spontaneity that makes regular "
                "interaction with native speakers quite possible without strain. You can produce "
                "clear, detailed text on a wide range of subjects."
            ),
            "C1": (
                "At C1 (Advanced), you can understand a wide range of demanding, longer texts and "
                "recognize implicit meaning. You can express yourself fluently and spontaneously "
                "without much obvious searching for expressions. You can use language flexibly "
                "and effectively for social, academic, and professional purposes. You can produce "
                "clear, well-structured, detailed text on complex subjects."
            ),
            "C2": (
                "At C2 (Proficient), you can understand with ease virtually everything heard or read. "
                "You can summarize information from different spoken and written sources, "
                "reconstructing arguments and accounts in a coherent presentation. You can express "
                "yourself spontaneously, very fluently and precisely, differentiating finer shades "
                "of meaning even in the most complex situations."
            ),
        }
        return explanations.get(level, "Level information not available.")
    
    def _on_continue_clicked(self) -> None:
        """Handle continue button click - proceed to lesson generation."""
        self.controller.proceed_to_lessons()


# ---------------------------------------------------------------------------
# Summary card
# ---------------------------------------------------------------------------

class SummaryCard(ttk.Frame):
    def __init__(self, parent, controller: LanguageBuddyApp) -> None:
        super().__init__(parent)
        self.controller = controller

        # Use scrollable frame for content
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        
        self.scrollable = ScrollableFrame(self)
        self.scrollable.grid(row=0, column=0, sticky="nsew")
        self.content = self.scrollable.content
        
        # Configure content to center widgets
        self.content.columnconfigure(0, weight=1)

        title = ttk.Label(
            self.content,
            text="Session Summary",
            font=("Helvetica", 28, "bold"),
            anchor="center",
            foreground="#ffffff",  # White text on dark background
        )
        title.grid(row=0, column=0, pady=(30, 15))

        # Loading spinner
        self.loading_spinner = LoadingSpinner(self.content, text="Generating summary...")
        self.loading_spinner.grid(row=1, column=0, pady=(40, 40))

        self.summary_label = ttk.Label(
            self.content, 
            text="", 
            wraplength=600, 
            justify="center",
            anchor="center",
            font=("Helvetica", 14),
            foreground="#d0d0d0",  # Light gray text on dark background
        )
        self.summary_label.grid(row=1, column=0, padx=30, pady=(0, 30))
        self.scrollable.canvas.bind("<Configure>", self._on_summary_resize)

        button_frame = ttk.Frame(self.content)
        button_frame.grid(row=2, column=0, pady=(10, 50))

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
        wrap = max(200, event.width - 60)
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
        """Set summary from final summary dict with change markers."""
        self.hide_loading()
        self.scrollable.scroll_to_top()
        
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
            
            # Get session stats for change markers
            stats = self.controller.session_stats
            teaching_plan = self.controller.teaching_plan
            
            # Build change markers section
            change_lines = []
            
            if stats:
                # Words learned this session
                if teaching_plan:
                    new_words = teaching_plan.new_words_count
                    if new_words > 0:
                        change_lines.append(f"📚 +{new_words} new words learned")
                    if teaching_plan.review_words_count > 0:
                        change_lines.append(f"🔄 {teaching_plan.review_words_count} words reviewed")
                
                # Fluency change
                fluency_after = self.controller.assessment_result.fluency_score if self.controller.assessment_result else 0
                fluency_diff = fluency_after - stats.fluency_before
                if fluency_diff > 0:
                    change_lines.append(f"📈 Fluency: +{fluency_diff} ({stats.fluency_before} → {fluency_after})")
                elif fluency_diff < 0:
                    change_lines.append(f"📉 Fluency: {fluency_diff} ({stats.fluency_before} → {fluency_after})")
                else:
                    change_lines.append(f"📊 Fluency: {fluency_after}/100")
                
                # Proficiency level change
                prof_after = self.controller.assessment_result.proficiency if self.controller.assessment_result else "A1"
                if stats.proficiency_before != prof_after:
                    change_lines.append(f"🎯 Level: {stats.proficiency_before} → {prof_after}")
            
            # Quiz performance
            if lesson_plan and lesson_plan.cards:
                correct = sum(1 for card in lesson_plan.cards if card.is_correct)
                total = len(lesson_plan.cards)
                change_lines.append(f"✅ Quiz: {correct}/{total} correct ({overall_score}%)")

            text_lines = [
                "═══════════════════════════════════════",
                "📊 SESSION RESULTS",
                "═══════════════════════════════════════",
                "",
            ]
            
            if change_lines:
                text_lines.extend([
                    "📈 Your Progress This Session:",
                    *[f"  {line}" for line in change_lines],
                    "",
                ])
            
            text_lines.extend([
                f"Progress: {proficiency_improvement}",
                "",
                "✅ Strengths:",
                *[f"  • {s}" for s in (strengths or ["Keep up the good work!"])],
                "",
                "🎯 Areas to Improve:",
                *[f"  • {a}" for a in (areas_to_improve or ["Continue practicing"])],
                "",
                "📖 Study Suggestions:",
                *[f"  • {s}" for s in (study_suggestions or ["Practice regularly", "Review vocabulary daily"])],
                "",
                "═══════════════════════════════════════",
                "",
                "Ready for another lesson?",
            ])
            text = "\n".join(text_lines)

        self.summary_label.configure(text=text)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.separator("Application Starting")
    logger.info("Creating main application window...")
    app = LanguageBuddyApp()
    logger.success("Application window created, entering main loop")
    logger.info("Ready for user interaction")
    app.mainloop()
    logger.separator("Application Closed")

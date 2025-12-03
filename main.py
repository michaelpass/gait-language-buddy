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

from typing import Optional, List, Dict, Any, Set

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
try:
    import sounddevice as sd
    import soundfile as sf
    import tempfile
    RECORDING_AVAILABLE = True
except ImportError:
    RECORDING_AVAILABLE = False
    print("Note: sounddevice/soundfile not installed. Recording will be disabled.")
    print("Install with: pip install sounddevice soundfile")

from core.logger import logger
from core.api import (
    SUPPORTED_LANGUAGES,
    is_api_available,  # Check if API key is loaded
    generate_assessment_cards,
    evaluate_assessment_responses,
    generate_structured_lesson_plan,
    generate_lesson_plan_from_assessment_responses,  # Optimized combined function
    evaluate_card_response,
    generate_final_summary,
    generate_image_async,
    generate_images_parallel,  # Parallel image generation
    generate_speech_async,  # TTS generation
    transcribe_audio_async,  # STT transcription
)
from core.models import (
    AssessmentCard, AssessmentResult, LessonPlan, LessonCard
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
    
    def __init__(self, parent, **kwargs) -> None:
        super().__init__(parent, **kwargs)
        
        self._audio_path: Optional[str] = None
        self._is_playing: bool = False
        self._is_loaded: bool = False
        self._playback_finished: bool = False
        
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
        self.play_button.pack(padx=5)
        
        # Status label
        self.status_label = ttk.Label(
            self,
            text="",
            font=("Helvetica", 11),
            foreground="#7bb3ff",
            anchor="center",
        )
        self.status_label.pack(pady=(5, 0))
        
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
    
    def _show_loading(self) -> None:
        """Show loading state."""
        self.loading_label.pack(pady=10)
        self.control_frame.pack_forget()
        self.status_label.pack_forget()
    
    def _show_controls(self) -> None:
        """Show playback controls."""
        self.loading_label.pack_forget()
        self.control_frame.pack(pady=10)
        self.status_label.pack(pady=(5, 0))
    
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
    """
    
    SAMPLE_RATE = 16000  # Whisper prefers 16kHz
    CHANNELS = 1  # Mono
    
    def __init__(self, parent, on_recording_complete: Optional[Callable[[str], None]] = None, **kwargs) -> None:
        super().__init__(parent, **kwargs)
        
        self._is_recording: bool = False
        self._recording_data: List = []
        self._recording_path: Optional[str] = None
        self._on_complete = on_recording_complete
        
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
        
        # Record button
        self.record_button = ttk.Button(
            self,
            text="🎤 Hold to Record",
            width=20,
        )
        self.record_button.pack(pady=10)
        
        # Bind press and release for hold-to-record
        self.record_button.bind("<ButtonPress-1>", self._start_recording)
        self.record_button.bind("<ButtonRelease-1>", self._stop_recording)
        
        # Status label
        self.status_label = ttk.Label(
            self,
            text="Press and hold the button to record",
            font=("Helvetica", 11),
            foreground="#7bb3ff",
            anchor="center",
        )
        self.status_label.pack(pady=(5, 10))
        
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
            self.status_label.configure(
                text="Recording not available. Install: pip install sounddevice soundfile",
                foreground="#ff6b6b"
            )
    
    def set_prompt(self, prompt_text: str) -> None:
        """Set the phrase the user should say."""
        self.prompt_label.configure(text=f'"{prompt_text}"')
        self.transcription_label.pack_forget()
        self.status_label.configure(text="Press and hold the button to record")
    
    def _start_recording(self, event=None) -> None:
        """Start recording audio."""
        if not RECORDING_AVAILABLE or self._is_recording:
            return
        
        self._is_recording = True
        self._recording_data = []
        self.record_button.configure(text="🔴 Recording...")
        self.status_label.configure(text="Recording... Release to stop", foreground="#ff6b6b")
        
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
    
    def _stop_recording(self, event=None) -> None:
        """Stop recording and save the audio."""
        if not self._is_recording:
            return
        
        self._is_recording = False
        self.record_button.configure(text="🎤 Hold to Record")
        self.status_label.configure(text="Processing...", foreground="#7bb3ff")
        
        # Save recording in a thread
        def save_and_callback():
            try:
                if not self._recording_data:
                    self.after(0, lambda: self.status_label.configure(
                        text="No audio recorded. Try again.", foreground="#ff6b6b"
                    ))
                    return
                
                import numpy as np
                # Concatenate all recorded chunks
                audio_data = np.concatenate(self._recording_data, axis=0)
                
                # Save to temp file
                fd, path = tempfile.mkstemp(suffix=".wav", prefix="gait_recording_")
                os.close(fd)
                sf.write(path, audio_data, self.SAMPLE_RATE)
                
                self._recording_path = path
                logger.ui(f"Recording saved: {path}")
                
                self.after(0, lambda: self.status_label.configure(
                    text="Recording complete! Transcribing...", foreground="#7bb3ff"
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
        self._recording_data = []
        self._recording_path = None
        self.transcription_label.pack_forget()
        self.status_label.configure(text="Press and hold the button to record", foreground="#7bb3ff")


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

        logger.ui("Creating card views...")
        for CardClass in (IntroCard, AssessmentCardView, LessonCardView, SummaryCard):
            card = CardClass(parent=container, controller=self)
            self.cards[CardClass.__name__] = card
            card.grid(row=0, column=0, sticky="nsew")
            logger.debug(f"  Created: {CardClass.__name__}")

        logger.ui("Application initialized successfully")
        self.show_card("IntroCard")

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

        # Show loading spinner and switch to assessment card
        assessment_card: AssessmentCardView = self.cards["AssessmentCardView"]
        assessment_card.show_loading("Generating assessment questions...")
        self.show_card("AssessmentCardView")

        # Run API calls in a background thread
        def generate_assessment_threaded():
            logger.task_start("generate_assessment_threaded")
            try:
                # Generate all assessment cards (fast text-only call)
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
        
        assessment_card: AssessmentCardView = self.cards["AssessmentCardView"]
        assessment_card.hide_loading()
        logger.ui("Showing assessment stage 1")
        assessment_card.show_stage(1, assessment_cards[0])

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
        logger.ui(f"Assessment response submitted for stage {stage}")
        
        if stage < 1 or stage > 3 or stage > len(self.assessment_cards):
            logger.warning(f"Invalid stage number: {stage}")
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
        logger.debug(f"Stored response for stage {stage}, total responses: {len(self.assessment_responses)}")
        
        # Move to next stage or evaluate
        if stage < 3:
            self.assessment_stage = stage + 1
            logger.ui(f"Moving to assessment stage {stage + 1}")
            assessment_card: AssessmentCardView = self.cards["AssessmentCardView"]
            assessment_card.show_stage(stage + 1, self.assessment_cards[stage])
        else:
            # All assessments done - evaluate and generate lesson plan
            logger.ui("All assessments complete, generating lesson plan...")
            assessment_card: AssessmentCardView = self.cards["AssessmentCardView"]
            assessment_card.show_loading("Evaluating your responses and generating personalized lessons...")
            
            def evaluate_and_generate_lesson():
                logger.task_start("evaluate_and_generate_lesson")
                try:
                    # OPTIMIZED: Combine assessment evaluation + lesson generation into ONE API call
                    logger.ui("Starting lesson generation...")
                    assessment_result, lesson_plan = generate_lesson_plan_from_assessment_responses(
                        self.assessment_responses,
                        self.selected_language or "Spanish"
                    )
                    logger.task_complete("evaluate_and_generate_lesson")
                    logger.ui("Lesson plan generated, switching to lesson view...")
                    
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
                        logger.ui(f"Starting background generation of {len(image_prompts_to_generate)} lesson images...")
                        generate_images_parallel(
                            image_prompts_to_generate,
                            lambda prompt, path: self._on_lesson_image_generated(prompt, path)
                        )
                    else:
                        logger.debug("No images to generate (all cached or no image prompts)")
                except Exception as e:
                    logger.task_error("evaluate_and_generate_lesson", str(e))
                    self.after(0, lambda: self._on_lesson_generation_error(str(e)))
            
            thread = threading.Thread(target=evaluate_and_generate_lesson, daemon=True)
            thread.start()
            logger.task("Spawned background thread for lesson generation")

    def _on_lesson_image_generated(self, image_prompt: str, image_path: Optional[str]) -> None:
        """Called when a lesson image is generated (in main thread)."""
        if image_path:
            logger.ui("Lesson image ready, updating cache...")
            self.image_cache[image_prompt] = image_path
            # Update UI if this card is currently displayed
            lesson_card: LessonCardView = self.cards["LessonCardView"]
            lesson_card.update_image_if_needed(image_prompt, image_path)

    def _on_lesson_plan_generated(self, assessment_result: AssessmentResult, lesson_plan: LessonPlan) -> None:
        """Called when lesson plan generation completes (in main thread)."""
        logger.ui("Lesson plan received, switching to lesson view...")
        logger.debug(f"Assessment: proficiency={assessment_result.proficiency}, "
                    f"fluency_score={assessment_result.fluency_score}")
        logger.debug(f"Lesson plan: {len(lesson_plan.cards)} cards, target={lesson_plan.proficiency_target}")
        
        self.assessment_result = assessment_result
        self.lesson_plan = lesson_plan
        self.lesson_index = 0
        
        # Hide assessment loading
        assessment_card: AssessmentCardView = self.cards["AssessmentCardView"]
        assessment_card.hide_loading()
        
        # Switch to lesson card view and show first card
        lesson_card: LessonCardView = self.cards["LessonCardView"]
        logger.ui("Showing first lesson card...")
        self.show_card("LessonCardView")
        lesson_card.hide_loading()
        lesson_card.show_card_index(0)

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
        self.lesson_plan = None
        self.lesson_index = 0
        self.final_summary = None
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
            justify="center",
            anchor="center",
            font=("Helvetica", 14),
            foreground="#d0d0d0",  # Light gray text on dark background
        )
        desc.grid(row=1, column=0, padx=40, pady=(0, 30))
        self.desc_label = desc

        lang_frame = ttk.Frame(self.content)
        lang_frame.grid(row=2, column=0, padx=40, pady=(0, 20))

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
        self.api_warning_label.grid(row=3, column=0, pady=(10, 10))
        
        # Check API availability and show/hide warning
        self.api_available = is_api_available()
        if self.api_available:
            self.api_warning_label.grid_remove()
        
        self.start_button = ttk.Button(
            self.content,
            text="Start Session" if self.api_available else "API Key Required",
            command=self._on_start_clicked,
            state="normal" if self.api_available else "disabled",
        )
        self.start_button.grid(row=4, column=0, pady=(20, 50))
        
        # Style the button to be more prominent
        style = ttk.Style()
        style.configure("Start.TButton", font=("Helvetica", 14, "bold"), padding=10)
        self.start_button.configure(style="Start.TButton")

        # Responsive wrapping - bind to scrollable canvas for width changes
        self.scrollable.canvas.bind("<Configure>", self._on_content_resize)

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
        self.language_var.set("")

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
            text="Stage 1 of 3",
            font=("Helvetica", 14),
            anchor="center",
            foreground="#7bb3ff",  # Light blue text on dark background
        )
        self.stage_label.grid(row=1, column=0, pady=(0, 10))

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
            self.audio_player = AudioPlayer(self.content)
        
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
                on_recording_complete=self._on_recording_complete
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
        """Set summary from final summary dict."""
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
    logger.separator("Application Starting")
    logger.info("Creating main application window...")
    app = LanguageBuddyApp()
    logger.success("Application window created, entering main loop")
    logger.info("Ready for user interaction")
    app.mainloop()
    logger.separator("Application Closed")

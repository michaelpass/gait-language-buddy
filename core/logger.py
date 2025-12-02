"""
Centralized logging configuration for GAIT Language Buddy.

Provides consistent, color-coded debug output for:
- API calls and responses
- Environment/configuration status
- UI state transitions
- Background task status
- Errors and warnings

Usage:
    from core.logger import logger
    
    logger.api("Making OpenAI chat completion call...")
    logger.success("API key loaded successfully")
    logger.error("Failed to generate image", exc_info=True)
"""

import logging
import sys
from datetime import datetime
from typing import Optional

# ---------------------------------------------------------------------------
# Force UTF-8 output on Windows (Python 3.7+)
# Prevents UnicodeEncodeError when printing Unicode characters (✓, ✗, ⠋, etc.)
# ---------------------------------------------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


class ColorCodes:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    # Colors
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    
    # Bright colors
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"


class DebugLogger:
    """
    Custom debug logger with categorized, color-coded output.
    
    Categories:
    - ENV: Environment/configuration (dotenv, API keys)
    - API: OpenAI API calls
    - IMG: Image generation
    - UI: User interface events
    - TASK: Background tasks/threads
    - OK: Success messages
    - WARN: Warnings
    - ERR: Errors
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._start_time = datetime.now()
    
    def _timestamp(self) -> str:
        """Get formatted timestamp with elapsed time."""
        now = datetime.now()
        elapsed = (now - self._start_time).total_seconds()
        return f"{now.strftime('%H:%M:%S')}.{now.microsecond // 1000:03d} (+{elapsed:>6.1f}s)"
    
    def _log(self, category: str, color: str, message: str, **kwargs) -> None:
        """Internal logging method."""
        if not self.enabled:
            return
        
        timestamp = self._timestamp()
        prefix = f"{ColorCodes.DIM}{timestamp}{ColorCodes.RESET}"
        tag = f"{color}{ColorCodes.BOLD}[{category:>4}]{ColorCodes.RESET}"
        
        # Handle multi-line messages
        lines = message.split('\n')
        for i, line in enumerate(lines):
            if i == 0:
                print(f"{prefix} {tag} {line}", file=sys.stdout, flush=True)
            else:
                # Indent continuation lines
                padding = " " * (len(timestamp) + 8)
                print(f"{ColorCodes.DIM}{padding}{ColorCodes.RESET}{line}", file=sys.stdout, flush=True)
        
        # Print exception info if provided
        if kwargs.get('exc_info'):
            import traceback
            tb = traceback.format_exc()
            for line in tb.split('\n'):
                if line.strip():
                    padding = " " * (len(timestamp) + 8)
                    print(f"{ColorCodes.DIM}{padding}{ColorCodes.RED}{line}{ColorCodes.RESET}", 
                          file=sys.stderr, flush=True)
    
    # === Environment/Configuration ===
    def env(self, message: str, **kwargs) -> None:
        """Log environment/configuration messages (dotenv, API keys, etc.)."""
        self._log("ENV", ColorCodes.MAGENTA, message, **kwargs)
    
    def env_success(self, message: str, **kwargs) -> None:
        """Log successful environment setup."""
        self._log("ENV", ColorCodes.GREEN, f"✓ {message}", **kwargs)
    
    def env_error(self, message: str, **kwargs) -> None:
        """Log environment setup errors."""
        self._log("ENV", ColorCodes.RED, f"✗ {message}", **kwargs)
    
    # === API Calls ===
    def api(self, message: str, **kwargs) -> None:
        """Log API-related messages."""
        self._log("API", ColorCodes.CYAN, message, **kwargs)
    
    def api_call(self, endpoint: str, model: Optional[str] = None, **kwargs) -> None:
        """Log an API call being made."""
        model_info = f" (model: {model})" if model else ""
        self._log("API", ColorCodes.CYAN, f"→ Calling {endpoint}{model_info}", **kwargs)
    
    def api_response(self, endpoint: str, duration_ms: Optional[float] = None, **kwargs) -> None:
        """Log an API response received."""
        duration_info = f" ({duration_ms:.0f}ms)" if duration_ms else ""
        self._log("API", ColorCodes.BRIGHT_CYAN, f"← Response from {endpoint}{duration_info}", **kwargs)
    
    def api_error(self, message: str, **kwargs) -> None:
        """Log API errors."""
        self._log("API", ColorCodes.BRIGHT_RED, f"✗ {message}", **kwargs)
    
    # === Image Generation ===
    def img(self, message: str, **kwargs) -> None:
        """Log image generation messages."""
        self._log("IMG", ColorCodes.YELLOW, message, **kwargs)
    
    def img_start(self, prompt: str, **kwargs) -> None:
        """Log start of image generation."""
        # Truncate long prompts
        display_prompt = prompt[:60] + "..." if len(prompt) > 60 else prompt
        self._log("IMG", ColorCodes.YELLOW, f"→ Generating: \"{display_prompt}\"", **kwargs)
    
    def img_complete(self, path: str, duration_ms: Optional[float] = None, **kwargs) -> None:
        """Log completed image generation."""
        duration_info = f" ({duration_ms:.0f}ms)" if duration_ms else ""
        self._log("IMG", ColorCodes.BRIGHT_GREEN, f"✓ Saved to: {path}{duration_info}", **kwargs)
    
    def img_error(self, message: str, **kwargs) -> None:
        """Log image generation errors."""
        self._log("IMG", ColorCodes.BRIGHT_RED, f"✗ {message}", **kwargs)
    
    # === UI Events ===
    def ui(self, message: str, **kwargs) -> None:
        """Log UI state changes and events."""
        self._log("UI", ColorCodes.BLUE, message, **kwargs)
    
    def ui_transition(self, from_state: str, to_state: str, **kwargs) -> None:
        """Log UI state transitions."""
        self._log("UI", ColorCodes.BRIGHT_BLUE, f"{from_state} → {to_state}", **kwargs)
    
    # === Background Tasks ===
    def task(self, message: str, **kwargs) -> None:
        """Log background task messages."""
        self._log("TASK", ColorCodes.WHITE, message, **kwargs)
    
    def task_start(self, task_name: str, **kwargs) -> None:
        """Log task start."""
        self._log("TASK", ColorCodes.WHITE, f"⚡ Starting: {task_name}", **kwargs)
    
    def task_complete(self, task_name: str, duration_ms: Optional[float] = None, **kwargs) -> None:
        """Log task completion."""
        duration_info = f" ({duration_ms:.0f}ms)" if duration_ms else ""
        self._log("TASK", ColorCodes.BRIGHT_GREEN, f"✓ Completed: {task_name}{duration_info}", **kwargs)
    
    def task_error(self, task_name: str, error: str, **kwargs) -> None:
        """Log task error."""
        self._log("TASK", ColorCodes.BRIGHT_RED, f"✗ Failed: {task_name} - {error}", **kwargs)
    
    # === General Status ===
    def success(self, message: str, **kwargs) -> None:
        """Log success messages."""
        self._log("OK", ColorCodes.BRIGHT_GREEN, f"✓ {message}", **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warnings."""
        self._log("WARN", ColorCodes.BRIGHT_YELLOW, f"⚠ {message}", **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log errors."""
        self._log("ERR", ColorCodes.BRIGHT_RED, f"✗ {message}", **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log general info messages."""
        self._log("INFO", ColorCodes.WHITE, message, **kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug details."""
        self._log("DBG", ColorCodes.DIM, message, **kwargs)
    
    # === Separators/Formatting ===
    def separator(self, title: Optional[str] = None) -> None:
        """Print a visual separator."""
        if not self.enabled:
            return
        
        if title:
            line = f"{'─' * 20} {title} {'─' * 20}"
        else:
            line = "─" * 60
        print(f"\n{ColorCodes.DIM}{line}{ColorCodes.RESET}\n", file=sys.stdout, flush=True)
    
    def banner(self, text: str) -> None:
        """Print a banner message."""
        if not self.enabled:
            return
        
        width = max(60, len(text) + 4)
        border = "═" * width
        padding = " " * ((width - len(text)) // 2)
        
        print(f"\n{ColorCodes.BRIGHT_CYAN}{border}{ColorCodes.RESET}", file=sys.stdout, flush=True)
        print(f"{ColorCodes.BRIGHT_CYAN}║{padding}{ColorCodes.BOLD}{text}{ColorCodes.RESET}{ColorCodes.BRIGHT_CYAN}{padding}║{ColorCodes.RESET}", file=sys.stdout, flush=True)
        print(f"{ColorCodes.BRIGHT_CYAN}{border}{ColorCodes.RESET}\n", file=sys.stdout, flush=True)


# Global logger instance
logger = DebugLogger(enabled=True)


# Utility function for timing
class Timer:
    """Context manager for timing operations."""
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.duration_ms: float = 0
    
    def __enter__(self) -> 'Timer':
        import time
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args) -> None:
        import time
        if self.start_time:
            self.duration_ms = (time.perf_counter() - self.start_time) * 1000


"""
Firebase Firestore database integration for Language Buddy.

This module handles all persistence of user learning data, including:
- User profiles and settings
- Vocabulary tracking with strength ratings
- Grammar pattern progress
- Session history and card responses
- Learning analytics for LLM context

Schema designed for:
- Single user prototype (default user)
- Easy migration to multi-user with Firebase Auth
- JSON serialization for LLM prompt context
"""

import os
import json
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum

from core.logger import logger

# Firebase imports - optional for graceful degradation
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    logger.warning("firebase-admin not installed. Database features disabled.")
    logger.warning("Install with: pip install firebase-admin")


# ---------------------------------------------------------------------------
# Enums and Constants
# ---------------------------------------------------------------------------

class ProficiencyLevel(str, Enum):
    """CEFR proficiency levels."""
    A1 = "A1"  # Beginner
    A2 = "A2"  # Elementary
    B1 = "B1"  # Intermediate
    B2 = "B2"  # Upper Intermediate
    C1 = "C1"  # Advanced
    C2 = "C2"  # Proficient

    @classmethod
    def from_string(cls, s: str) -> "ProficiencyLevel":
        try:
            return cls(s.upper())
        except ValueError:
            return cls.A1


class StrengthRating(str, Enum):
    """Vocabulary/grammar strength categories."""
    NEW = "new"           # Just encountered
    LEARNING = "learning" # Seen a few times, still making mistakes
    FAMILIAR = "familiar" # Getting most right
    STRONG = "strong"     # Consistently correct
    MASTERED = "mastered" # Long-term retention proven


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class VocabularyItem:
    """
    A single vocabulary word being tracked.
    Strength is calculated from encounter history.
    Includes spaced repetition scheduling.
    """
    word: str                          # Word in target language
    translation: str                   # English translation
    language: str                      # Target language (e.g., "Spanish")
    
    # Tracking metrics
    times_seen: int = 0
    times_correct: int = 0
    times_incorrect: int = 0
    
    # Strength (0-100 score + category)
    strength_score: int = 0            # 0-100
    strength_rating: str = "new"       # StrengthRating value
    
    # Temporal data
    first_seen: Optional[str] = None   # ISO timestamp
    last_seen: Optional[str] = None    # ISO timestamp
    last_correct: Optional[str] = None # ISO timestamp
    last_incorrect: Optional[str] = None
    
    # Spaced Repetition fields
    next_review: Optional[str] = None  # ISO date when next review is due
    review_interval_days: int = 1      # Current interval between reviews
    ease_factor: float = 2.5           # Multiplier for interval (SM-2 algorithm)
    consecutive_correct: int = 0       # Consecutive correct answers (resets on error)
    
    # Context and examples
    example_sentences: List[str] = field(default_factory=list)
    related_words: List[str] = field(default_factory=list)
    notes: str = ""                    # User or system notes
    
    # Card types where this word appeared
    card_types_seen: List[str] = field(default_factory=list)
    
    def calculate_strength(self) -> None:
        """Calculate strength score and rating from metrics."""
        if self.times_seen == 0:
            self.strength_score = 0
            self.strength_rating = StrengthRating.NEW.value
            return
        
        # Base score from accuracy
        accuracy = self.times_correct / self.times_seen if self.times_seen > 0 else 0
        base_score = int(accuracy * 100)
        
        # Bonus for repeated exposure
        exposure_bonus = min(20, self.times_seen * 2)
        
        # Recency factor (decay if not seen recently)
        recency_penalty = 0
        if self.last_seen:
            try:
                last = datetime.fromisoformat(self.last_seen.replace('Z', '+00:00'))
                days_since = (datetime.now(timezone.utc) - last).days
                if days_since > 30:
                    recency_penalty = min(30, (days_since - 30) // 7 * 5)
            except Exception:
                pass
        
        self.strength_score = max(0, min(100, base_score + exposure_bonus - recency_penalty))
        
        # Determine rating category
        if self.strength_score >= 90 and self.times_seen >= 10:
            self.strength_rating = StrengthRating.MASTERED.value
        elif self.strength_score >= 75:
            self.strength_rating = StrengthRating.STRONG.value
        elif self.strength_score >= 50:
            self.strength_rating = StrengthRating.FAMILIAR.value
        elif self.times_seen >= 2:
            self.strength_rating = StrengthRating.LEARNING.value
        else:
            self.strength_rating = StrengthRating.NEW.value
    
    def record_encounter(self, correct: bool, card_type: str = "", example: str = "") -> None:
        """Record a new encounter with this word and update spaced repetition schedule."""
        now = datetime.now(timezone.utc)
        now_str = now.isoformat()
        
        self.times_seen += 1
        if correct:
            self.times_correct += 1
            self.last_correct = now_str
            self.consecutive_correct += 1
            
            # Spaced Repetition: Increase interval on correct answer (SM-2 inspired)
            if self.consecutive_correct == 1:
                self.review_interval_days = 1
            elif self.consecutive_correct == 2:
                self.review_interval_days = 3
            else:
                # Apply ease factor for subsequent reviews
                self.review_interval_days = int(self.review_interval_days * self.ease_factor)
                self.review_interval_days = min(self.review_interval_days, 180)  # Cap at 6 months
            
            # Increase ease factor slightly for correct answers
            self.ease_factor = min(3.0, self.ease_factor + 0.1)
        else:
            self.times_incorrect += 1
            self.last_incorrect = now_str
            self.consecutive_correct = 0  # Reset streak
            
            # Spaced Repetition: Reset interval on incorrect answer
            self.review_interval_days = 1
            # Decrease ease factor on errors
            self.ease_factor = max(1.3, self.ease_factor - 0.2)
        
        # Calculate next review date
        next_review_date = now + timedelta(days=self.review_interval_days)
        self.next_review = next_review_date.date().isoformat()
        
        self.last_seen = now_str
        if not self.first_seen:
            self.first_seen = now_str
        
        if card_type and card_type not in self.card_types_seen:
            self.card_types_seen.append(card_type)
        
        if example and example not in self.example_sentences:
            self.example_sentences.append(example)
            # Keep only last 5 examples
            self.example_sentences = self.example_sentences[-5:]
        
        self.calculate_strength()
    
    def is_due_for_review(self) -> bool:
        """Check if this word is due for review based on spaced repetition."""
        if not self.next_review:
            return True  # Never reviewed, always due
        
        try:
            next_date = datetime.fromisoformat(self.next_review).date()
            return datetime.now(timezone.utc).date() >= next_date
        except Exception:
            return True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VocabularyItem":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class GrammarPattern:
    """
    A grammar concept being tracked (e.g., "present tense conjugation", "ser vs estar").
    """
    pattern_id: str                    # Unique identifier (hashed from name)
    name: str                          # Human-readable name
    language: str                      # Target language
    description: str = ""              # What this pattern covers
    
    # Tracking
    times_practiced: int = 0
    times_correct: int = 0
    times_incorrect: int = 0
    
    # Strength
    strength_score: int = 0
    strength_rating: str = "new"
    
    # Examples encountered
    examples: List[str] = field(default_factory=list)
    
    # Temporal
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    
    def calculate_strength(self) -> None:
        """Calculate strength from practice history."""
        if self.times_practiced == 0:
            self.strength_score = 0
            self.strength_rating = StrengthRating.NEW.value
            return
        
        accuracy = self.times_correct / self.times_practiced
        self.strength_score = int(accuracy * 100)
        
        if self.strength_score >= 90 and self.times_practiced >= 5:
            self.strength_rating = StrengthRating.MASTERED.value
        elif self.strength_score >= 75:
            self.strength_rating = StrengthRating.STRONG.value
        elif self.strength_score >= 50:
            self.strength_rating = StrengthRating.FAMILIAR.value
        else:
            self.strength_rating = StrengthRating.LEARNING.value
    
    def record_practice(self, correct: bool, example: str = "") -> None:
        """Record practice of this grammar pattern."""
        now = datetime.now(timezone.utc).isoformat()
        
        self.times_practiced += 1
        if correct:
            self.times_correct += 1
        else:
            self.times_incorrect += 1
        
        self.last_seen = now
        if not self.first_seen:
            self.first_seen = now
        
        if example and example not in self.examples:
            self.examples.append(example)
            self.examples = self.examples[-10:]
        
        self.calculate_strength()
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GrammarPattern":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ErrorPattern:
    """
    Tracks specific error patterns the learner makes repeatedly.
    Used to generate targeted practice and feedback.
    """
    error_id: str                      # Unique identifier
    language: str                      # Target language
    error_type: str                    # Category: "gender", "verb_conjugation", "word_order", "spelling", etc.
    description: str = ""              # Human-readable description
    
    # Specific details
    incorrect_form: str = ""           # What the user wrote
    correct_form: str = ""             # What it should have been
    context: str = ""                  # Sentence/question context
    
    # Tracking
    occurrence_count: int = 1          # How many times this error occurred
    first_seen: Optional[str] = None   # ISO timestamp
    last_seen: Optional[str] = None    # ISO timestamp
    
    # Examples of this error
    examples: List[Dict[str, str]] = field(default_factory=list)  # [{"incorrect": "...", "correct": "...", "context": "..."}]
    
    # Has user shown improvement?
    times_corrected: int = 0           # Times user got it right after making this error
    is_resolved: bool = False          # True if user consistently gets it right now
    
    def record_occurrence(self, incorrect: str, correct: str, context: str = "") -> None:
        """Record another occurrence of this error."""
        now = datetime.now(timezone.utc).isoformat()
        
        self.occurrence_count += 1
        self.last_seen = now
        if not self.first_seen:
            self.first_seen = now
        
        self.incorrect_form = incorrect
        self.correct_form = correct
        if context:
            self.context = context
        
        # Add to examples (keep last 5)
        self.examples.append({
            "incorrect": incorrect,
            "correct": correct,
            "context": context,
            "timestamp": now
        })
        self.examples = self.examples[-5:]
    
    def record_correction(self) -> None:
        """Record that user got this pattern correct."""
        self.times_corrected += 1
        # Mark as resolved if corrected 3+ times consecutively
        if self.times_corrected >= 3:
            self.is_resolved = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ErrorPattern":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SessionRecord:
    """
    Record of a single learning session.
    """
    session_id: str
    started_at: str                    # ISO timestamp
    ended_at: Optional[str] = None
    
    language: str = ""
    session_type: str = "lesson"       # "assessment" or "lesson"
    
    # Proficiency at session start
    proficiency_at_start: str = "A1"
    proficiency_at_end: str = "A1"
    
    # Performance
    cards_completed: int = 0
    cards_correct: int = 0
    total_score: int = 0
    
    # Vocabulary encountered this session
    vocabulary_practiced: List[str] = field(default_factory=list)
    new_vocabulary: List[str] = field(default_factory=list)
    
    # Duration
    duration_minutes: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionRecord":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class LanguageProfile:
    """
    User's learning profile for a single target language.
    """
    language: str                      # e.g., "Spanish"
    
    # Current levels
    overall_proficiency: str = "A1"
    vocabulary_level: str = "A1"
    grammar_level: str = "A1"
    listening_level: str = "A1"
    speaking_level: str = "A1"
    fluency_score: int = 0             # 0-100
    
    # Assessment data
    last_assessment_date: Optional[str] = None
    assessment_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Qualitative feedback
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Learning goals (user-set or system-suggested)
    current_goals: List[str] = field(default_factory=list)
    completed_goals: List[str] = field(default_factory=list)
    
    # Statistics
    total_sessions: int = 0
    total_cards_completed: int = 0
    total_vocabulary_learned: int = 0
    total_time_minutes: int = 0
    
    # Streak tracking
    current_streak_days: int = 0
    longest_streak_days: int = 0
    last_practice_date: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LanguageProfile":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class UserProfile:
    """
    Complete user profile including all language learning data.
    This is the root document stored in Firestore.
    """
    user_id: str                       # Unique user ID
    display_name: str = "Language Learner"
    email: Optional[str] = None        # For future auth
    
    # Account timestamps
    created_at: str = ""
    last_active: str = ""
    
    # Preferences
    preferred_voice: str = "alloy"     # TTS voice preference
    session_length_preference: int = 10  # Preferred cards per session
    
    # Language profiles (keyed by language name)
    # Stored separately in Firestore, referenced here
    active_languages: List[str] = field(default_factory=list)
    primary_language: str = ""         # Current focus language
    
    # App settings
    settings: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Database Client
# ---------------------------------------------------------------------------

class DatabaseClient:
    """
    Firebase Firestore client for Language Buddy.
    
    Collection structure:
    - users/{user_id}                    -> UserProfile
    - users/{user_id}/languages/{lang}   -> LanguageProfile
    - users/{user_id}/vocabulary/{word_hash} -> VocabularyItem
    - users/{user_id}/grammar/{pattern_id}   -> GrammarPattern
    - users/{user_id}/sessions/{session_id}  -> SessionRecord
    """
    
    DEFAULT_USER_ID = "default_user"
    
    def __init__(self):
        self.db = None
        self._initialized = False
        self._user_id = self.DEFAULT_USER_ID
        
        # Local cache for offline/fallback
        self._cache: Dict[str, Any] = {}
    
    def initialize(self, credentials_path: Optional[str] = None) -> bool:
        """
        Initialize Firebase connection.
        
        Args:
            credentials_path: Path to Firebase service account JSON.
                            If None, uses FIREBASE_CREDENTIALS_PATH env var.
        
        Returns:
            True if initialized successfully, False otherwise.
        """
        logger.separator("Database Initialization")
        
        if not FIREBASE_AVAILABLE:
            logger.error("[DB] Firebase package not installed")
            logger.error("[DB] Install with: pip install firebase-admin")
            return False
        
        logger.debug("[DB] firebase-admin package available")
        
        if self._initialized:
            logger.debug("[DB] Already initialized, skipping")
            return True
        
        try:
            # Get credentials path
            creds_path = credentials_path or os.getenv("FIREBASE_CREDENTIALS_PATH")
            logger.debug(f"[DB] Credentials path from env: {creds_path}")
            
            if not creds_path:
                logger.warning("[DB] FIREBASE_CREDENTIALS_PATH not set in .env file")
                logger.warning("[DB] Add: FIREBASE_CREDENTIALS_PATH=./your-credentials.json")
                return False
            
            if not os.path.exists(creds_path):
                logger.error(f"[DB] Credentials file not found at: {creds_path}")
                logger.error("[DB] Check the path in your .env file")
                return False
            
            logger.debug(f"[DB] Found credentials file: {creds_path}")
            
            # Initialize Firebase
            logger.debug("[DB] Loading Firebase credentials...")
            cred = credentials.Certificate(creds_path)
            
            logger.debug("[DB] Initializing Firebase app...")
            firebase_admin.initialize_app(cred)
            
            logger.debug("[DB] Creating Firestore client...")
            self.db = firestore.client()
            self._initialized = True
            
            logger.success("[DB] Firebase Firestore connected successfully!")
            return True
            
        except Exception as e:
            logger.error(f"[DB] Failed to initialize Firebase: {e}")
            import traceback
            logger.debug(f"[DB] Traceback: {traceback.format_exc()}")
            return False
    
    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self._initialized and self.db is not None
    
    # ---------------------------------------------------------------------------
    # User Profile Operations
    # ---------------------------------------------------------------------------
    
    def get_or_create_user(self, user_id: Optional[str] = None) -> Optional[UserProfile]:
        """Get existing user or create default user."""
        uid = user_id or self._user_id
        
        if not self.is_connected():
            # Return cached or new profile
            if uid in self._cache:
                return UserProfile.from_dict(self._cache[uid])
            return self._create_default_user(uid)
        
        try:
            doc_ref = self.db.collection("users").document(uid)
            doc = doc_ref.get()
            
            if doc.exists:
                logger.debug(f"Loaded user profile: {uid}")
                return UserProfile.from_dict(doc.to_dict())
            else:
                # Create new user
                user = self._create_default_user(uid)
                doc_ref.set(user.to_dict())
                logger.success(f"Created new user profile: {uid}")
                return user
                
        except Exception as e:
            logger.error(f"Error getting user profile: {e}")
            return self._create_default_user(uid)
    
    def _create_default_user(self, user_id: str) -> UserProfile:
        """Create a default user profile."""
        now = datetime.now(timezone.utc).isoformat()
        return UserProfile(
            user_id=user_id,
            display_name="Language Learner",
            created_at=now,
            last_active=now,
        )
    
    def update_user(self, user: UserProfile) -> bool:
        """Update user profile."""
        user.last_active = datetime.now(timezone.utc).isoformat()
        
        if not self.is_connected():
            self._cache[user.user_id] = user.to_dict()
            return True
        
        try:
            self.db.collection("users").document(user.user_id).set(user.to_dict())
            return True
        except Exception as e:
            logger.error(f"Error updating user: {e}")
            return False
    
    # ---------------------------------------------------------------------------
    # Language Profile Operations
    # ---------------------------------------------------------------------------
    
    def get_language_profile(
        self, 
        language: str, 
        user_id: Optional[str] = None
    ) -> Optional[LanguageProfile]:
        """Get language profile for a user."""
        uid = user_id or self._user_id
        
        if not self.is_connected():
            cache_key = f"{uid}_lang_{language}"
            if cache_key in self._cache:
                return LanguageProfile.from_dict(self._cache[cache_key])
            return LanguageProfile(language=language)
        
        try:
            doc = self.db.collection("users").document(uid)\
                        .collection("languages").document(language).get()
            
            if doc.exists:
                return LanguageProfile.from_dict(doc.to_dict())
            return LanguageProfile(language=language)
            
        except Exception as e:
            logger.error(f"Error getting language profile: {e}")
            return LanguageProfile(language=language)
    
    def update_language_profile(
        self,
        profile: LanguageProfile,
        user_id: Optional[str] = None
    ) -> bool:
        """Update language profile."""
        uid = user_id or self._user_id
        
        if not self.is_connected():
            cache_key = f"{uid}_lang_{profile.language}"
            self._cache[cache_key] = profile.to_dict()
            return True
        
        try:
            self.db.collection("users").document(uid)\
                   .collection("languages").document(profile.language)\
                   .set(profile.to_dict())
            return True
        except Exception as e:
            logger.error(f"Error updating language profile: {e}")
            return False
    
    # ---------------------------------------------------------------------------
    # Vocabulary Operations
    # ---------------------------------------------------------------------------
    
    def _word_hash(self, word: str, language: str) -> str:
        """Generate consistent hash for a word."""
        key = f"{language}:{word.lower().strip()}"
        return hashlib.md5(key.encode()).hexdigest()[:16]
    
    def get_vocabulary_item(
        self,
        word: str,
        language: str,
        user_id: Optional[str] = None
    ) -> Optional[VocabularyItem]:
        """Get a specific vocabulary item."""
        uid = user_id or self._user_id
        word_id = self._word_hash(word, language)
        
        if not self.is_connected():
            cache_key = f"{uid}_vocab_{word_id}"
            if cache_key in self._cache:
                return VocabularyItem.from_dict(self._cache[cache_key])
            return None
        
        try:
            doc = self.db.collection("users").document(uid)\
                        .collection("vocabulary").document(word_id).get()
            
            if doc.exists:
                return VocabularyItem.from_dict(doc.to_dict())
            return None
            
        except Exception as e:
            logger.error(f"Error getting vocabulary item: {e}")
            return None
    
    def save_vocabulary_item(
        self,
        item: VocabularyItem,
        user_id: Optional[str] = None
    ) -> bool:
        """Save or update a vocabulary item."""
        uid = user_id or self._user_id
        word_id = self._word_hash(item.word, item.language)
        
        # Recalculate strength
        item.calculate_strength()
        
        if not self.is_connected():
            cache_key = f"{uid}_vocab_{word_id}"
            self._cache[cache_key] = item.to_dict()
            return True
        
        try:
            self.db.collection("users").document(uid)\
                   .collection("vocabulary").document(word_id)\
                   .set(item.to_dict())
            return True
        except Exception as e:
            logger.error(f"Error saving vocabulary item: {e}")
            return False
    
    def get_all_vocabulary(
        self,
        language: str,
        user_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[VocabularyItem]:
        """Get all vocabulary for a language. If limit is None, returns ALL words."""
        uid = user_id or self._user_id
        
        if not self.is_connected():
            items = []
            for key, value in self._cache.items():
                if key.startswith(f"{uid}_vocab_"):
                    item = VocabularyItem.from_dict(value)
                    if item.language == language:
                        items.append(item)
            return items
        
        try:
            query = self.db.collection("users").document(uid)\
                         .collection("vocabulary")\
                         .where("language", "==", language)
            
            # Only apply limit if specified
            if limit is not None:
                query = query.limit(limit)
            
            docs = query.stream()
            return [VocabularyItem.from_dict(doc.to_dict()) for doc in docs]
            
        except Exception as e:
            logger.error(f"Error getting vocabulary: {e}")
            return []
    
    def get_weak_vocabulary(
        self,
        language: str,
        user_id: Optional[str] = None,
        max_strength: int = 50,
        limit: int = 20
    ) -> List[VocabularyItem]:
        """Get vocabulary items that need more practice."""
        uid = user_id or self._user_id
        
        all_vocab = self.get_all_vocabulary(language, uid)
        weak = [v for v in all_vocab if v.strength_score <= max_strength]
        weak.sort(key=lambda v: v.strength_score)
        return weak[:limit]
    
    def get_vocabulary_due_for_review(
        self,
        language: str,
        user_id: Optional[str] = None,
        limit: int = 20
    ) -> List[VocabularyItem]:
        """
        Get vocabulary items due for review based on spaced repetition scheduling.
        Returns words where next_review date is today or earlier.
        """
        uid = user_id or self._user_id
        
        all_vocab = self.get_all_vocabulary(language, uid)
        today = datetime.now(timezone.utc).date()
        
        due_words = []
        for vocab in all_vocab:
            if vocab.is_due_for_review():
                due_words.append(vocab)
        
        # Sort by priority: oldest due first, then by strength (weaker first)
        def review_priority(v: VocabularyItem) -> tuple:
            try:
                if v.next_review:
                    days_overdue = (today - datetime.fromisoformat(v.next_review).date()).days
                else:
                    days_overdue = 999  # Never reviewed, high priority
            except Exception:
                days_overdue = 999
            return (-days_overdue, v.strength_score)  # Negative for descending order
        
        due_words.sort(key=review_priority)
        logger.debug(f"[DB] Found {len(due_words)} words due for review in {language}")
        return due_words[:limit]
    
    # ---------------------------------------------------------------------------
    # Error Pattern Operations
    # ---------------------------------------------------------------------------
    
    def _error_hash(self, error_type: str, incorrect: str, language: str) -> str:
        """Generate a unique hash for an error pattern."""
        import hashlib
        key = f"{language}:{error_type}:{incorrect.lower()}"
        return hashlib.md5(key.encode()).hexdigest()[:12]
    
    def get_error_pattern(
        self,
        error_type: str,
        incorrect_form: str,
        language: str,
        user_id: Optional[str] = None
    ) -> Optional[ErrorPattern]:
        """Get a specific error pattern if it exists."""
        uid = user_id or self._user_id
        error_id = self._error_hash(error_type, incorrect_form, language)
        cache_key = f"{uid}_error_{error_id}"
        
        if not self.is_connected():
            return ErrorPattern.from_dict(self._cache[cache_key]) if cache_key in self._cache else None
        
        try:
            doc = self.db.collection("users").document(uid)\
                        .collection("errors").document(error_id).get()
            if doc.exists:
                return ErrorPattern.from_dict(doc.to_dict())
            return None
        except Exception as e:
            logger.error(f"Error getting error pattern: {e}")
            return None
    
    def save_error_pattern(
        self,
        error_type: str,
        incorrect_form: str,
        correct_form: str,
        language: str,
        context: str = "",
        user_id: Optional[str] = None
    ) -> Optional[ErrorPattern]:
        """Save or update an error pattern."""
        uid = user_id or self._user_id
        error_id = self._error_hash(error_type, incorrect_form, language)
        
        # Check if pattern exists
        existing = self.get_error_pattern(error_type, incorrect_form, language, uid)
        
        if existing:
            existing.record_occurrence(incorrect_form, correct_form, context)
            pattern = existing
        else:
            pattern = ErrorPattern(
                error_id=error_id,
                language=language,
                error_type=error_type,
                incorrect_form=incorrect_form,
                correct_form=correct_form,
                context=context,
                first_seen=datetime.now(timezone.utc).isoformat(),
                last_seen=datetime.now(timezone.utc).isoformat(),
            )
        
        cache_key = f"{uid}_error_{error_id}"
        
        if not self.is_connected():
            self._cache[cache_key] = pattern.to_dict()
            return pattern
        
        try:
            self.db.collection("users").document(uid)\
                  .collection("errors").document(error_id).set(pattern.to_dict())
            logger.debug(f"[DB] Saved error pattern: {error_type} - '{incorrect_form}'")
            return pattern
        except Exception as e:
            logger.error(f"Error saving error pattern: {e}")
            return None
    
    def get_frequent_errors(
        self,
        language: str,
        user_id: Optional[str] = None,
        min_occurrences: int = 2,
        limit: int = 10
    ) -> List[ErrorPattern]:
        """Get the most frequent unresolved error patterns."""
        uid = user_id or self._user_id
        
        if not self.is_connected():
            errors = []
            for key, value in self._cache.items():
                if key.startswith(f"{uid}_error_"):
                    pattern = ErrorPattern.from_dict(value)
                    if pattern.language == language and not pattern.is_resolved:
                        if pattern.occurrence_count >= min_occurrences:
                            errors.append(pattern)
            errors.sort(key=lambda e: e.occurrence_count, reverse=True)
            return errors[:limit]
        
        try:
            docs = self.db.collection("users").document(uid)\
                         .collection("errors")\
                         .where("language", "==", language)\
                         .where("is_resolved", "==", False)\
                         .stream()
            
            errors = [ErrorPattern.from_dict(doc.to_dict()) for doc in docs]
            errors = [e for e in errors if e.occurrence_count >= min_occurrences]
            errors.sort(key=lambda e: e.occurrence_count, reverse=True)
            logger.debug(f"[DB] Found {len(errors)} frequent errors in {language}")
            return errors[:limit]
        except Exception as e:
            logger.error(f"Error getting frequent errors: {e}")
            return []
    
    def delete_language_progress(
        self,
        language: str,
        user_id: Optional[str] = None
    ) -> bool:
        """
        Delete all progress for a specific language.
        This removes vocabulary, grammar patterns, sessions, and resets the profile.
        """
        uid = user_id or self._user_id
        logger.warning(f"Deleting all progress for {language} (user: {uid})")
        
        if not self.is_connected():
            # Clear from cache
            keys_to_delete = [
                key for key in self._cache.keys()
                if f"_lang_{language}" in key or 
                   (f"_vocab_" in key and self._cache[key].get("language") == language) or
                   (f"_grammar_" in key and self._cache[key].get("language") == language) or
                   (f"_session_" in key and self._cache[key].get("language") == language)
            ]
            for key in keys_to_delete:
                del self._cache[key]
            return True
        
        try:
            user_ref = self.db.collection("users").document(uid)
            
            # Delete vocabulary for this language
            vocab_docs = user_ref.collection("vocabulary")\
                                .where("language", "==", language).stream()
            for doc in vocab_docs:
                doc.reference.delete()
            logger.debug(f"Deleted vocabulary for {language}")
            
            # Delete grammar patterns for this language
            grammar_docs = user_ref.collection("grammar")\
                                  .where("language", "==", language).stream()
            for doc in grammar_docs:
                doc.reference.delete()
            logger.debug(f"Deleted grammar patterns for {language}")
            
            # Delete sessions for this language
            session_docs = user_ref.collection("sessions")\
                                  .where("language", "==", language).stream()
            for doc in session_docs:
                doc.reference.delete()
            logger.debug(f"Deleted sessions for {language}")
            
            # Reset language profile
            user_ref.collection("languages").document(language).delete()
            logger.debug(f"Deleted language profile for {language}")
            
            logger.success(f"All progress deleted for {language}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting language progress: {e}")
            return False
    
    # ---------------------------------------------------------------------------
    # Grammar Pattern Operations
    # ---------------------------------------------------------------------------
    
    def get_grammar_pattern(
        self,
        pattern_name: str,
        language: str,
        user_id: Optional[str] = None
    ) -> Optional[GrammarPattern]:
        """Get a grammar pattern."""
        uid = user_id or self._user_id
        pattern_id = self._word_hash(pattern_name, language)
        
        if not self.is_connected():
            cache_key = f"{uid}_grammar_{pattern_id}"
            if cache_key in self._cache:
                return GrammarPattern.from_dict(self._cache[cache_key])
            return None
        
        try:
            doc = self.db.collection("users").document(uid)\
                        .collection("grammar").document(pattern_id).get()
            
            if doc.exists:
                return GrammarPattern.from_dict(doc.to_dict())
            return None
            
        except Exception as e:
            logger.error(f"Error getting grammar pattern: {e}")
            return None
    
    def save_grammar_pattern(
        self,
        pattern: GrammarPattern,
        user_id: Optional[str] = None
    ) -> bool:
        """Save a grammar pattern."""
        uid = user_id or self._user_id
        pattern_id = pattern.pattern_id or self._word_hash(pattern.name, pattern.language)
        pattern.pattern_id = pattern_id
        pattern.calculate_strength()
        
        if not self.is_connected():
            cache_key = f"{uid}_grammar_{pattern_id}"
            self._cache[cache_key] = pattern.to_dict()
            return True
        
        try:
            self.db.collection("users").document(uid)\
                   .collection("grammar").document(pattern_id)\
                   .set(pattern.to_dict())
            return True
        except Exception as e:
            logger.error(f"Error saving grammar pattern: {e}")
            return False
    
    # ---------------------------------------------------------------------------
    # Session Operations
    # ---------------------------------------------------------------------------
    
    def save_session(
        self,
        session: SessionRecord,
        user_id: Optional[str] = None
    ) -> bool:
        """Save a session record and update streak."""
        uid = user_id or self._user_id
        
        if not self.is_connected():
            cache_key = f"{uid}_session_{session.session_id}"
            self._cache[cache_key] = session.to_dict()
            return True
        
        try:
            self.db.collection("users").document(uid)\
                   .collection("sessions").document(session.session_id)\
                   .set(session.to_dict())
            
            # Update streak in language profile
            if session.language:
                self._update_streak(session.language, uid)
            
            return True
        except Exception as e:
            logger.error(f"Error saving session: {e}")
            return False
    
    def _update_streak(
        self,
        language: str,
        user_id: Optional[str] = None
    ) -> None:
        """Calculate and update streak from session history."""
        uid = user_id or self._user_id
        
        try:
            # Get all sessions for this language, ordered by date
            sessions = self.get_recent_sessions(language, uid, limit=365)
            
            if not sessions:
                return
            
            # Calculate streak from session dates
            current_streak, longest_streak = self._calculate_streak_from_sessions(sessions)
            
            # Get and update language profile
            profile = self.get_language_profile(language, uid)
            if profile:
                profile.current_streak_days = current_streak
                # Use the calculated longest_streak directly from session data (authoritative)
                profile.longest_streak_days = longest_streak
                profile.last_practice_date = datetime.now(timezone.utc).date().isoformat()
                self.update_language_profile(profile, uid)
                logger.debug(f"[DB] Streak updated: current={current_streak}, longest={longest_streak}")
        except Exception as e:
            logger.error(f"Error updating streak: {e}")
    
    def _calculate_streak_from_sessions(
        self,
        sessions: List[SessionRecord]
    ) -> tuple:
        """
        Calculate current and longest streak from session records.
        
        Uses the user's local timezone to determine calendar days.
        This ensures that practicing at 11pm and 1am local time on the same
        calendar day counts as one day, not two.
        
        Returns:
            Tuple of (current_streak, longest_streak)
        """
        if not sessions:
            return 0, 0
        
        # Get local timezone for accurate calendar day calculation
        try:
            from datetime import timezone as tz
            import time
            # Get local timezone offset
            if time.daylight and time.localtime().tm_isdst > 0:
                local_offset = timedelta(seconds=-time.altzone)
            else:
                local_offset = timedelta(seconds=-time.timezone)
            local_tz = tz(local_offset)
        except Exception:
            # Fallback to UTC if timezone detection fails
            local_tz = timezone.utc
        
        # Extract unique LOCAL dates from sessions
        session_dates = set()
        for session in sessions:
            try:
                # Parse the ISO timestamp
                if session.started_at:
                    dt = datetime.fromisoformat(session.started_at.replace('Z', '+00:00'))
                    # Convert to local timezone before extracting date
                    local_dt = dt.astimezone(local_tz)
                    session_dates.add(local_dt.date())
            except Exception:
                continue
        
        if not session_dates:
            return 0, 0
        
        # Sort dates in descending order (most recent first)
        sorted_dates = sorted(session_dates, reverse=True)
        
        # Use LOCAL today for comparison
        today = datetime.now(local_tz).date()
        
        # Calculate current streak (consecutive calendar days from today or yesterday)
        current_streak = 0
        expected_date = today
        
        for date in sorted_dates:
            if date == expected_date:
                current_streak += 1
                expected_date = date - timedelta(days=1)
            elif date == expected_date - timedelta(days=1):
                # Allow starting from yesterday if no practice today yet
                if current_streak == 0:
                    current_streak = 1
                    expected_date = date - timedelta(days=1)
                else:
                    break
            elif date < expected_date:
                break
        
        # Calculate longest streak by counting consecutive calendar days
        longest_streak = 0
        streak = 0
        prev_date = None
        
        for date in sorted(session_dates):  # Ascending order
            if prev_date is None:
                streak = 1
            elif (date - prev_date).days == 1:
                # Consecutive day - extend streak
                streak += 1
            elif (date - prev_date).days > 1:
                # Gap in days - record streak and start new one
                longest_streak = max(longest_streak, streak)
                streak = 1
            # If (date - prev_date).days == 0, same day - skip (shouldn't happen with set)
            prev_date = date
        
        # Don't forget the final streak
        longest_streak = max(longest_streak, streak, current_streak)
        
        logger.debug(f"[DB] Streak calculation (local tz): {len(session_dates)} unique days, current={current_streak}, longest={longest_streak}")
        
        return current_streak, longest_streak
    
    def calculate_streaks(
        self,
        language: str,
        user_id: Optional[str] = None,
    ) -> tuple:
        """
        Calculate current and longest streaks from session timestamps.
        
        This is the authoritative method - calculates from actual session data,
        not stored values.
        
        Args:
            language: The language to calculate streaks for
            user_id: Optional user ID (defaults to current user)
            
        Returns:
            Tuple of (current_streak, longest_streak)
        """
        # Get ALL sessions, not just recent ones
        sessions = self.get_recent_sessions(language, user_id, limit=1000)
        return self._calculate_streak_from_sessions(sessions)
    
    def get_recent_sessions(
        self,
        language: str,
        user_id: Optional[str] = None,
        limit: int = 10
    ) -> List[SessionRecord]:
        """Get recent sessions for a language."""
        uid = user_id or self._user_id
        
        if not self.is_connected():
            sessions = []
            for key, value in self._cache.items():
                if key.startswith(f"{uid}_session_"):
                    session = SessionRecord.from_dict(value)
                    if session.language == language:
                        sessions.append(session)
            sessions.sort(key=lambda s: s.started_at, reverse=True)
            return sessions[:limit]
        
        try:
            docs = self.db.collection("users").document(uid)\
                         .collection("sessions")\
                         .where("language", "==", language)\
                         .order_by("started_at", direction=firestore.Query.DESCENDING)\
                         .limit(limit).stream()
            
            return [SessionRecord.from_dict(doc.to_dict()) for doc in docs]
            
        except Exception as e:
            logger.error(f"Error getting sessions: {e}")
            return []
    
    # ---------------------------------------------------------------------------
    # LLM Context Generation
    # ---------------------------------------------------------------------------
    
    def generate_llm_context(
        self,
        language: str,
        user_id: Optional[str] = None,
        max_sessions: int = 5
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive context object for LLM prompts.
        This captures the user's entire learning state for personalized lessons.
        """
        uid = user_id or self._user_id
        
        # Get user and language profile
        user = self.get_or_create_user(uid)
        lang_profile = self.get_language_profile(language, uid)
        
        # Get ALL vocabulary - no limit, track every word the user has learned
        all_vocab = self.get_all_vocabulary(language, uid, limit=None)  # No limit
        weak_vocab = self.get_weak_vocabulary(language, uid, limit=30)
        
        # ALL known words (for avoiding duplicates in new lessons)
        all_known_words = [v.word for v in all_vocab if v.word]
        
        # Get words due for review (spaced repetition)
        due_for_review = self.get_vocabulary_due_for_review(language, uid, limit=20)
        
        # Get frequent error patterns
        frequent_errors = self.get_frequent_errors(language, uid, min_occurrences=2, limit=10)
        
        # Vocabulary summary
        vocab_summary = {
            "total_words_learned": len(all_vocab),
            "mastered_count": len([v for v in all_vocab if v.strength_rating == "mastered"]),
            "strong_count": len([v for v in all_vocab if v.strength_rating == "strong"]),
            "learning_count": len([v for v in all_vocab if v.strength_rating == "learning"]),
            "weak_words_needing_practice": [
                {"word": v.word, "translation": v.translation, "strength": v.strength_score}
                for v in weak_vocab[:10]
            ],
            "due_for_review": [
                {"word": v.word, "translation": v.translation, "days_overdue": v.review_interval_days}
                for v in due_for_review[:10]
            ],
            "all_known_words": all_known_words,  # ALL words learned (for avoiding duplicates)
        }
        
        # Error patterns summary
        error_summary = {
            "frequent_errors": [
                {
                    "error_type": e.error_type,
                    "incorrect": e.incorrect_form,
                    "correct": e.correct_form,
                    "occurrences": e.occurrence_count,
                    "context": e.context
                }
                for e in frequent_errors
            ]
        }
        
        # Recent sessions summary
        sessions = self.get_recent_sessions(language, uid, limit=max_sessions)
        session_summary = {
            "total_sessions": lang_profile.total_sessions if lang_profile else 0,
            "recent_sessions": [
                {
                    "date": s.started_at,
                    "cards_completed": s.cards_completed,
                    "accuracy": s.cards_correct / s.cards_completed if s.cards_completed > 0 else 0,
                    "new_vocabulary": len(s.new_vocabulary),
                }
                for s in sessions
            ],
            "current_streak": lang_profile.current_streak_days if lang_profile else 0,
        }
        
        # Build the context object
        context = {
            "user_id": uid,
            "language": language,
            "proficiency": {
                "overall": lang_profile.overall_proficiency if lang_profile else "A1",
                "vocabulary": lang_profile.vocabulary_level if lang_profile else "A1",
                "grammar": lang_profile.grammar_level if lang_profile else "A1",
                "listening": lang_profile.listening_level if lang_profile else "A1",
                "speaking": lang_profile.speaking_level if lang_profile else "A1",
                "fluency_score": lang_profile.fluency_score if lang_profile else 0,
            },
            "strengths": lang_profile.strengths if lang_profile else [],
            "weaknesses": lang_profile.weaknesses if lang_profile else [],
            "recommendations": lang_profile.recommendations if lang_profile else [],
            "vocabulary": vocab_summary,
            "errors": error_summary,
            "sessions": session_summary,
            "learning_goals": lang_profile.current_goals if lang_profile else [],
        }
        
        return context
    
    def generate_llm_context_string(
        self,
        language: str,
        user_id: Optional[str] = None,
    ) -> str:
        """
        Generate a formatted string of user context for LLM prompts.
        This can be directly appended to system prompts.
        """
        context = self.generate_llm_context(language, user_id)
        
        lines = [
            "=== LEARNER PROFILE ===",
            f"Language: {context['language']}",
            f"Overall Proficiency: {context['proficiency']['overall']}",
            f"Vocabulary Level: {context['proficiency']['vocabulary']}",
            f"Grammar Level: {context['proficiency']['grammar']}",
            f"Fluency Score: {context['proficiency']['fluency_score']}/100",
            "",
            "STRENGTHS:",
        ]
        for s in context['strengths'][:5]:
            lines.append(f"  - {s}")
        
        lines.extend([
            "",
            "AREAS TO IMPROVE:",
        ])
        for w in context['weaknesses'][:5]:
            lines.append(f"  - {w}")
        
        lines.extend([
            "",
            f"VOCABULARY: {context['vocabulary']['total_words_learned']} words learned",
            f"  - Mastered: {context['vocabulary']['mastered_count']}",
            f"  - Strong: {context['vocabulary']['strong_count']}",
            f"  - Learning: {context['vocabulary']['learning_count']}",
            "",
            "WORDS NEEDING PRACTICE (weak vocabulary):",
        ])
        for v in context['vocabulary']['weak_words_needing_practice'][:10]:
            lines.append(f"  - {v['word']} ({v['translation']}) - strength: {v['strength']}/100")
        
        # Add words due for spaced repetition review
        due_for_review = context['vocabulary'].get('due_for_review', [])
        if due_for_review:
            lines.extend([
                "",
                "WORDS DUE FOR REVIEW (spaced repetition - INCLUDE THESE):",
            ])
            for v in due_for_review[:10]:
                lines.append(f"  - {v['word']} ({v['translation']})")
        
        # Add error patterns to target
        frequent_errors = context.get('errors', {}).get('frequent_errors', [])
        if frequent_errors:
            lines.extend([
                "",
                "FREQUENT ERRORS - CREATE EXERCISES TARGETING THESE:",
            ])
            for e in frequent_errors[:5]:
                lines.append(f"  - {e['error_type']}: '{e['incorrect']}' → '{e['correct']}' ({e['occurrences']}x)")
        
        # Add ALL known words to AVOID for new vocabulary
        all_known_words = context['vocabulary'].get('all_known_words', [])
        if all_known_words:
            lines.extend([
                "",
                f"WORDS ALREADY KNOWN - DO NOT TEACH THESE ({len(all_known_words)} words):",
                f"  {', '.join(all_known_words)}"  # Include ALL known words
            ])
        
        lines.extend([
            "",
            f"SESSIONS: {context['sessions']['total_sessions']} total",
            f"Current Streak: {context['sessions']['current_streak']} days",
        ])
        
        # Add recent session performance for holistic evaluation
        recent_sessions = context['sessions'].get('recent_sessions', [])
        if recent_sessions:
            lines.extend([
                "",
                "RECENT SESSION PERFORMANCE (newest first):",
            ])
            for i, session in enumerate(recent_sessions[:5], 1):
                accuracy = session.get('accuracy', 0) * 100
                cards = session.get('cards_completed', 0)
                new_vocab = session.get('new_vocabulary', 0)
                date = session.get('date', 'Unknown')[:10]  # Just the date part
                lines.append(f"  {i}. {date}: {accuracy:.0f}% accuracy, {cards} cards, +{new_vocab} words")
        
        if context['learning_goals']:
            lines.extend([
                "",
                "CURRENT GOALS:",
            ])
            for g in context['learning_goals'][:3]:
                lines.append(f"  - {g}")
        
        lines.append("=== END PROFILE ===")
        
        return "\n".join(lines)
    
    def get_all_known_words(
        self,
        language: str,
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[str]:
        """
        Get a simple list of all words the user has learned in a language.
        Useful for preventing duplicate vocabulary in lesson generation.
        
        Returns:
            List of word strings (in the target language)
        """
        if not self._connected:
            return []
        
        uid = user_id or self._user_id
        all_vocab = self.get_all_vocabulary(language, uid, limit=limit)
        return [v.word for v in all_vocab if v.word]


# ---------------------------------------------------------------------------
# Global Database Instance
# ---------------------------------------------------------------------------

# Singleton database client
db_client = DatabaseClient()


def initialize_database(credentials_path: Optional[str] = None) -> bool:
    """Initialize the global database client."""
    return db_client.initialize(credentials_path)


def get_db() -> DatabaseClient:
    """Get the global database client."""
    return db_client


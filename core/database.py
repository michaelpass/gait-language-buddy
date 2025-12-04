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
        """Record a new encounter with this word."""
        now = datetime.now(timezone.utc).isoformat()
        
        self.times_seen += 1
        if correct:
            self.times_correct += 1
            self.last_correct = now
        else:
            self.times_incorrect += 1
            self.last_incorrect = now
        
        self.last_seen = now
        if not self.first_seen:
            self.first_seen = now
        
        if card_type and card_type not in self.card_types_seen:
            self.card_types_seen.append(card_type)
        
        if example and example not in self.example_sentences:
            self.example_sentences.append(example)
            # Keep only last 5 examples
            self.example_sentences = self.example_sentences[-5:]
        
        self.calculate_strength()
    
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
        limit: int = 500
    ) -> List[VocabularyItem]:
        """Get all vocabulary for a language."""
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
            docs = self.db.collection("users").document(uid)\
                         .collection("vocabulary")\
                         .where("language", "==", language)\
                         .limit(limit).stream()
            
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
                profile.longest_streak_days = max(profile.longest_streak_days, longest_streak)
                profile.last_practice_date = datetime.now(timezone.utc).date().isoformat()
                self.update_language_profile(profile, uid)
                logger.debug(f"[DB] Streak updated: current={current_streak}, longest={profile.longest_streak_days}")
        except Exception as e:
            logger.error(f"Error updating streak: {e}")
    
    def _calculate_streak_from_sessions(
        self,
        sessions: List[SessionRecord]
    ) -> tuple:
        """
        Calculate current and longest streak from session records.
        
        Returns:
            Tuple of (current_streak, longest_streak)
        """
        if not sessions:
            return 0, 0
        
        # Extract unique dates from sessions
        session_dates = set()
        for session in sessions:
            try:
                # Parse the ISO timestamp and extract the date
                if session.started_at:
                    dt = datetime.fromisoformat(session.started_at.replace('Z', '+00:00'))
                    session_dates.add(dt.date())
            except Exception:
                continue
        
        if not session_dates:
            return 0, 0
        
        # Sort dates in descending order (most recent first)
        sorted_dates = sorted(session_dates, reverse=True)
        today = datetime.now(timezone.utc).date()
        
        # Calculate current streak (consecutive days from today or yesterday)
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
        
        # Calculate longest streak
        longest_streak = 0
        streak = 0
        prev_date = None
        
        for date in sorted(session_dates):
            if prev_date is None:
                streak = 1
            elif (date - prev_date).days == 1:
                streak += 1
            elif (date - prev_date).days > 1:
                longest_streak = max(longest_streak, streak)
                streak = 1
            prev_date = date
        
        longest_streak = max(longest_streak, streak, current_streak)
        
        logger.debug(f"[DB] Streak calculation: {len(session_dates)} unique days, current={current_streak}, longest={longest_streak}")
        
        return current_streak, longest_streak
    
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
        max_vocab: int = 100,
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
        
        # Get vocabulary
        all_vocab = self.get_all_vocabulary(language, uid, limit=max_vocab)
        weak_vocab = self.get_weak_vocabulary(language, uid, limit=20)
        strong_vocab = [v for v in all_vocab if v.strength_score >= 80][:20]
        
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
            "strong_words": [v.word for v in strong_vocab[:15]],
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
            "WORDS NEEDING PRACTICE:",
        ])
        for v in context['vocabulary']['weak_words_needing_practice'][:5]:
            lines.append(f"  - {v['word']} ({v['translation']}) - strength: {v['strength']}/100")
        
        lines.extend([
            "",
            f"SESSIONS: {context['sessions']['total_sessions']} total",
            f"Current Streak: {context['sessions']['current_streak']} days",
        ])
        
        if context['learning_goals']:
            lines.extend([
                "",
                "CURRENT GOALS:",
            ])
            for g in context['learning_goals'][:3]:
                lines.append(f"  - {g}")
        
        lines.append("=== END PROFILE ===")
        
        return "\n".join(lines)


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


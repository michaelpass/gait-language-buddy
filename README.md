# GAIT Language Buddy

GAIT Language Buddy is an AI-powered language-learning prototype built for the **Generative AI Tools (GAIT)** course project.

This application provides a comprehensive, multimodal language learning experience:

1. **Assessment** - Determines your proficiency level (A1-C2) through targeted questions
2. **Personalized Lessons** - 12-card lessons tailored to your level with diverse exercise types
3. **Visual Learning** - AI-generated images for vocabulary and context
4. **Audio Practice** - Text-to-speech for listening comprehension and transcription
5. **Speaking Exercises** - Speech-to-text for pronunciation practice
6. **Progress Tracking** - Firebase database stores vocabulary strength, grammar progress, and learning history

---

## ✨ Features

### 🎯 Adaptive Assessment
- 3-stage assessment to determine CEFR proficiency level (A1-C2)
- Evaluates vocabulary, grammar, and fluency
- Identifies strengths and weaknesses

### 📚 Diverse Lesson Types
- **Multiple Choice** - Test comprehension
- **Fill in the Blank** - Grammar practice
- **Image Questions** - Visual vocabulary ("What is this?")
- **Vocabulary Cards** - Word learning with examples
- **Audio Transcription** - Listen and write (TTS via OpenAI)
- **Audio Comprehension** - Listen to passages and answer questions
- **Speaking Exercises** - Record yourself and get STT feedback

### 🔊 Multimodal Learning
- **Image Generation** - DALL-E 3 creates contextual images
- **Text-to-Speech** - Native-quality audio in target language
- **Speech-to-Text** - Whisper transcribes your speech

### 💾 Progress Persistence (Firebase)
- Vocabulary tracking with strength ratings
- Grammar pattern progress
- Session history
- Personalized LLM context for evolving lessons

---

## 🧱 Project Structure

```
gait-language-buddy/
├── .env                  # API keys (create from .env.example)
├── requirements.txt
├── main.py               # Tkinter UI application
└── core/
    ├── __init__.py
    ├── models.py         # Data classes (LessonCard, AssessmentResult, etc.)
    ├── schemas.py        # LLM JSON schemas for card generation
    ├── api.py            # OpenAI API integration
    ├── logger.py         # Debug logging utility
    └── database.py       # Firebase Firestore integration
```

---

## 📦 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/gait-language-buddy.git
cd gait-language-buddy
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
# Required: OpenAI API Key
# Get yours at: https://platform.openai.com/api-keys
OPENAI_API_KEY=sk-your-api-key-here

# Optional: Firebase for progress persistence
# 1. Create project at: https://console.firebase.google.com/
# 2. Go to Project Settings > Service Accounts
# 3. Click "Generate new private key"
# 4. Save the JSON file and set path below
FIREBASE_CREDENTIALS_PATH=/path/to/firebase-credentials.json
```

### 4. Run the application

```bash
python main.py
```

---

## 🔥 Firebase Setup (Optional)

Firebase enables progress tracking across sessions and devices.

### Step 1: Create Firebase Project

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Click "Create a project"
3. Name it (e.g., "language-buddy")
4. Disable Google Analytics (optional for prototype)

### Step 2: Enable Firestore

1. In your project, go to **Build > Firestore Database**
2. Click "Create database"
3. Start in **test mode** for development
4. Choose a region close to you

### Step 3: Generate Service Account Key

1. Go to **Project Settings** (gear icon)
2. Click **Service accounts** tab
3. Click **Generate new private key**
4. Save the JSON file securely
5. Set `FIREBASE_CREDENTIALS_PATH` in your `.env`

### Step 4: Security Rules (Production)

For multi-user production, update Firestore rules:

```javascript
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    match /users/{userId}/{document=**} {
      allow read, write: if request.auth != null && request.auth.uid == userId;
    }
  }
}
```

---

## 🖥 Usage Guide

1. **Select Language** - Choose Spanish, French, German, etc.
2. **Take Assessment** - Answer 3 questions to determine your level
3. **Complete Lessons** - Work through 12 diverse cards
4. **Review Progress** - See your score and vocabulary growth
5. **Continue Learning** - Each session builds on your history

---

## 🧪 Tech Stack

- **Python 3.10+**
- **Tkinter** - Cross-platform UI
- **OpenAI API** - GPT-4o-mini, DALL-E 3, Whisper, TTS
- **Firebase Firestore** - Cloud database
- **pygame** - Audio playback
- **sounddevice/soundfile** - Audio recording

---

## 📊 Database Schema

### User Profile
```json
{
  "user_id": "default_user",
  "display_name": "Language Learner",
  "active_languages": ["Spanish"],
  "primary_language": "Spanish"
}
```

### Language Profile
```json
{
  "language": "Spanish",
  "overall_proficiency": "A2",
  "vocabulary_level": "A2",
  "grammar_level": "A1",
  "fluency_score": 65,
  "strengths": ["Basic greetings", "Present tense"],
  "weaknesses": ["Gender agreement", "Ser vs Estar"],
  "total_sessions": 5,
  "current_streak_days": 3
}
```

### Vocabulary Item
```json
{
  "word": "manzana",
  "translation": "apple",
  "times_seen": 8,
  "times_correct": 6,
  "strength_score": 75,
  "strength_rating": "familiar",
  "example_sentences": ["La manzana es roja."]
}
```

---

## 👥 Contributors

- Michael Pass
- Dani Perez
- Mishka Mohamed Nour

---

## 📄 License

This project is currently for academic use within the GAIT course.

---

## 🙌 Acknowledgments

Thanks to the GAIT course faculty for guidance and inspiration in applied multimodal AI.

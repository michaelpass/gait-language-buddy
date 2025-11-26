# GAIT Language Buddy

GAIT Language Buddy is an AI-powered language-learning prototype built for the **Generative AI Tools (GAIT)** course project.  
This application guides a learner through a multimodal workflow:

1. A scene is generated (currently text-based; planned for image generation).  
2. The learner writes a description in their target language.  
3. The app evaluates the writing using an OpenAI model.  
4. A personalized mini-lesson is generated based on learner strengths/weaknesses.  
5. (Future) The app will produce audio examples and rich multimodal outputs.

The user interface is built using **PySimpleGUI** for rapid prototyping and easy team collaboration.

---

## ✨ Features

### ✔ Scene Generation (Text for Now)
- Provides a simple scene to describe.
- Will later integrate OpenAI image generation.

### ✔ LLM-Powered Writing Evaluation
- Detects grammar/vocabulary issues.
- Infers CEFR-style proficiency (A1–C2).
- Provides strengths, weaknesses, suggestions.

### ✔ LLM-Powered Mini-Lesson
- Tailored feedback based on the learner’s writing.
- Includes example sentences and vocabulary suggestions.

### ✔ Graceful Fallbacks
If the OpenAI API key is missing or a request fails:
- A rule-based evaluator is used.
- A rule-based mini-lesson is generated.

This ensures anyone (e.g., classmates) can run the app without an API key.

---

## 🧱 Project Structure
gait-language-buddy/
├─ .env                  # Stores OPENAI_API_KEY (ignored by git)
├─ .gitignore
├─ README.md
├─ requirements.txt
├─ main.py               # PySimpleGUI app controller
└─ core/
├─ init.py
├─ models.py          # Data classes for structured outputs
└─ api.py             # OpenAI-backed evaluation + mini-lesson logic

---

## 📦 Getting Started

### 1. Clone the repository

```
git clone https://github.com/<your-username>/gait-language-buddy.git
cd gait-language-buddy
```

🖥 Usage Guide
	1.	Choose your target language.
	2.	Click New Scene to load a description prompt.
	3.	Write your paragraph in the target language.
	4.	Click Evaluate Writing to receive:
	•	proficiency level
	•	strengths
	•	weaknesses
	•	suggestions
	•	auto-generated mini-lesson
	5.	Click Generate Audio (stub; real TTS planned for Phase 2).

⸻

🧪 Tech Stack
	•	Python 3.10+
	•	PySimpleGUI for the user interface
	•	OpenAI API for evaluation and mini-lessons
	•	python-dotenv for environment variable management

⸻

🛠 How the Code Works

core/models.py

Contains structured dataclasses:
	•	TextAnalysis
	•	MiniLesson
	•	AudioInfo

core/api.py

Handles:
	•	environment loading (os.getenv, load_dotenv)
	•	OpenAI LLM calls
	•	fallback heuristics when the API isn’t available

main.py

Defines:
	•	the GUI layout
	•	GUI event loop
	•	rendering logic for analysis, lessons, audio

⸻

🌱 Planned Enhancements

Phase 2 (Multimodal)
	•	Real image generation for scene creation
	•	Real text-to-speech audio output
	•	Learner speech input + pronunciation evaluation

Phase 3 (Intelligent Tutoring)
	•	Learner profiles and progress tracking
	•	Dynamic difficulty adjustment
	•	Rubric-based CEFR scoring (A1–C2)
	•	More advanced lesson generation

⸻

👥 Contributors
	•	Michael Pass
  •	Dani Perez
	•	Mishka Mohamed Nour
  

⸻

📄 License

This project is currently for academic use within the GAIT course.
A standard license (MIT/GPL/etc.) may be added later based on team preference.

⸻

🎓 Instructor Notes (Optional Section)

This project is designed as a demonstration of:
	•	multimodal LLM interactions
	•	stateful evaluation over multiple steps
	•	GUI-backed applications using OpenAI APIs
	•	safe development patterns with fallbacks

⸻

🙌 Acknowledgments

Thanks to the GAIT course faculty for guidance and inspiration in applied multimodal AI.

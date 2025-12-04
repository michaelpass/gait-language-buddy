"""
Structured JSON schemas for lesson cards and assessments.

This module defines the JSON structure that the LLM should use when generating
lesson cards. The client application knows how to render each card type.
"""

# Schema documentation for LLM prompts
LESSON_CARD_SCHEMA = """
The lesson card JSON structure supports the following card types:

1. "text_question" - Simple text-based question requiring text input
   {
     "type": "text_question",
     "question": "Question text in target language",
     "instruction": "Optional instruction in English",
     "image_prompt": "Optional: prompt for generating an image",
     "correct_answer": "Expected correct answer",
     "feedback": "Feedback shown after submission",
     "alternatives": ["Alternative correct answer 1", "Alternative 2"]
   }

2. "multiple_choice" - Multiple choice question
   {
     "type": "multiple_choice",
     "question": "REQUIRED: Clear question like 'What is the German word for apple?'",
     "instruction": "Optional instruction in English",
     "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
     "correct_index": 0,  // Index of correct option (0-based)
     "image_prompt": "Optional: prompt for generating an image",
     "feedback": "Feedback shown after submission",
     "vocabulary_expansion": ["Additional vocabulary word 1", "Word 2"]
   }
   🚨 CRITICAL: The "question" field is MANDATORY for multiple_choice cards!
   🚨 Without a question, the user won't know what they're being asked.
   🚨 Good examples:
      - "What is the German word for 'chair'?"
      - "Which word means 'to go' in German?"
      - "Select the correct translation for 'der Hund'"
   🚨 Do NOT include letter prefixes (A, B, C, D) in option text - UI adds these.

3. "image_question" - "What is this?" style question with image
   {
     "type": "image_question",
     "question": "What is this?",
     "instruction": "Optional instruction in English",
     "image_prompt": "REQUIRED: prompt for generating the image",
     "correct_answer": "Correct answer in target language",
     "feedback": "Feedback shown after submission",
     "alternatives": ["Synonym 1", "Synonym 2"],
     "vocabulary_expansion": ["Related word 1", "Related word 2"]
   }

4. "fill_in_blank" - Fill-in-the-blank exercise
   {
     "type": "fill_in_blank",
     "question": "Ich habe drei Spielzeuge. Zusammen habe ich eine ______",
     "instruction": "Optional instruction in English",
     "image_prompt": "Optional: prompt for generating an image",
     "correct_answer": "Expected word or phrase",
     "feedback": "Feedback shown after submission",
     "alternatives": ["Alternative correct answer"],
     "vocabulary_expansion": ["Related vocabulary"]
   }

5. "vocabulary" - Vocabulary learning card
   {
     "type": "vocabulary",
     "word": "Word in target language",
     "translation": "English translation",
     "example": "Example sentence in target language",
     "image_prompt": "Optional: prompt for generating an image",
     "related_words": ["Related word 1", "Related word 2"]
   }

6. "audio_transcription" - Listen and transcribe exercise (1-3 sentences)
   {
     "type": "audio_transcription",
     "instruction": "Listen to the audio and write what you hear in [target language].",
     "audio_text": "The text in target language that will be spoken (1-3 sentences)",
     "correct_answer": "Same as audio_text (for comparison)",
     "alternatives": ["Acceptable alternative transcriptions"],
     "feedback": "Feedback shown after submission",
     "vocabulary_expansion": ["Key vocabulary from the audio"]
   }
   The audio_text will be converted to speech using TTS. Keep it short (1-3 sentences)
   for transcription exercises. Focus on clear, common vocabulary and grammar patterns.

7. "audio_comprehension" - Listen to longer audio and answer questions
   {
     "type": "audio_comprehension",
     "instruction": "Listen to the passage and answer the question below.",
     "audio_text": "A longer passage in target language (paragraph or two)",
     "question": "Question about the audio content (in English)",
     "correct_answer": "Expected answer (can be in target language or English)",
     "alternatives": ["Alternative correct answers"],
     "comprehension_questions": [
       {
         "question": "Additional question about the passage",
         "correct_answer": "Answer",
         "alternatives": ["Alt answer"]
       }
     ],
     "feedback": "Feedback explaining the correct answer",
     "vocabulary_expansion": ["Key vocabulary from the passage"]
   }
   Audio comprehension uses longer passages and tests understanding through questions.
   Questions should be in English, but answers may be in the target language.

8. "speaking" - Speaking exercise where user records themselves saying a phrase
   {
     "type": "speaking",
     "instruction": "Say the following phrase out loud in [target language]:",
     "speaking_prompt": "The phrase/sentence the user should say (in target language)",
     "correct_answer": "Same as speaking_prompt (for comparison with transcription)",
     "alternatives": ["Acceptable variations of the phrase"],
     "feedback": "Feedback on their speaking",
     "vocabulary_expansion": ["Key vocabulary from the phrase"]
   }
   Speaking exercises use speech-to-text to transcribe the user's speech and compare
   it to the expected phrase. This tests production of language, not just recognition.
   Keep speaking prompts short and clear (1-2 sentences max for beginners, up to 3 for advanced).

9. "reading_comprehension" - Read a passage and answer questions about it
   {
     "type": "reading_comprehension",
     "question": "Main question about the passage (e.g., 'What does María buy at the market?')",
     "reading_passage": "REQUIRED: A paragraph (3-8 sentences) in the target language to read",
     "reading_translation": "English translation of the passage (shown after answering)",
     "reading_questions": [
       {"question": "Question 1 about the passage", "answer": "Expected answer"},
       {"question": "Question 2 about the passage", "answer": "Expected answer"}
     ],
     "vocabulary_highlights": [
       {"word": "mercado", "translation": "market"},
       {"word": "comprar", "translation": "to buy"}
     ],
     "correct_answer": "Expected answer to the main question",
     "feedback": "Feedback explaining the correct answer",
     "vocabulary_expansion": ["Additional vocabulary from the passage"]
   }
   🚨 CRITICAL: reading_passage is REQUIRED - the user needs text to read!
   Reading comprehension tests understanding of written text. The passage should be
   appropriate for the learner's level. Include vocabulary highlights for difficult words.

10. "writing_practice" - Free writing exercise with a prompt
    {
      "type": "writing_practice",
      "question": "Write about the following topic:",
      "writing_prompt": "REQUIRED: The topic/scenario to write about (e.g., 'Describe your typical morning routine')",
      "instruction": "Write 3-5 sentences describing your morning in [target language]",
      "writing_min_words": 20,
      "writing_max_words": 100,
      "correct_answer": "Example response (for reference)",
      "feedback": "Feedback will be generated based on their writing",
      "vocabulary_expansion": ["Helpful vocabulary for this topic"]
    }
    🚨 CRITICAL: writing_prompt is REQUIRED - it tells the user what to write about!
    Writing practice exercises test productive writing ability. Provide clear prompts
    appropriate for the learner's level.

11. "word_order" - Duolingo-style sentence building exercise
    {
      "type": "word_order",
      "question": "Arrange the words to translate: [English sentence]",
      "source_sentence": "REQUIRED: The English sentence to translate (e.g., 'I always eat pizza')",
      "correct_word_order": ["REQUIRED", "list", "of", "words", "in", "correct", "order"],
      "scrambled_words": ["REQUIRED", "same", "words", "but", "shuffled", "randomly"],
      "distractor_words": ["optional", "wrong", "words"],
      "correct_answer": "The complete correct sentence as a string",
      "feedback": "Feedback on the sentence construction",
      "vocabulary_expansion": ["Key vocabulary from the sentence"]
    }
    🚨 CRITICAL: Both scrambled_words AND correct_word_order are REQUIRED!
    🚨 scrambled_words provides the clickable word tiles for the user
    🚨 correct_word_order is used to check their answer
    Example for German A1:
      source_sentence: "The dog is big"
      correct_word_order: ["Der", "Hund", "ist", "groß"]
      scrambled_words: ["groß", "Hund", "Der", "ist"]
      correct_answer: "Der Hund ist groß"

All cards should target the learner's proficiency level.
Images should be used frequently to aid learning.
Audio exercises help develop listening skills - include 2-3 audio cards per lesson.
Speaking exercises help develop oral production - include 1-2 speaking cards per lesson.

IMPORTANT: Image prompts must be safe and educational:
- Use simple, everyday objects and scenes (e.g., "a red apple", "a friendly cat", "a sunny park")
- Avoid: violence, weapons, adult content, controversial topics
- Focus on: food, animals, nature, everyday objects, simple activities, educational scenes
- Keep prompts clear, descriptive, and suitable for language learning
"""

ASSESSMENT_CARD_SCHEMA = """
Assessment cards use the same structure as lesson cards, but are used
for the initial 3-stage fluency assessment. They MUST rely on typed
responses only. Allowed types:
- "image_question" type for visual prompts that require a written answer
- "fill_in_blank" type for grammar accuracy (learner types the missing word/phrase)
- "text_question" type for short or multi-sentence descriptions

Do NOT use "multiple_choice" for assessments. Every stage should require
the learner to produce language in a free-response text box.

The assessment should determine:
- Vocabulary range
- Grammar complexity
- Writing fluency
- Comprehension level

IMPORTANT: All image prompts must be safe, educational, and appropriate for all ages.
Use simple, clear prompts like "a red apple on a white table" or "a friendly dog playing in a park".
"""



import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from openai import OpenAI
import logging
import json
import zipfile
import io
import os
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_transcript(url: str, language: str = "en") -> str:
    """
    Extract transcript from a YouTube video in a specified language.
    """
    try:
        video_id = url.split("v=")[-1]
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Try fetching the transcript in the requested language
        if language in transcript_list:
            transcript = transcript_list.find_transcript([language]).fetch()
        else:
            transcript = transcript_list.find_transcript([language]).translate(language).fetch()

        cleaned_transcript = " ".join([entry['text'] for entry in transcript])
        return cleaned_transcript

    except Exception as e:
        logger.error(f"Error extracting YouTube transcript: {e}")
        st.error(f"Could not extract transcript: {e}")
        return ""

def get_ai_analysis(client: OpenAI, transcript: str, prompt: str) -> str:
    """
    Generate AI analysis of the transcript using OpenAI API.
    """
    try:
        full_prompt = f"{prompt}\n\nTranscript:\n{transcript}"
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": full_prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating AI analysis: {e}")
        raise Exception(f"Failed to generate AI analysis: {str(e)}")

def clean_text(text: str) -> str:
    """Clean up text by replacing special characters."""
    return text.replace('ß', 'ss')

def transform_mcq(json_str: str) -> list:
    """Transform MCQ JSON to H5P-compatible question list."""
    try:
        data = json.loads(json_str)
        questions = []

        for q in data.get('questions_list', []):
            question = {
                "library": "H5P.MultiChoice 1.16",
                "params": {
                    "question": clean_text(q['question_text']),
                    "answers": [
                        {
                            "text": clean_text(answer['text']),
                            "correct": answer['is_correct'],
                            "tipsAndFeedback": {
                                "tip": "",
                                "chosenFeedback": clean_text(answer['feedback']),
                                "notChosenFeedback": ""
                            }
                        } for answer in q['answers']
                    ],
                    "behaviour": {
                        "singleAnswer": True,
                        "enableRetry": True,
                        "enableSolutionsButton": True,
                        "enableCheckButton": True,
                        "type": "auto",
                        "singlePoint": False,
                        "randomAnswers": True,
                        "showSolutionsRequiresInput": True,
                        "confirmCheckDialog": False,
                        "confirmRetryDialog": False,
                        "autoCheck": False,
                        "passPercentage": 100,
                        "showScorePoints": True
                    },
                    "media": {"disableImageZooming": False},
                    "overallFeedback": [{"from": 0, "to": 100}],
                    "UI": {
                        "checkAnswerButton": "Überprüfen",
                        "submitAnswerButton": "Absenden",
                        "showSolutionButton": "Lösung anzeigen",
                        "tryAgainButton": "Wiederholen",
                        "tipsLabel": "Hinweis anzeigen",
                        "scoreBarLabel": "Du hast :num von :total Punkten erreicht.",
                        "tipAvailable": "Hinweis verfügbar",
                        "feedbackAvailable": "Rückmeldung verfügbar",
                        "readFeedback": "Rückmeldung vorlesen",
                        "wrongAnswer": "Falsche Antwort",
                        "correctAnswer": "Richtige Antwort",
                        "shouldCheck": "Hätte gewählt werden müssen",
                        "shouldNotCheck": "Hätte nicht gewählt werden sollen",
                        "noInput": "Bitte antworte, bevor du die Lösung ansiehst",
                        "a11yCheck": "Die Antworten überprüfen. Die Auswahlen werden als richtig, falsch oder fehlend markiert.",
                        "a11yShowSolution": "Die Lösung anzeigen. Die richtigen Lösungen werden in der Aufgabe angezeigt.",
                        "a11yRetry": "Die Aufgabe wiederholen. Alle Versuche werden zurückgesetzt und die Aufgabe wird erneut gestartet."
                    },
                    "confirmCheck": {
                        "header": "Beenden?",
                        "body": "Ganz sicher beenden?",
                        "cancelLabel": "Abbrechen",
                        "confirmLabel": "Beenden"
                    },
                    "confirmRetry": {
                        "header": "Wiederholen?",
                        "body": "Ganz sicher wiederholen?",
                        "cancelLabel": "Abbrechen",
                        "confirmLabel": "Bestätigen"
                    }
                },
                "subContentId": str(uuid.uuid4()),
                "metadata": {
                    "contentType": "Multiple Choice",
                    "license": "U",
                    "title": "Unbenannt: Multiple Choice",
                    "authors": [],
                    "changes": [],
                    "extraTitle": "Unbenannt: Multiple Choice"
                }
            }
            questions.append(question)

        return questions
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing MCQ JSON: {e}")
        raise Exception(f"Failed to transform MCQ format: {str(e)}")
    except Exception as e:
        logger.error(f"Error transforming MCQ: {e}")
        raise Exception(f"Failed to transform MCQ format: {str(e)}")

def transform_drag(drag_str: str) -> dict:
    """Transform drag words text to H5P-compatible format."""
    try:
        data = json.loads(drag_str)
        drag_content = data.get('drag_the_words', {}).get('output_template', [])
        if not drag_content:
            drag_content = data.get('drag_the_words', {}).get('output_example', [])
        
        if not drag_content:
            raise Exception("No drag words content found in the response")
            
        # Join all sentences with line breaks
        text_field = "\n".join(clean_text(text) for text in drag_content)
        
        return {
            "media": {
                "disableImageZooming": False
            },
            "taskDescription": "Ziehe die Wörter in die richtigen Felder!",
            "overallFeedback": [
                {"from": 0, "to": 100}
            ],
            "checkAnswer": "Überprüfen",
            "submitAnswer": "Absenden",
            "tryAgain": "Wiederholen",
            "showSolution": "Lösung anzeigen",
            "dropZoneIndex": "Ablagefeld @index.",
            "empty": "Ablagefeld @index ist leer.",
            "contains": "Ablagefeld @index enthält ziehbaren Text @draggable.",
            "ariaDraggableIndex": "@index von @count ziehbaren Texten.",
            "tipLabel": "Tipp anzeigen",
            "correctText": "Richtig!",
            "incorrectText": "Falsch!",
            "resetDropTitle": "Ablagefelder zurücksetzen",
            "resetDropDescription": "Bist du sicher, dass du dieses Ablagefeld zurücksetzen möchtest?",
            "grabbed": "Ziehbarer Text wurde aufgenommen.",
            "cancelledDragging": "Ziehen abgebrochen.",
            "correctAnswer": "Korrekte Antwort:",
            "feedbackHeader": "Rückmeldung",
            "behaviour": {
                "enableRetry": True,
                "enableSolutionsButton": False,
                "enableCheckButton": True,
                "instantFeedback": False
            },
            "scoreBarLabel": "Du hast :num von :total Punkten erreicht.",
            "a11yCheck": "Die Antworten überprüfen. Die Eingaben werden als richtig, falsch oder unbeantwortet markiert.",
            "a11yShowSolution": "Die Lösung anzeigen. Die richtigen Lösungen werden in der Aufgabe angezeigt.",
            "a11yRetry": "Die Aufgabe wiederholen. Alle Eingaben werden zurückgesetzt und die Aufgabe wird erneut gestartet.",
            "textField": text_field
        }
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing drag words JSON: {e}")
        logger.error(f"Received content: {drag_str}")
        raise Exception(f"Failed to transform drag words format: {str(e)}")
    except Exception as e:
        logger.error(f"Error transforming drag words: {e}")
        logger.error(f"Received content: {drag_str}")
        raise Exception(f"Failed to transform drag words format: {str(e)}")

def transform_glossary(glossary_str: str) -> dict:
    """Transform glossary text to H5P-compatible format."""
    try:
        data = json.loads(glossary_str)
        glossary_content = data.get('glossary', {}).get('output_template', [])
        if not glossary_content:
            glossary_content = data.get('glossary', {}).get('output_example', [])
            
        if not glossary_content:
            logger.error(f"Received glossary content: {glossary_str}")
            raise Exception("No glossary content found in the response")
            
        # Format the text field with the glossary entries
        text_field = "\n".join(clean_text(entry) for entry in glossary_content)
        
        # Return the complete H5P DragText parameters
        return {
            "media": {
                "disableImageZooming": False
            },
            "taskDescription": "Ordne die Begriffe den richtigen Definitionen zu!",
            "overallFeedback": [
                {"from": 0, "to": 100}
            ],
            "checkAnswer": "Überprüfen",
            "submitAnswer": "Absenden",
            "tryAgain": "Wiederholen",
            "showSolution": "Lösung anzeigen",
            "dropZoneIndex": "Ablagefeld @index.",
            "empty": "Ablagefeld @index ist leer.",
            "contains": "Ablagefeld @index enthält ziehbaren Text @draggable.",
            "ariaDraggableIndex": "@index von @count ziehbaren Texten.",
            "tipLabel": "Tipp anzeigen",
            "correctText": "Richtig!",
            "incorrectText": "Falsch!",
            "resetDropTitle": "Ablagefelder zurücksetzen",
            "resetDropDescription": "Bist du sicher, dass du dieses Ablagefeld zurücksetzen möchtest?",
            "grabbed": "Ziehbarer Text wurde aufgenommen.",
            "cancelledDragging": "Ziehen abgebrochen.",
            "correctAnswer": "Korrekte Antwort:",
            "feedbackHeader": "Rückmeldung",
            "behaviour": {
                "enableRetry": True,
                "enableSolutionsButton": False,
                "enableCheckButton": True,
                "instantFeedback": False
            },
            "scoreBarLabel": "Du hast :num von :total Punkten erreicht.",
            "a11yCheck": "Die Antworten überprüfen. Die Eingaben werden als richtig, falsch oder unbeantwortet markiert.",
            "a11yShowSolution": "Die Lösung anzeigen. Die richtigen Lösungen werden in der Aufgabe angezeigt.",
            "a11yRetry": "Die Aufgabe wiederholen. Alle Eingaben werden zurückgesetzt und die Aufgabe wird erneut gestartet.",
            "textField": text_field
        }
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing glossary JSON: {e}")
        logger.error(f"Received content: {glossary_str}")
        raise Exception(f"Failed to transform glossary format: {str(e)}")
    except Exception as e:
        logger.error(f"Error transforming glossary: {e}")
        logger.error(f"Received content: {glossary_str}")
        raise Exception(f"Failed to transform glossary format: {str(e)}")

def get_welcome_message(client: OpenAI, transcript: str) -> tuple[str, str]:
    """Generate a welcome message based on the video transcript. Returns (welcome_text, topic)."""
    try:
        # Modified prompt to be more explicit and structured
        prompt = """Analyze this transcript and provide:
1. A concise topic title (maximum 5 words)
2. A welcome message in HTML format that includes:
   - Brief introduction
   - 3 bullet points on why this topic is important
   - 3 bullet points on learning objectives

Format your response EXACTLY like this example:
{
    "topic": "Introduction to Quantum Physics",
    "welcome_html": "<p>Willkommen zu dieser Einheit über Quantenphysik!</p><h3>❗ Wieso ist es wichtig?</h3><ul><li>Point 1</li><li>Point 2</li><li>Point 3</li></ul><h3>🎯 Lernziele</h3><ul><li>Objective 1</li><li>Objective 2</li><li>Objective 3</li></ul>"
}

Transcript:
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates structured educational content in German. Always respond with valid JSON."},
                {"role": "user", "content": prompt + transcript}
            ],
            temperature=0.7,  # Add some creativity while maintaining coherence
            max_tokens=1000   # Ensure enough space for the response
        )
        
        try:
            # Get the response content
            content = response.choices[0].message.content.strip()
            
            # Log the raw response for debugging
            logger.info(f"OpenAI response: {content}")
            
            # Try to parse the JSON
            result = json.loads(content)
            
            # Validate the required fields
            if not isinstance(result, dict) or 'topic' not in result or 'welcome_html' not in result:
                raise ValueError("Response missing required fields")
                
            welcome_text = result['welcome_html'].strip()
            topic = result['topic'].strip()
            
            # Validate the content
            if not welcome_text or not topic:
                raise ValueError("Empty content received")
                
            return welcome_text, topic
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Raw response: {content}")
            raise
            
    except Exception as e:
        logger.error(f"Error generating welcome message: {e}")
        st.error(f"Failed to generate welcome message: {str(e)}")
        return None, None

def create_content_json(video_url: str, mcq_content: str = None, glossary_content: str = None, drag_content: str = None, welcome_text: str = None) -> str:
    """Create the content.json structure based on the generated content."""
    content_json = {
        "content": [
            # Welcome Message
            {
                "content": {
                    "params": {
                        "text": welcome_text if welcome_text else "<p>Willkommen zu dieser Einheit! Bitte schaue dir das Video an und beantworte anschließend die Fragen unten.</p>"
                                "<h3>❗ Wieso ist es wichtig?</h3>"
                                "<ul>"
                                "<li>Es hilft, das Thema besser zu verstehen.</li>"
                                "<li>Fördert kritisches Denken.</li>"
                                "<li>Bereitet auf Prüfungen vor.</li>"
                                "</ul>"
                                "<h3>Lernziele</h3>"
                                "<ul>"
                                "<li>Verstehen der Grundkonzepte.</li>"
                                "<li>Anwenden des Gelernten in praktischen Beispielen.</li>"
                                "</ul>"
                    },
                    "library": "H5P.AdvancedText 1.1",
                    "metadata": {
                        "contentType": "Text",
                        "license": "U",
                        "title": "Unbenannt: Text",
                        "authors": [],
                        "changes": []
                    },
                    "subContentId": "d03c6172-1d14-429e-a4ba-b0fd4856c35c"
                },
                "useSeparator": "enabled"
            },
            # Video Block
            {
                "content": {
                    "params": {
                        "visuals": {
                            "fit": True,
                            "controls": True
                        },
                        "playback": {
                            "autoplay": False,
                            "loop": False
                        },
                        "l10n": {
                            "name": "Video",
                            "loading": "Videoplayer lädt...",
                            "noPlayers": "Keine Videoplayer gefunden, die das vorliegende Videoformat unterstützen.",
                            "noSources": "Es wurden für das Video keine Quellen angegeben.",
                            "aborted": "Das Abspielen des Videos wurde abgebrochen.",
                            "networkFailure": "Netzwerkfehler.",
                            "cannotDecode": "Dekodierung des Mediums nicht möglich.",
                            "formatNotSupported": "Videoformat wird nicht unterstützt.",
                            "mediaEncrypted": "Medium verschlüsselt.",
                            "unknownError": "Unbekannter Fehler.",
                            "invalidYtId": "Ungültige YouTube-ID.",
                            "unknownYtId": "Video mit dieser YouTube-ID konnte nicht gefunden werden.",
                            "restrictedYt": "Der Besitzer dieses Videos erlaubt kein Einbetten."
                        },
                        "sources": [
                            {
                                "path": video_url,
                                "mime": "video/YouTube",
                                "copyright": {
                                    "license": "U"
                                },
                                "aspectRatio": "16:9"
                            }
                        ]
                    },
                    "library": "H5P.Video 1.6",
                    "metadata": {
                        "contentType": "Video",
                        "license": "U",
                        "title": "Unbenannt: Video",
                        "authors": [],
                        "changes": [],
                        "extraTitle": "Unbenannt: Video"
                    },
                    "subContentId": "9897af8b-60d1-4d11-a5e4-21694eab09ce"
                },
                "useSeparator": "enabled"
            },
            # Verständnisfragen Header
            {
                "content": {
                    "params": {
                        "text": "<h3>Verständnisfragen</h3>"
                    },
                    "library": "H5P.AdvancedText 1.1",
                    "metadata": {
                        "contentType": "Text",
                        "license": "U",
                        "title": "Unbenannt: Text",
                        "authors": [],
                        "changes": []
                    },
                    "subContentId": "1af00a81-64bc-457a-87d9-238296da10b4"
                },
                "useSeparator": "enabled"
            }
        ]
    }

    # Add Multiple Choice Questions if present
    if mcq_content:
        content_json["content"].append({
            "content": {
                "params": {
                    "introPage": {
                        "showIntroPage": False,
                        "startButtonText": "Quiz starten",
                        "introduction": ""
                    },
                    "progressType": "dots",
                    "passPercentage": 50,
                    "disableBackwardsNavigation": False,
                    "randomQuestions": True,
                    "endGame": {
                        "showResultPage": True,
                        "showSolutionButton": True,
                        "showRetryButton": True,
                        "noResultMessage": "Quiz beendet",
                        "message": "Dein Ergebnis:",
                        "scoreBarLabel": "Du hast @score von @total Punkten erreicht.",
                        "overallFeedback": [
                            {"from": 0, "to": 100}
                        ],
                        "solutionButtonText": "Lösung anzeigen",
                        "retryButtonText": "Wiederholen",
                        "finishButtonText": "Beenden",
                        "submitButtonText": "Absenden",
                        "showAnimations": False,
                        "skippable": False,
                        "skipButtonText": "Video überspringen"
                    },
                    "texts": {
                        "prevButton": "Zurück",
                        "nextButton": "Weiter",
                        "finishButton": "Beenden",
                        "submitButton": "Absenden",
                        "textualProgress": "Aktuelle Frage: @current von @total Fragen",
                        "jumpToQuestion": "Frage %d von %total",
                        "questionLabel": "Frage",
                        "readSpeakerProgress": "Frage @current von @total",
                        "unansweredText": "Unbeantwortet",
                        "answeredText": "Beantwortet",
                        "currentQuestionText": "Aktuelle Frage",
                        "navigationLabel": "Fragen"
                    },
                    "override": {
                        "checkButton": True,
                        "showSolutionButton": "off",
                        "retryButton": "off"
                    },
                    "questions": mcq_content,  # Eingefügte MCQs
                    "poolSize": 5
                },
                "library": "H5P.QuestionSet 1.20",
                "metadata": {
                    "contentType": "Question Set",
                    "license": "U",
                    "title": "Multiple Choice Fragen",
                    "authors": [],
                    "changes": [],
                    "extraTitle": "Multiple Choice Fragen"
                },
                "subContentId": "ffae1922-ba4b-43b2-b3a0-3e776817fc58"
            },
            "useSeparator": "enabled"
        })

    # Add DragText: Lückentext if present
    if drag_content:
        content_json["content"].append({
            "content": {
                "params": drag_content,  # Now using the full params dictionary
                "library": "H5P.DragText 1.10",
                "metadata": {
                    "contentType": "Drag the Words",
                    "license": "U",
                    "title": "Lückentext",
                    "authors": [],
                    "changes": [],
                    "extraTitle": "Lückentext"
                },
                "subContentId": "ca61533c-d106-410f-929b-c223a852c995"
            },
            "useSeparator": "enabled"
        })

    # Add DragText: Glossar if present
    if glossary_content:
        content_json["content"].append({
            "content": {
                "params": glossary_content,  # Now using the full params dictionary
                "library": "H5P.DragText 1.10",
                "metadata": {
                    "contentType": "Drag the Words",
                    "license": "U",
                    "title": "Glossar",
                    "authors": [],
                    "changes": [],
                    "extraTitle": "Glossar"
                },
                "subContentId": "a203f8b4-8c1e-448a-9cd5-7e81cc413ba5"
            },
            "useSeparator": "enabled"
        })

    return json.dumps(content_json, ensure_ascii=False, indent=2)

def create_h5p_json(topic: str) -> str:
    """Create the h5p.json structure with the given topic."""
    h5p_json = {
        "embedTypes": ["iframe"],
        "language": "de",
        "defaultLanguage": "de",
        "license": "U",
        "extraTitle": topic,
        "title": topic,
        "mainLibrary": "H5P.Column",
        "preloadedDependencies": [
            {"machineName": "H5P.AdvancedText", "majorVersion": 1, "minorVersion": 1},
            {"machineName": "H5P.Video", "majorVersion": 1, "minorVersion": 6},
            {"machineName": "H5P.MultiChoice", "majorVersion": 1, "minorVersion": 16},
            {"machineName": "FontAwesome", "majorVersion": 4, "minorVersion": 5},
            {"machineName": "H5P.JoubelUI", "majorVersion": 1, "minorVersion": 3},
            {"machineName": "H5P.Transition", "majorVersion": 1, "minorVersion": 0},
            {"machineName": "H5P.FontIcons", "majorVersion": 1, "minorVersion": 0},
            {"machineName": "H5P.Question", "majorVersion": 1, "minorVersion": 5},
            {"machineName": "H5P.QuestionSet", "majorVersion": 1, "minorVersion": 20},
            {"machineName": "H5P.DragText", "majorVersion": 1, "minorVersion": 10},
            {"machineName": "jQuery.ui", "majorVersion": 1, "minorVersion": 10},
            {"machineName": "H5P.Column", "majorVersion": 1, "minorVersion": 18}
        ]
    }
    return json.dumps(h5p_json, ensure_ascii=False)

def main():
    st.set_page_config(page_title="YouTube Content Analyzer", page_icon="🎥")
    
    # Initialize session state for results if not exists
    if 'results' not in st.session_state:
        st.session_state.results = {}
    if 'transcript' not in st.session_state:
        st.session_state.transcript = ""
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")
        api_key = st.text_input("OpenAI API Key", type="password")
        if api_key:
            client = OpenAI(api_key=api_key)
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This app extracts YouTube video transcripts and analyzes them using AI.
        1. Enter your OpenAI API Key
        2. Paste a YouTube URL
        3. Select content types to generate
        4. Get AI-powered analysis
        """)
    
    # Main content
    st.title("🎥 YouTube Content Analyzer")
    st.markdown("### Transform video content into valuable insights using AI")
    
    # Input section
    url = st.text_input("YouTube Video URL", placeholder="https://www.youtube.com/watch?v=example")
    
    col1, col2 = st.columns(2)
    with col1:
        language = st.selectbox(
            "Transcript Language",
            options=["en", "de", "es", "fr", "auto"],
            index=0
        )
    
    with col2:
        model = st.selectbox(
            "OpenAI Model",
            options=["gpt-4o-mini", "gpt-4o"],
            index=0
        )
    
    # Content type selection
    st.markdown("### Select Content Types to Generate")
    generate_mcq = st.checkbox("Multiple Choice Questions")
    generate_glossary = st.checkbox("Glossary")
    generate_drag = st.checkbox("Drag The Words")

    # Store prompts in variables (hidden from UI)
    mcq_prompt = """//goal
- you are specialized in generating multiple choice questions tailored to the format outlined below.
- you answer in the same language as the input.
- You focus on clarity and relevance for 15-20 years old students in switzerland, avoiding overly complex language and providing outputs ready for immediate use.

//steps
1. The user uploads the transcript of a video.
2. read the text and identify key topics to be understood
3. read the instructions below
4. generate 4 multiple choice questions level 'Erinnern' according to the 'bloom_levels_closed' guidelines in the same language as the user's input.
5. generate 4 multiple choice questions level 'Verstehen' according to the 'bloom_levels_closed' guidelines in the same language as the user's input.
6. You always answer in German per 'Sie-Form' or in the Language of the upload
7. refer to the 'templates_closed' for rendering output.

//output
- OUTPUT include the generated questions
- STRICTLY follow the formatting of the 'templates_closed'
- IMPORTANT: the output is just the json schema.

//bloom_levels_closed 
# Bloom Level: 'Erinnern'
Question Type: For recall-based tasks
Design Approach:
Focus on recognition and recall of facts.
Use straightforward questions that require identification of correct information.

# Bloom Level: 'Verstehen'
Question Type: Questions at this level assess comprehension and interpretation
Design Approach:
Emphasize explanation of ideas or concepts.
Questions should assess comprehension through interpretation or summary.

//rules
- Each question has ALWAYS 3 Answers
- there are 1 or 2 correct answers
- All the answers have a feedback.
- Generate plausible incorrect answers.
- feedback_correct contain additional information with a real life example in two short sentences with bold key terms for enhanced readability between **. E.g. this is a **bold term**
- feedback_wrong contain the correct answer inclusive and an explanation in one sentence, why it is the correct one with bold key terms for enhanced readability between **. E.g. this is a **bold term**
- Use an empty line to separate each question.
- ALWAYS generate for each textblock one multiple choice question for each level according to the 'bloom_levels_closed' 

Please generate a list of questions in the following structure:

{
  "questions_list": [
    {
      "bloom_level": "Erinnern",
      "question_text": "What is the capital of France?",
      "answers": [
        {
          "text": "Paris",
          "is_correct": true,
          "feedback": "✅ Paris is the correct answer."
        },
        {
          "text": "London",
          "is_correct": false,
          "feedback": "❌ London is incorrect. The correct answer is Paris."
        }
      ]
    }
  ]
}

Ensure that each item in the list has:
- A **bloom_level** string (e.g., "Erinnern").
- A **question_text** string.
- An **answers** array containing multiple answers, with each answer having **text**, **is_correct**, and **feedback** fields.

//templates_closed
{
  "questions_list": [
    {
      "bloom_level": "Erinnern",  // Bloom Level 'Erinnern' - Recall-based task
      "question_text": "{{question_text_erinnern}}",  // Text of the recall question
      "answers": [
        {
          "text": "{{correct_answer_1}}",  // Correct answer text
          "is_correct": true,  // Indicates this is the correct answer
          "feedback": "✅ {{feedback_correct_1}}"  // Feedback for the correct answer, explaining why it's correct with additional information and **bold** keywords
        },
        {
          "text": "{{wrong_answer_1}}",  // Plausible wrong answer 1
          "is_correct": false,  // Indicates this is an incorrect answer
          "feedback": "❌ {{feedback_wrong_1}}"  // Feedback for the wrong answer, explaining why it's wrong and including the correct answer and **bold** keywords
        },
        {
          "text": "{{wrong_answer_2}}",  // Plausible wrong answer 2
          "is_correct": false,  // Indicates this is an incorrect answer
          "feedback": "❌ {{feedback_wrong_2}}"  // Feedback for the wrong answer, explaining why it's wrong and including the correct answer and **bold** keywords
        }
      ]
      // Instruction: Generate three more 'Erinnern' level questions.
      // Each question should focus on recall-based tasks, ensuring students can recognize and recall factual information from the text.
      // For each question, provide 1 or 2 correct answers and plausible wrong answers, ensuring the feedback follows the same format.
    },
    {
      "bloom_level": "Verstehen",  // Bloom Level 'Verstehen' - Comprehension-based task
      "question_text": "{{question_text_verstehen}}",  // Text of the comprehension question
      "answers": [
        {
          "text": "{{correct_answer_2}}",  // Correct answer text
          "is_correct": true,  // Indicates this is the correct answer
          "feedback": "✅ {{feedback_correct_2}}"  // Feedback for the correct answer, explaining why it's correct with additional information and **bold** keywords
        },
        {
          "text": "{{wrong_answer_3}}",  // Plausible wrong answer 3
          "is_correct": false,  // Indicates this is an incorrect answer
          "feedback": "❌ {{feedback_wrong_3}}"  // Feedback for the wrong answer, explaining why it's wrong and including the correct answer and **bold** keywords
        },
        {
          "text": "{{wrong_answer_4}}",  // Plausible wrong answer 4
          "is_correct": false,  // Indicates this is an incorrect answer
          "feedback": "❌ {{feedback_wrong_4}}"  // Feedback for the wrong answer, explaining why it's wrong and including the correct answer and **bold** keywords
        }
      ]
      // Instruction: Generate three more 'Verstehen' level questions.
      // Focus on questions that assess the students' comprehension of concepts.
      // Provide 1 or 2 correct answers and plausible wrong answers, ensuring feedback clearly explains why the answers are correct or incorrect, using real-life examples when relevant.
    }
  ]
}
"""

    glossary_prompt = """//goal
You are specialized in creating glossary for Swiss students aged 15 to 20, based on the levels of Bloom's Taxonomy and according to the format 'templatesH5P.txt'.
You answer in the same language of the user.

//assignment
- Your main task is to analyze texts provided by users, extract the main keywords for the understanding of the video, and generate suitable glossary, as desired by the user.
- You strictly follow the formatting rules from 'templatesH5P.txt', including specific feedback and textual hints for glossary and drag the wordsquestions.

//output
- The output consists exclusively of formatted texts strictly adhering to the 'templatesH5P.txt' standards, without additional explanations.
- You always respond in the language of the input text. The interaction style is clear and precise, focused on the exact compliance with the given format, suitable for an educational environment.

//'templatesH5P.txt'
{
  "glossary": {
    "output_template": [
      "*term1:hint for term1*: Definition of term1",
      "*term2:hint for term2*: Definition of term2",
      "*term3:hint for term3*: Definition of term3"
    ]
  }
}

//output_example
{
  "glossary": {
    "output_example": [
      "*photosynthesis:Process plants use to convert sunlight into energy*: The process by which green plants and some other organisms use sunlight to synthesize foods from carbon dioxide and water.",
      "*mitochondria:Organelle known as the powerhouse of the cell*: A membrane-bound organelle found in the cytoplasm of eukaryotic cells that produces energy in the form of ATP.",
      "*ecosystem:Interaction of living organisms and their environment*: A biological community of interacting organisms and their physical environment."
    ]
  }
}
"""

    drag_prompt = """//goal
You are specialized in creating educational drag the words for Swiss students aged 15 to 20, based on the levels of Bloom's Taxonomy and according to the format 'templatesH5P.txt'.
You answer in the same language of the user.

//assignment
- Your main task is to analyze texts provided by users, extract the main information, and generate suitable drag the words texts as desired by the user.
- The texts check various levels of 'Bloom's Taxonomy'.
- You strictly follow the formatting rules from 'templatesH5P.txt', including textual hints and drag the words texts.

//Bloom's Taxonomy
- Level 1 Knowledge: Learners reproduce what they have previously learned. The examination material had to be memorized or practiced.
- Level 2 Understanding: Learners demonstrate understanding by having the learned material present in a context that differs from the context in which it was learned.
- Level 3 Application: Learners apply something learned in a new situation. This application situation has not occurred before.
- Level 4 Analysis: Learners break down models, procedures, or others into their components. They must discover the principles of structure or internal structures in complex situations. They recognize relationships.

//output
- The output consists exclusively of formatted texts strictly adhering to the 'templatesH5P.txt' standards, without additional explanations, Bloom levels, or types of questions.
- You always respond in the language of the input text. The interaction style is clear and precise, focused on the exact compliance with the given format, suitable for an educational environment.

//'templatesH5P.txt'
{
  "drag_the_words": {
    "output_template": [
      "Sentence with *word1:hint for word1*, followed by *word2:hint for word2*, and *word3:hint for word3*."
    ]
  }
}

//output_example
{
  "drag_the_words": {
    "output_example": [
      "In the United States, the Government includes three distinct branches: the *legislative:Which branch is the U.S. Congress part of?*, the Executive headed by the *President:Who leads the executive branch?*, and the judicial branch, which includes the *Supreme Court:What is the highest court in the United States?*.",
      "The water cycle involves processes such as *evaporation:What is the process of water turning into vapor?*, *condensation:What happens when water vapor cools and forms clouds?*, and *precipitation:What is the term for rain, snow, sleet, or hail falling from the sky?*."
    ]
  }
}
"""

    # Process button
    if st.button("🚀 Generate Content"):
        if not url:
            st.error("Please enter a YouTube URL")
            return
        if not api_key:
            st.error("Please enter your OpenAI API key")
            return
        if not any([generate_mcq, generate_glossary, generate_drag]):
            st.error("Please select at least one content type to generate")
            return
            
        try:
            with st.spinner("Processing content..."):
                # Extract transcript
                st.session_state.transcript = extract_transcript(url, language)
                if not st.session_state.transcript:
                    st.error("Failed to extract transcript")
                    return

                # Generate welcome message and topic
                welcome_text, topic = get_welcome_message(client, st.session_state.transcript)
                if welcome_text is None or topic is None:
                    st.warning("Using default welcome message and topic")
                    welcome_text = "<p>Willkommen zu dieser Einheit!</p>"
                    topic = "Unbenannte Einheit"
                else:
                    st.success("Welcome message and topic generated successfully")

                # Generate selected content types
                mcq_content = None
                glossary_content = None
                drag_content = None

                if generate_mcq:
                    mcq_raw = get_ai_analysis(client, st.session_state.transcript, mcq_prompt)
                    mcq_content = transform_mcq(mcq_raw)

                if generate_glossary:
                    glossary_raw = get_ai_analysis(client, st.session_state.transcript, glossary_prompt)
                    logger.info(f"Raw glossary response: {glossary_raw}")
                    glossary_content = transform_glossary(glossary_raw)

                if generate_drag:
                    drag_raw = get_ai_analysis(client, st.session_state.transcript, drag_prompt)
                    drag_content = transform_drag(drag_raw)

                # Store results in session state
                st.session_state.results = {
                    'mcq': mcq_content,
                    'glossary': glossary_content,
                    'drag': drag_content,
                    'welcome': welcome_text,
                    'topic': topic,  # Store topic in session state
                    'url': url
                }

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            return

    # Display results if they exist
    if st.session_state.results and any(st.session_state.results.values()):
        st.success("Content generated successfully!")
        
        # Download buttons moved above the content tabs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.download_button(
                label="📥 Transcript",
                data=clean_text(st.session_state.transcript),
                file_name=f"youtube_transcript_{language}.txt",
                mime="text/plain"
            )
        
        # Transform the content before creating download buttons and tabs
        transformed_content = {}
        
        if 'mcq' in st.session_state.results and st.session_state.results['mcq']:
            transformed_content['mcq'] = json.dumps(st.session_state.results['mcq'], indent=2)
            with col2:
                st.download_button(
                    label="📥 MCQ",
                    data=transformed_content['mcq'],
                    file_name="mcq_questions.txt",
                    mime="text/plain"
                )
        
        if 'glossary' in st.session_state.results and st.session_state.results['glossary']:
            transformed_content['glossary'] = json.dumps(st.session_state.results['glossary'], indent=2)
            with col3:
                st.download_button(
                    label="📥 Glossary",
                    data=transformed_content['glossary'],
                    file_name="glossary.txt",
                    mime="text/plain"
                )
        
        if 'drag' in st.session_state.results and st.session_state.results['drag']:
            transformed_content['drag'] = json.dumps(st.session_state.results['drag'], indent=2)
            with col4:
                st.download_button(
                    label="📥 Drag Words",
                    data=transformed_content['drag'],
                    file_name="drag_words.txt",
                    mime="text/plain"
                )
        
        # Display transformed content in tabs instead of raw JSON
        tabs = st.tabs([k.upper() for k in transformed_content.keys()])
        for tab, (content_type, content) in zip(tabs, transformed_content.items()):
            with tab:
                st.markdown(content)
        
        # H5P Package Generation
        st.markdown("---")
        st.markdown("### Download H5P Package")
        
        # Resolve absolute path to template.zip
        current_dir = os.path.dirname(os.path.abspath(__file__))
        template_zip_path = os.path.join(current_dir, 'template.zip')
        
        # Get required variables from session state
        mcq_content = st.session_state.results.get('mcq')
        glossary_content = st.session_state.results.get('glossary')
        drag_content = st.session_state.results.get('drag')
        welcome_text = st.session_state.results.get('welcome')
        topic = st.session_state.results.get('topic', 'Unbenannte Einheit')
        video_url = st.session_state.results.get('url')
        
        # Generate the JSON strings for H5P content and metadata
        content_json_str = create_content_json(
            video_url=video_url,
            mcq_content=mcq_content,
            glossary_content=glossary_content,
            drag_content=drag_content,
            welcome_text=welcome_text
        )
        h5p_json_str = create_h5p_json(topic)
        
        # Check and use the template.zip file
        if not os.path.exists(template_zip_path):
            st.error(f"The template file '{template_zip_path}' does not exist.")
            st.write("Current directory contents:", os.listdir(current_dir))
        else:
            st.success(f"Found the template file: {template_zip_path}")
            # Process the template.zip file
            try:
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zip_new:
                    with zipfile.ZipFile(template_zip_path, 'r') as zip_ref:
                        for item in zip_ref.infolist():
                            if item.filename not in ['content/content.json', 'h5p.json']:
                                zip_new.writestr(item, zip_ref.read(item.filename))
                    
                    zip_new.writestr('content/content.json', content_json_str)
                    zip_new.writestr('h5p.json', h5p_json_str)
            
                buffer.seek(0)
                updated_zip_bytes = buffer.getvalue()
            
                # Clean the topic to create a valid filename
                clean_filename = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
                clean_filename = clean_filename.replace(' ', '_')

                st.download_button(
                    label="📥 Download H5P Package",
                    data=updated_zip_bytes,
                    file_name=f"{clean_filename}.h5p",  # Use the cleaned topic name
                    mime="application/zip"
                )
            except Exception as e:
                st.error(f"Failed to generate H5P package: {str(e)}")

if __name__ == "__main__":
    main()





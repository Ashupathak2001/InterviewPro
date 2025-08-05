import streamlit as st
st.set_page_config(
        page_title="ForgeMe",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from datetime import datetime, timedelta
import json
import time
import re
from typing import Dict, List, Any, Optional
import sqlite3
from dataclasses import dataclass, asdict
import hashlib
import uuid
import cv2
import queue
import threading
import os
from PIL import Image

from db_manager import (
    User, UserProfile, PracticeSession, UserAchievement,
    create_user, get_user_by_email, get_user_by_id, update_user,
    save_user_profile, get_user_profile,
    save_practice_session, get_user_practice_sessions, get_user_achievements,
    save_user_achievement, get_db_connection, initialize_db
)

# Enhanced imports for new features
# from textblob import TextBlob
import requests
import PyPDF2
# from io import BytesIO
# import asyncio
# import websockets
from concurrent.futures import ThreadPoolExecutor
import logging
from speech_recognition import Recognizer, Microphone, WaitTimeoutError, UnknownValueError, RequestError
from utils.llm_feedback import get_feedback, default_feedback
from utils.research import generate_company_insights
from utils.resume_analysis import analyze_resume_text
from utils.resume_text_extractor import extract_text_from_file
from utils.coding_coach import show_coding_practice_page
from utils.jd_help import generate_jd_questions_and_tips
from dashboard import show_dashboard
from analytics import show_analytics
from utils.questions_handler import QuestionsHandler, QuestionCategory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Database Initialization (Call this once per session) ---
if 'db_initialized' not in st.session_state:
    initialize_db()
    st.session_state.db_initialized = True


@dataclass
class RealTimeMetrics:
    """Real-time metrics during interview"""
    timestamp: datetime
    audio_confidence: float
    speech_clarity: float
    eye_contact_score: float
    posture_score: float
    gesture_appropriateness: float
    speaking_pace: float
    volume_level: float
    facial_expression: str
    attention_score: float

@dataclass
class InterviewSession:
    """Enhanced interview session with real-time data"""
    id: str
    user_id: str
    question: str
    transcription: str
    audio_file_path: str
    video_metrics: List[RealTimeMetrics]
    scores: Dict[str, float]
    feedback: str
    duration: int
    industry: str
    question_type: str
    created_at: datetime
    real_time_feedback: List[str]


class SpeechRecognitionManager:
    """Enhanced speech recognition with real-time processing"""
    
    def __init__(self):
        self.recognizer = Recognizer()
        self.microphone = Microphone()
        self.is_recording = False
        self.audio_queue = queue.Queue(maxsize=100)
        self.transcription_queue = queue.Queue(maxsize=100)
        self.current_transcription = ""
        self.lock = threading.Lock()
        self.thread_timeout = 2.0
        
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
        except Exception as e:
            logger.error(f"Microphone initialization failed: {e}")
            raise RuntimeError("Microphone initialization failed") from e
        
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        self.recognizer.operation_timeout = None
        self.recognizer.phrase_threshold = 0.3
        self.recognizer.non_speaking_duration = 0.8
    
    def start_recording(self):
        """Start continuous speech recognition"""
        if self.is_recording:
            logger.warning("Recording already in progress")
            return
            
        self.is_recording = True
        self.current_transcription = ""
        
        try:
            self.audio_thread = threading.Thread(
                target=self._audio_capture_loop,
                name="AudioCaptureThread"
            )
            self.audio_thread.daemon = True
            self.audio_thread.start()
            
            self.transcription_thread = threading.Thread(
                target=self._transcription_loop,
                name="TranscriptionThread"
            )
            self.transcription_thread.daemon = True
            self.transcription_thread.start()
            
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            self.stop_recording()
            raise
    
    def stop_recording(self):
        """Stop recording and return final transcription"""
        if not self.is_recording:
            return self.current_transcription
            
        self.is_recording = False
        final_transcription = ""
        
        try:
            if hasattr(self, 'audio_thread') and self.audio_thread.is_alive():
                self.audio_thread.join(timeout=self.thread_timeout)
            
            if hasattr(self, 'transcription_thread') and self.transcription_thread.is_alive():
                self.transcription_thread.join(timeout=self.thread_timeout)
            
            self._clear_queue(self.audio_queue)
            self._clear_queue(self.transcription_queue)
            
            final_transcription = self.current_transcription.strip()
            self.current_transcription = ""
            
        except Exception as e:
            logger.error(f"Error during recording stop: {e}")
            final_transcription = self.current_transcription.strip()
            
        return final_transcription
    
    def _clear_queue(self, q):
        """Safely clear a queue"""
        with q.mutex:
            q.queue.clear()
            q.all_tasks_done.notify_all()
            q.unfinished_tasks = 0
    
    def _audio_capture_loop(self):
        """Continuous audio capture loop"""
        logger.info("Audio capture started")
        try:
            with self.microphone as source:
                while self.is_recording:
                    try:
                        audio = self.recognizer.listen(
                            source, 
                            timeout=1, 
                            phrase_time_limit=5
                        )
                        if not self.audio_queue.full():
                            self.audio_queue.put(audio, timeout=0.5)
                    except WaitTimeoutError:
                        continue
                    except Exception as e:
                        logger.error(f"Audio capture error: {e}")
                        break
        except Exception as e:
            logger.error(f"Audio capture loop failed: {e}")
        finally:
            logger.info("Audio capture ended")
    
    def _transcription_loop(self):
        """Continuous transcription loop"""
        logger.info("Transcription started")
        while self.is_recording:
            try:
                audio = self.audio_queue.get(timeout=0.5)
                text = self._transcribe_audio(audio)
                if text:
                    with self.lock:
                        self.current_transcription += " " + text
                    if not self.transcription_queue.full():
                        self.transcription_queue.put(text, timeout=0.5)
                self.audio_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Transcription error: {e}")
                break
        logger.info("Transcription ended")
    
    def _transcribe_audio(self, audio):
        """Transcribe audio chunk with fallback"""
        if not audio:
            return ""
            
        try:
            try:
                return self.recognizer.recognize_google(audio, language='en-US')
            except UnknownValueError:
                try:
                    return self.recognizer.recognize_sphinx(audio)
                except Exception:
                    return ""
            except RequestError as e:
                logger.error(f"Google API error: {e}")
                return ""
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""
    
    def get_real_time_transcription(self):
        """Get current transcription"""
        with self.lock:
            return self.current_transcription.strip()
    
    def __del__(self):
        """Cleanup when instance is destroyed"""
        if self.is_recording:
            self.stop_recording()
    

class ComputerVisionAnalyzer:
    """Real-time computer vision analysis for interview behavior"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        
    def analyze_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        metrics = {
            "face_detected": False,
            "num_faces": 0,
            "eye_contact": False,
            "smiling": False
        }

        if len(faces) > 0:
            metrics["face_detected"] = True
            metrics["num_faces"] = len(faces)
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                smiles = self.smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=20)
                metrics["eye_contact"] = len(eyes) >= 2
                metrics["smiling"] = len(smiles) > 0
                break  # Analyze only first face

        return metrics
    
    def start_analysis(self):
        self.metrics_history = []

    def stop_analysis(self):
        self.metrics_history = []


class RealTimeCoach:
    """Real-time interview coaching system"""

    def __init__(self):
        self.speech_manager = SpeechRecognitionManager()
        self.vision_analyzer = ComputerVisionAnalyzer()
        self.is_active = False
        self.feedback_queue = queue.Queue()
        self.metrics_queue = queue.Queue()
        self.coaching_lock = threading.Lock()

        self.coaching_messages = {
            'speaking_too_fast': "🐌 Try speaking a bit slower - your pace is quite fast.",
            'too_many_filler_words': "🎯 Watch out for filler words like 'um', 'uh', 'like'.",
            'be_specific': "📝 Try to be more specific - add concrete examples.",
            'confidence_boost': "💪 You're doing great! Stay confident.",
            'almost_done': "⏰ You have about 15 seconds left to wrap up."
        }

    def start_session(self):
        self.is_active = True
        self.speech_manager.start_recording()
        self.vision_analyzer.start_analysis()

    def stop_session(self):
        self.is_active = False
        transcription = self.speech_manager.stop_recording()
        self.vision_analyzer.stop_analysis()
        return transcription

    def process_frame(self, frame):
        if not self.is_active:
            return frame

        with self.coaching_lock:
            metrics = self.vision_analyzer.analyze_frame(frame)
            if metrics:
                self.metrics_queue.put(metrics)
                # Skip if get_real_time_feedback is not implemented
                feedback = []  # or self.vision_analyzer.get_real_time_feedback()
                for fb in feedback:
                    self.feedback_queue.put(fb)

            return self._annotate_frame(frame, metrics)

    def _annotate_frame(self, frame, metrics):
        if not metrics:
            return frame

        overlay = frame.copy()

        eye_color = (0, 255, 0) if metrics.get("eye_contact_score", 0) > 70 else (0, 165, 255)
        cv2.circle(overlay, (30, 30), 10, eye_color, -1)
        cv2.putText(overlay, f"Eye Contact: {metrics.get('eye_contact_score', 0):.0f}%", (50, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, eye_color, 2)

        posture_color = (0, 255, 0) if metrics.get("posture_score", 0) > 70 else (0, 165, 255)
        cv2.circle(overlay, (30, 60), 10, posture_color, -1)
        cv2.putText(overlay, f"Posture: {metrics.get('posture_score', 0):.0f}%", (50, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, posture_color, 2)

        gesture_color = (0, 255, 0) if metrics.get("gesture_appropriateness", 0) > 70 else (0, 165, 255)
        cv2.circle(overlay, (30, 90), 10, gesture_color, -1)
        cv2.putText(overlay, f"Gestures: {metrics.get('gesture_appropriateness', 0):.0f}%", (50, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, gesture_color, 2)

        return overlay

    def get_real_time_transcription(self):
        return self.speech_manager.get_real_time_transcription()

    def get_real_time_feedback(self):
        feedback = []
        while not self.feedback_queue.empty():
            try:
                feedback.append(self.feedback_queue.get_nowait())
            except queue.Empty:
                break
        return feedback

    def analyze_real_time_performance(self, transcription: str, elapsed_time: int) -> str:
        if not transcription:
            return "🎤 Start speaking - I'm listening!"

        words = transcription.split()
        words_per_minute = len(words) / (elapsed_time / 60) if elapsed_time > 0 else 0

        if words_per_minute > 180:
            return self.coaching_messages['speaking_too_fast']

        filler_words = ['um', 'uh', 'like', 'you know', 'basically', 'actually']
        filler_count = sum(transcription.lower().count(word) for word in filler_words)
        if filler_count > len(words) * 0.1:
            return self.coaching_messages['too_many_filler_words']

        vague_words = ['something', 'things', 'stuff', 'various', 'many']
        if any(word in transcription.lower() for word in vague_words):
            return self.coaching_messages['be_specific']

        if elapsed_time > 45:
            return self.coaching_messages['almost_done']

        return self.coaching_messages['confidence_boost']


class MultiModalAnalyzer:
    def analyze_comprehensive_performance(
        self, 
        transcription: str, 
        duration: int,
        simulated_video_data: Dict = None
    ) -> Dict[str, Any]:
        """Analyze performance based on text + simulated video metrics."""

        text_analysis = self.analyze_text_quality(transcription)
        video_analysis = self.simulate_video_analysis(duration)

        # Allow override or patching in test mode
        if simulated_video_data:
            video_analysis.update(simulated_video_data)

        return {
            'text_analysis': text_analysis,
            'video_analysis': video_analysis
        }

    def simulate_video_analysis(self, duration: int) -> Dict[str, Any]:
        """Simulate video analysis (replace with real vision data in future)"""
        confidence_score = np.random.uniform(60, 90)
        engagement_score = np.random.uniform(70, 95)

        return {
            'confidence_score': confidence_score,
            'engagement_score': engagement_score,
            'eye_contact_percentage': np.random.uniform(60, 85),
            'speaking_pace': 'appropriate' if duration > 30 else 'too fast',
            'body_language': 'confident' if confidence_score > 75 else 'needs improvement'
        }

    def analyze_text_quality(self, transcription: str) -> Dict[str, Any]:
        """Simple text-based heuristics placeholder."""
        word_count = len(transcription.split())
        avg_sentence_length = word_count / (transcription.count('.') + 1)

        return {
            'word_count': word_count,
            'average_sentence_length': round(avg_sentence_length, 2),
            'clarity': 'good' if word_count > 50 else 'needs improvement'
        }

class PredictiveAnalyzer:
    """Analyzes practice session scores to find strengths and improvement areas."""

    def __init__(self):
        self.key_factors = [
            'communication_clarity',
            'technical_depth',
            'behavioral_examples',
            'confidence_presence',
            'question_relevance'
        ]

    def identify_key_factors(self, sessions: List[PracticeSession]) -> List[str]:
        """Return factors where average score is >= 80%."""
        return self._get_factors_by_threshold(sessions, threshold=80, above=True)

    def identify_improvement_areas(self, sessions: List[PracticeSession]) -> List[str]:
        """Return factors where average score is < 70%."""
        return self._get_factors_by_threshold(sessions, threshold=70, above=False)

    def _get_factors_by_threshold(self, sessions: List[PracticeSession], threshold: int, above: bool) -> List[str]:
        factor_scores = {factor: [] for factor in self.key_factors}

        for session in sessions:
            session_scores = json.loads(session.scores)
            for factor in self.key_factors:
                score = session_scores.get(factor)
                if score is not None:
                    factor_scores[factor].append(score)

        selected_factors = []
        for factor, scores in factor_scores.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                if (above and avg_score >= threshold) or (not above and avg_score < threshold):
                    selected_factors.append(factor.replace('_', ' ').title())

        return selected_factors or (['Building foundation skills'] if above else ['Continue practicing all areas'])



# Initialize session state
if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())


# # NEW: Try to load existing profile
# if 'profile' not in st.session_state:
#     db = DatabaseManager()
#     profile = db.load_user_profile(st.session_state.user_id)
#     if profile:
#         st.session_state.profile = profile

if 'current_session' not in st.session_state:
    st.session_state.current_session = None
if 'real_time_coach' not in st.session_state:
    st.session_state.real_time_coach = RealTimeCoach()
if 'multimodal_analyzer' not in st.session_state:
    st.session_state.multimodal_analyzer = MultiModalAnalyzer()
if 'predictive_analyzer' not in st.session_state:
    st.session_state.predictive_analyzer = PredictiveAnalyzer()
if 'questions_handler' not in st.session_state:
    st.session_state.questions_handler = QuestionsHandler("data/questions.json")

def main():
    
    
    # Enforce profile creation
    if 'profile' not in st.session_state and st.session_state.get("page") != "👤 Profile Setup":
        st.warning("⚠️ Please complete your profile setup first.")
        st.session_state.page = "👤 Profile Setup"
        st.rerun()

    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        color: #f0f0f0;
        font-size: 1.2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .achievement-badge {
        background: linear-gradient(45deg, #ffd700, #ffed4a);
        color: #333;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.2rem;
        display: inline-block;
        font-weight: bold;
    }
    .feedback-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
            
    .footer {
        margin-top: 3rem;
        width: 100%;
        background: linear-gradient(to right, #667eea, #764ba2);
        color: white;
        text-align: center;
        padding: 1rem 0;
        font-size: 14px;
        border-top: 2px solid #ddd;
        border-radius: 8px 8px 0 0;
    }

    </style>

    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🎯 ForgeMe</h1>
        <p>Your AI-Powered Interview Simulator</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.markdown("## 📚 Navigation")

    # Default page setup if not already in session state
    if 'page' not in st.session_state:
        st.session_state.page = "🏠 Dashboard"

    # Custom menu
    menu_options = {
        "👤 Profile Setup": "profile_setup",
        "🏠 Dashboard": "dashboard",
        "🎤 Practice Interview": "practice",
        "📊 Analytics": "analytics",
        "📄 Extra Help": "extra_help",
        "📄 ATS Score": "ats score",
        "🏢 Coding": "coding",
        "⚙️ Settings": "settings"
    }

    # Render buttons for navigation
    for label in menu_options:
        if st.sidebar.button(label, key=menu_options[label]):
            st.session_state.page = label


    page_router = {
        "👤 Profile Setup": show_profile_setup,
        "🏠 Dashboard": show_dashboard,
        "🎤 Practice Interview": show_practice_interview,
        "practice_progress": show_practice_progress,
        "practice_summary": show_practice_summary,
        "📊 Analytics": show_analytics,
        "📄 Extra Help": show_extra_help,
        "📄 ATS Score": show_ats_score,
        "🏢 Coding": show_coding_practice,
        "⚙️ Settings": show_settings
    }

    # Call the right function
    current_page = st.session_state.page
    page_router.get(current_page, lambda: st.error("Page not found"))()

    # ✅ Footer at the very end
    st.markdown("""
    <div class="footer">
        © 2025 ForgeMe | Made with ❤️ by Ashish Pathak
    </div>
    """, unsafe_allow_html=True)

def show_profile_setup():
    
    st.title("👤 Candidate Profile")
    
    user_id = st.session_state.user_id
    
    # 🔧 FIX: Always check database first for persistent data
    profile = get_user_profile(user_id)
    
    # Sync database data to session state for current session
    if profile:
        st.session_state.profile = profile
    else:
        # Clear session if no database profile exists
        if "profile" in st.session_state:
            del st.session_state.profile
        profile = None
    
    if profile:
        st.success("✅ You've already set up your profile.")
        with st.expander("👁️ View Profile"):
            st.write(f"**Name:** {profile.name}")
            st.write(f"**Email:** {profile.email}")
            st.write(f"**Level:** {profile.level}")
            st.write(f"**Roles:** {profile.roles}")
            st.write(f"**Industries:** {profile.industries}")
        
        if st.button("🚪 Logout / Reset Profile"):
            # Clear session state
            if "profile" in st.session_state:
                del st.session_state.profile
            # Optional: Delete from database too (uncomment if needed)
            # db.delete_user_profile(user_id)
            st.rerun()
        return
    
    # If no profile exists, show the form
    st.info("📝 No profile found. Please set up your profile to continue.")
    
    # Add a manual load button
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🔍 Check for Existing Profile"):
            profile = get_user_profile(user_id)
            if profile:
                st.session_state.profile = profile
                st.success("Found existing profile!")
                st.rerun()
            else:
                st.warning("No existing profile found.")
    
    with col2:
        if st.button("🆕 Create New Profile"):
            # Clear any existing data and show fresh form
            pass
    
    # Load any existing data from database as defaults (in case of partial save)
    existing_data = profile or {}
    
    name = st.text_input("Full Name", value=existing_data.get("name", ""))
    email = st.text_input("Email", value=existing_data.get("email", ""))
    level = st.selectbox("Career Stage", 
                        ["Student", "Fresher", "Experienced"], 
                        index=["Student", "Fresher", "Experienced"].index(existing_data.get("level", "Student")))
    roles = st.multiselect("Preferred Roles", 
                          ["SDE", "Data Scientist", "PM"], 
                          default=existing_data.get("roles", []))
    industries = st.multiselect("Preferred Industries", 
                               ["IT", "Finance", "Marketing"], 
                               default=existing_data.get("industries", []))
    
    if st.button("💾 Save Profile"):
        # Save to database
        save_user_profile(UserProfile(
            user_id=user_id,
            name=name,
            email=email,
            level=level,
            roles=json.dumps(roles),
            industries=json.dumps(industries)
        ))

        # 🔧 FIX: Create the profile dict with consistent structure
        profile_data = {
            "user_id": user_id,
            "name": name,
            "email": email,
            "level": level,
            "roles": roles,
            "industries": industries
        }
        
        # Save to session for immediate use
        st.session_state.profile = profile_data
        
        st.success("✅ Profile saved successfully!")
        st.rerun()


# Helper to ensure compatibility with Question object or string

def get_question_text(q):
    return q.text if hasattr(q, "text") else q

def show_practice_interview():
    st.title("🎤 Practice Interview Setup")

    # Interview setup with original options
    st.subheader("🎯 Interview Setup")

    col1, col2 = st.columns(2)
    with col1:
        industry = st.selectbox("Select Industry:", list(st.session_state.questions_handler.get_all_fields()))
    with col2:
        question_type = st.selectbox("Question Type:", ["Technical", "Behavioral", "System Design", "Role-Based"])

    with st.expander("🔧 Advanced Options"):
        col1, col2 = st.columns(2)
        with col1:
            custom_question = st.text_area("Custom Question (optional):", placeholder="Enter your own question or leave blank")
        with col2:
            target_company = st.text_input("Target Company (optional):", placeholder="e.g., Google, Microsoft")
            difficulty = st.select_slider("Difficulty Level:", options=['Easy', 'Medium', 'Hard'], value='Medium')

    num_questions = st.slider("How many questions in this session?", 1, 10, 3)

    if st.button("🎙️ Start Interview Session"):
        handler = st.session_state.questions_handler

        if custom_question:
            selected_questions = [custom_question] * num_questions
        else:
            field = industry.lower().replace(" ", "_")
            if "technical" in question_type.lower():
                category = QuestionCategory.TECHNICAL
            elif "behavioral" in question_type.lower():
                category = QuestionCategory.BEHAVIORAL
            elif "system" in question_type.lower():
                category = QuestionCategory.SYSTEM_DESIGN
            else:
                category = QuestionCategory.ROLE_BASED

            selected_questions = handler.get_random_questions(num_questions, field=field, category=category)

        st.session_state.practice_session = {
            "questions": selected_questions,
            "industry": industry,
            "question_type": question_type,
            "target_company": target_company,
            "current": 0,
            "responses": []
        }
        st.session_state.page = "practice_progress"
        st.rerun()

def show_practice_progress():
    session = st.session_state.practice_session
    question_index = session["current"]
    total_questions = len(session["questions"])

    # Handle end of session
    if question_index >= total_questions:
        st.info("You've completed all the questions in this session. Heading to your summary now!")
        st.session_state.page = "practice_summary"
        st.rerun()
        return

    # Display question and progress
    st.title("🎙️ Interview Practice Session")
    st.subheader(f"Question {question_index + 1} of {total_questions}")
    
    question_obj = session["questions"][question_index]
    question_text = get_question_text(question_obj) # Assuming this function exists

    st.info(f"**Question:** {question_text}")
    
    # Display company context if available
    if session.get("target_company"):
        st.markdown("---")
        st.subheader(f"Company Context for {session['target_company']}")
        try:
            company_data = st.session_state.company_researcher.research_company(session["target_company"])
            st.write(f"**Key Values:** {', '.join(company_data['values'])}")
            st.write(f"**Work Culture:** {company_data['culture']['work_style']}")
        except Exception:
            st.warning(f"Could not retrieve company data for {session['target_company']}. Please proceed with the question.")

    st.markdown("---")
    
    # Start answering button and logic
    if st.button("🎤 Start Answering", key="start_answer_btn"):
        coach = st.session_state.real_time_coach
        duration = 60 # You can make this configurable

        # Check for webcam access
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("⚠️ **Webcam not found.** Please ensure a camera is connected and permissions are granted, then try again.")
            return
        
        # Initialize recording and analysis
        st.toast("🔴 Recording started! Please begin your answer.", icon="🎙️")
        
        # Create empty containers for the countdown and video frame
        st_countdown = st.empty()
        st_frame = st.empty()
        
        coach.speech_manager.start_recording()
        coach.vision_analyzer.start_analysis()
        
        start_time = time.time()
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                st.error("Error reading from webcam.")
                break

            # Vision analysis and real-time feedback
            metrics = coach.vision_analyzer.analyze_frame(frame)
            feedback_color = (0, 255, 0) if metrics["eye_contact"] else (0, 0, 255)
            feedback_text = "Good Eye Contact" if metrics["eye_contact"] else " Try Looking at the Camera"
            cv2.putText(frame, feedback_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, feedback_color, 2)
            
            # Display countdown timer in a large font
            time_left = int(duration - (time.time() - start_time))
            st_countdown.markdown(f"## ⏳ Time Left: `{time_left}` seconds")
            
            # Resize the frame to be a bit smaller
            frame_width = int(frame.shape[1] * 0.7) # 70% of original width
            frame_height = int(frame.shape[0] * 0.7) # 70% of original height
            resized_frame = cv2.resize(frame, (frame_width, frame_height))

            # Display webcam feed
            st_frame.image(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=False)

        # Stop recording and analysis
        cap.release()
        coach.vision_analyzer.stop_analysis()
        transcription = coach.speech_manager.stop_recording()

        # Save the response
        session["responses"].append({
            "question": question_text,
            "transcription": transcription,
            "duration": duration,
        })
        
        # Advance to the next question or summary
        session["current"] += 1
        st.session_state.page = "practice_summary" if session["current"] >= total_questions else "practice_progress"
        st.rerun()

def show_practice_summary():
    session = st.session_state.practice_session
    if not session["responses"]:
        st.warning("No responses to summarize. Returning to dashboard.")
        st.session_state.page = "🏠 Dashboard"
        st.rerun()
        return

    st.title("📋 Interview Summary")
    full_report = ""
    sessions_to_save = []

    for i, responses in enumerate(session["responses"]):
        question = responses["question"]
        transcription = responses.get("transcription", "").strip()

        # --- Step 1: Get Feedback ---
        try:
            if not transcription or len(transcription) < 10:
                feedback_result = get_feedback("", question)
            else:
                feedback_result = get_feedback(transcription, question)
        except Exception as e:
            print(f"⚠️ Error getting feedback: {e}")
            feedback_result = default_feedback()

        # --- Step 2: Extract Scores & Detect Default ---
        scores = feedback_result.get("scores", {})
        feedback_text = feedback_result.get("feedback", "").strip()

        # Detect if feedback is generic
        is_generic = (
            feedback_result.get("is_default", False)
            or not scores
            or all(v == 0 for v in scores.values())
            or not feedback_text
        )

        if is_generic:
            feedback_result = default_feedback()
            scores = feedback_result["scores"]
            feedback_text = feedback_result["feedback"]

        responses["score"] = scores.get("overall_score", 0)
        responses["feedback"] = feedback_text

        # --- Step 3: Show UI for Each Question ---
        with st.expander(f"Question {i + 1}"):
            st.write(f"**Q:** {question}")
            st.write(f"**Your Answer:** {transcription or '_No answer provided._'}")
            st.write(f"**Score:** {responses['score']}%")
            st.markdown(feedback_text)
            if is_generic:
                st.warning("⚠️ Automated feedback is generic due to a network or LLM error, or an empty answer.")

        # --- Step 4: Build Full Report ---
        full_report += f"Question {i+1}: {question}\n"
        full_report += f"Answer: {transcription or 'No answer provided'}\n"
        full_report += f"Score: {responses['score']}%\n"
        full_report += f"Feedback: {feedback_text}\n\n"

        # Convert scores dictionary to JSON string
        scores_for_db = json.dumps(scores)

        # Check if feedback_text might also be a dictionary and convert it
        feedback_for_db = feedback_text
        if isinstance(feedback_for_db, dict): # This check ensures it's only dumped if truly a dict
            feedback_for_db = json.dumps(feedback_for_db)
        
        # --- Step 5: Save to DB ---
        session_obj = PracticeSession(
            id=str(uuid.uuid4()),
            user_id=st.session_state.user_id,
            question=question,
            transcription=transcription,
            scores=scores_for_db,
            feedback=feedback_for_db,
            duration=responses["duration"],
            industry=session["industry"],
            question_type=session["question_type"],
            created_at=datetime.now()
        )
        save_practice_session(session_obj)  
        sessions_to_save.append(session_obj)

    # --- Step 6: Show Summary & Download ---
    avg_score = sum(r['score'] for r in session["responses"]) / len(session["responses"])
    st.metric("📊 Average Interview Score", f"{avg_score:.1f}%")
    st.download_button("📥 Download Summary as Text", full_report.encode(), file_name="interview_summary.txt")

    if st.button("🏠 Return to Dashboard"):
        del st.session_state.practice_session
        st.session_state.page = "🏠 Dashboard"
        st.rerun()


            
def show_coding_practice():
    show_coding_practice_page()
    
def show_extra_help():
    st.title("📄 Extra Help")
    st.subheader("💡 Job Description Based Interview")
    st.write("Upload or paste a job description to get tailored interview questions and preparation tips.")

    jd_text = st.text_area(
        "📄 Paste Job Description",
        placeholder="Paste the job description here...",
        height=200
    )

    uploaded_file = st.file_uploader(
        "📎 Or Upload Job Description File",
        type=['pdf', 'docx', 'txt']
    )

    if st.button("🚀 Generate Interview Questions & Tips"):
        if uploaded_file is not None or jd_text.strip():
            with st.spinner("Analyzing job description..."):
                if uploaded_file:
                    job_description_text = extract_text_from_file(uploaded_file)
                else:
                    job_description_text = jd_text

                questions, tips = generate_jd_questions_and_tips(job_description_text)

                if questions or tips:
                    st.success("✅ Analysis complete!")

                    if questions:
                        st.subheader("📝 Tailored Interview Questions")
                        for question in questions:
                            st.write(f"• {question}")

                    if tips:
                        st.subheader("💡 Tips for Success")
                        for tip in tips:
                            st.write(f"• {tip}")
                else:
                    st.error("❌ Gemini could not generate output. Please revise your input.")
        else:
            st.warning("⚠️ Please enter or upload a job description before clicking submit.")



def show_ats_score():
    st.title("ATS Score Analysis & Company Research")
    tab1, tab2 = st.tabs(["Resume Analysis", "Company Research"])
    with tab1:
        st.title("📄 Resume Analysis")
        st.subheader("📋 Upload Your Resume")

        uploaded_file = st.file_uploader(
            "Choose your resume file",
            type=['pdf', 'docx', 'txt'],
            help="Upload your resume in PDF, DOCX, or TXT format"
        )

        if uploaded_file is not None:
            target_role = st.selectbox(
                "🎯 Target Role (for role-specific feedback):",
                ["Software Engineer", "Data Scientist", "Product Manager", "Marketing Manager"]
            )

            with st.spinner("Analyzing your resume with LLM..."):
                resume_text = extract_text_from_file(uploaded_file)
                resume_analysis = analyze_resume_text(resume_text, target_role)

                st.success("Resume analysis complete!")

                # Display Score + Strengths/Improvements
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("📊 Resume Score")
                    score = resume_analysis['overall_score']
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = score,
                        title = {'text': "Resume Score"},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 75], 'color': "yellow"},
                                {'range': [75, 100], 'color': "green"}
                            ]
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.subheader("🎯 Key Strengths")
                    for strength in resume_analysis['strengths']:
                        st.write(f"✅ {strength}")
                    st.subheader("⚠️ Areas for Improvement")
                    for improvement in resume_analysis['improvements']:
                        st.write(f"🔧 {improvement}")

                # Detailed breakdown
                st.subheader("📈 Detailed Analysis")
                breakdown_data = []
                for category, score in resume_analysis['detailed_scores'].items():
                    breakdown_data.append({
                        'Category': category.replace('_', ' ').title(),
                        'Score': score
                    })
                df_breakdown = pd.DataFrame(breakdown_data)
                fig = px.bar(df_breakdown, x='Category', y='Score', title="Resume Section Breakdown")
                st.plotly_chart(fig, use_container_width=True)

                # Keyword Matching — Optional Enhancement
                st.subheader("🔍 Keyword Matching (Experimental)")
                predefined_keywords = {
                    "Software Engineer": ["Python", "Git", "Agile", "REST", "Unit Testing"],
                    "Data Scientist": ["Python", "Pandas", "Machine Learning", "SQL", "EDA"],
                    "Product Manager": ["Roadmap", "Stakeholder", "Agile", "MVP", "Metrics"],
                    "Marketing Manager": ["SEO", "Campaign", "Brand", "Conversion", "Analytics"]
                }
                role_keywords = predefined_keywords.get(target_role, [])
                found_keywords = [kw for kw in role_keywords if kw.lower() in resume_text.lower()]
                missing_keywords = list(set(role_keywords) - set(found_keywords))

                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Found Keywords:**")
                    for kw in found_keywords:
                        st.write(f"✅ {kw}")
                with col2:
                    st.write("**Missing Keywords:**")
                    for kw in missing_keywords:
                        st.write(f"❌ {kw}")

                # Suggestions
                st.subheader("💡 Improvement Suggestions")
                for suggestion in resume_analysis['suggestions']:
                    st.info(f"💡 {suggestion}")

    with tab2:
        st.title("🏢 Company Research")
        st.subheader("🔍 Research Target Company")

        company_name = st.text_input(
            "Enter company name:",
            placeholder="e.g., Google, Microsoft, Amazon"
        )

        if st.button("🔍 Research Company") and company_name:
            with st.spinner(f"Researching {company_name}..."):
                insights = generate_company_insights(company_name)
                st.success(f"Research complete for {company_name}!")
                st.markdown(insights)

            # Optional Company-Specific Practice Launch
            st.subheader("🎤 Company-Specific Practice")
            if st.button(f"🚀 Start {company_name} Practice Interview"):
                st.session_state.target_company = company_name
                st.session_state.page = "🎤 Practice Interview"
                st.rerun()


def show_settings():
    st.title("⚙️ Settings")
    
    # Notification settings
    with st.expander("🔔 Notification Settings"):
        daily_reminders = st.checkbox("Daily Practice Reminders", value=True)
        achievement_notifications = st.checkbox("Achievement Notifications", value=True)
        progress_updates = st.checkbox("Weekly Progress Updates", value=True)
    
    # Data and privacy
    with st.expander("🔐 Data & Privacy"):
        st.write("**Data Usage:**")
        st.write("• Your interview practice data is stored locally")
        st.write("• No personal information is shared with third parties")
        st.write("• You can export or delete your data at any time")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export Data"):
                st.info("Data export feature would be implemented here")
        
        with col2:
            if st.button("🗑️ Delete All Data"):
                st.error("Data deletion feature would be implemented here")
    
    with st.expander("🔗 Integrations"):
        st.write("Connect with other services:")
        
        # Calendar integration
        cal_integration = st.checkbox("Google Calendar Integration",
                                    help="Sync practice sessions with your calendar")
        if cal_integration:
            st.button("Connect Google Calendar")
        
        # LinkedIn integration
        li_integration = st.checkbox("LinkedIn Profile Import",
                                help="Import your experience from LinkedIn")
        if li_integration:
            st.button("Connect LinkedIn Account")
        
        # Job platforms
        job_alerts = st.checkbox("Job Platform Integrations",
                            help="Get alerts when new jobs match your profile")

    # Save settings
    if st.button("💾 Save Settings"):
        st.success("Settings saved successfully!")

# Helper functions
def calculate_practice_streak(sessions):
    """Calculate consecutive days of practice"""
    if not sessions:
        return 0
    
    dates = [datetime.fromisoformat(session.created_at).date() for session in sessions]
    dates = sorted(set(dates), reverse=True)
    
    if not dates or dates[0] != datetime.now().date():
        return 0
    
    streak = 1
    for i in range(1, len(dates)):
        if (dates[i-1] - dates[i]).days == 1:
            streak += 1
        else:
            break
    
    return streak

def filter_sessions_by_time(sessions, time_period):
    """Filter sessions based on time period"""
    now = datetime.now()
    
    if time_period == "Last 7 days":
        cutoff = now - timedelta(days=7)
    elif time_period == "Last 30 days":
        cutoff = now - timedelta(days=30)
    elif time_period == "Last 90 days":
        cutoff = now - timedelta(days=90)
    else:  # All time
        return sessions
    
    return [s for s in sessions if datetime.fromisoformat(s.created_at) >= cutoff]

def calculate_improvement(sessions):
    """Calculate improvement between first and last sessions"""
    if len(sessions) < 2:
        return 0
    
    first_score = json.loads(sessions[-1].scores).get('overall_score', 0)
    last_score = json.loads(sessions[0].scores).get('overall_score', 0)
    
    return last_score - first_score

def calculate_skill_averages(sessions):
    """Calculate average scores for each skill"""
    if not sessions:
        return {}
    
    skills = ['communication_clarity', 'technical_depth', 'behavioral_examples', 
              'confidence_presence', 'question_relevance']
    
    skill_averages = {}
    for skill in skills:
        scores = [json.loads(s.scores).get(skill, 0) for s in sessions]
        skill_averages[skill.replace('_', ' ').title()] = sum(scores) / len(scores)
    
    return skill_averages


if __name__ == "__main__":
    main()
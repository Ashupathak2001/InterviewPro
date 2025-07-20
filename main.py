import streamlit as st
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
import random


# Enhanced imports for new features
from textblob import TextBlob
import requests
import PyPDF2
from io import BytesIO
import asyncio
import websockets
from concurrent.futures import ThreadPoolExecutor
import logging
from speech_recognition import Recognizer, Microphone, WaitTimeoutError, UnknownValueError, RequestError
from utils.llm_feedback import get_feedback, default_feedback
from utils.research import generate_company_insights
from utils.resume_analysis import analyze_resume_text
from utils.resume_text_extractor import extract_text_from_file
from utils.coding_coach import show_coding_practice_page
from utils.questions_handler import QuestionsHandler, QuestionCategory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ACHIEVEMENTS = {
    'first_interview': {'name': 'Getting Started', 'points': 10, 'description': 'Complete your first practice interview'},
    'perfect_score': {'name': 'Ace Performer', 'points': 100, 'description': 'Score 90+ on an interview'},
    'week_streak': {'name': 'Consistent Learner', 'points': 50, 'description': 'Practice for 7 days straight'},
    'improvement_master': {'name': 'Growth Mindset', 'points': 75, 'description': 'Improve score by 20+ points'},
    'industry_expert': {'name': 'Industry Expert', 'points': 200, 'description': 'Complete 50 interviews in one industry'},
    'mentor': {'name': 'Mentor', 'points': 150, 'description': 'Help 5 other users through peer practice'}
}

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
        self.audio_queue = queue.Queue()
        self.transcription_queue = queue.Queue()
        self.current_transcription = ""
        self.lock = threading.Lock()
        
        
        # Optimize recognizer settings
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        
        # Set recognition parameters
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        self.recognizer.operation_timeout = None
        self.recognizer.phrase_threshold = 0.3
        self.recognizer.non_speaking_duration = 0.8
    
    def start_recording(self):
        """Start continuous speech recognition"""
        self.is_recording = True
        self.current_transcription = ""
        
        # Start background thread for audio capture
        self.audio_thread = threading.Thread(target=self._audio_capture_loop)
        self.audio_thread.daemon = True
        self.audio_thread.start()
        
        # Start background thread for transcription
        self.transcription_thread = threading.Thread(target=self._transcription_loop)
        self.transcription_thread.daemon = True
        self.transcription_thread.start()
    
    def stop_recording(self):
        """Stop recording and return final transcription"""
        
        self.is_recording = False
        
        # Wait for threads to finish
        if hasattr(self, 'audio_thread'):
            self.audio_thread.join(timeout=2)
        if hasattr(self, 'transcription_thread'):
            self.transcription_thread.join(timeout=2)
        
        return self.current_transcription
    
    def _audio_capture_loop(self):
        """Continuous audio capture loop"""
        try:
            with self.microphone as source:
                while self.is_recording:
                    try:
                        # Listen for audio with timeout
                        audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                        self.audio_queue.put(audio)
                    except WaitTimeoutError:
                        continue
                    except Exception as e:
                        logger.error(f"Audio capture error: {e}")
                        break
        except Exception as e:
            logger.error(f"Audio capture loop error: {e}")
    
    def _transcription_loop(self):
        """Continuous transcription loop"""
        while self.is_recording:
            try:
                with self.lock:
                # Get audio from queue
                    audio = self.audio_queue.get(timeout=1)
                    
                    # Transcribe audio
                    text = self._transcribe_audio(audio)
                    if text:
                        self.current_transcription += " " + text
                        self.transcription_queue.put(text)
                    
                    self.audio_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Transcription error: {e}")
    
    def _transcribe_audio(self, audio):
        """Transcribe audio chunk"""
        try:
            # Try Google Speech Recognition first
            try:
                text = self.recognizer.recognize_google(audio, language='en-US')
                return text
            except UnknownValueError:
                # Try with alternative service
                try:
                    text = self.recognizer.recognize_sphinx(audio)
                    return text
                except:
                    return ""
            except RequestError as e:
                logger.error(f"Google Speech Recognition error: {e}")
                return ""
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""
    
    def get_real_time_transcription(self):
        """Get current transcription for real-time display"""
        return self.current_transcription.strip()
    

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


@dataclass
class PracticeSession:
    id: str
    user_id: str
    question: str
    transcription: str
    scores: Dict[str, float]
    feedback: str
    duration: int
    industry: str
    question_type: str
    created_at: datetime
    
@dataclass
class UserProfile:
    id: str
    email: str
    target_role: str
    industry: str
    experience_level: str
    total_points: int
    achievements: List[str]
    learning_path: Dict[str, Any]
    created_at: datetime

class DatabaseManager:
    def __init__(self, db_path="interviewpro.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE,
                target_role TEXT,
                industry TEXT,
                experience_level TEXT,
                total_points INTEGER DEFAULT 0,
                achievements TEXT,
                learning_path TEXT,
                created_at TIMESTAMP
            )
        """)
        
        # Practice sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS practice_sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                question TEXT,
                transcription TEXT,
                scores TEXT,
                feedback TEXT,
                duration INTEGER,
                industry TEXT,
                question_type TEXT,
                created_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        
        # Achievements table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_achievements (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                achievement_type TEXT,
                earned_at TIMESTAMP,
                points INTEGER,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                name TEXT,
                email TEXT,
                level TEXT,
                roles TEXT,
                industries TEXT
            )
        """)

        
        conn.commit()
        conn.close()
    
    def save_user(self, user: UserProfile):
        """Save user profile to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO users 
            (id, email, target_role, industry, experience_level, total_points, achievements, learning_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user.id, user.email, user.target_role, user.industry, user.experience_level,
            user.total_points, json.dumps(user.achievements), json.dumps(user.learning_path),
            user.created_at.isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def save_session(self, session: PracticeSession):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO practice_sessions 
                (id, user_id, question, transcription, scores, feedback, duration, industry, question_type, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session.id, session.user_id, session.question, session.transcription,
                json.dumps(session.scores), session.feedback, session.duration,
                session.industry, session.question_type, session.created_at.isoformat()
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error saving session: {e}")
            # Optionally, show a Streamlit error if in UI context
    
    def get_user_sessions(self, user_id: str) -> List[PracticeSession]:
        """Get all sessions for a user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM practice_sessions WHERE user_id = ? ORDER BY created_at DESC
        """, (user_id,))
        
        sessions = []
        for row in cursor.fetchall():
            sessions.append(PracticeSession(
                id=row[0], user_id=row[1], question=row[2], transcription=row[3],
                scores=json.loads(row[4]), feedback=row[5], duration=row[6],
                industry=row[7], question_type=row[8], created_at=datetime.fromisoformat(row[9])
            ))
        
        conn.close()
        return sessions
    
    def get_user_achievements(self, user_id: str) -> Dict[str, datetime]:
        """Fetch user's achievements as a dictionary {achievement_type: earned_at}"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT achievement_type, earned_at FROM user_achievements WHERE user_id = ?
        """, (user_id,))
        
        rows = cursor.fetchall()
        conn.close()

        return {
            row[0]: datetime.fromisoformat(row[1]) for row in rows
        }

    def save_user_achievement(self, user_id: str, achievement_type: str, earned_at: datetime, points: int = 50):
        """Save a new achievement for the user if not already earned"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if already exists
        cursor.execute("""
            SELECT 1 FROM user_achievements WHERE user_id = ? AND achievement_type = ?
        """, (user_id, achievement_type))
        exists = cursor.fetchone()

        if not exists:
            achievement_id = f"{user_id}_{achievement_type}"
            cursor.execute("""
                INSERT INTO user_achievements (id, user_id, achievement_type, earned_at, points)
                VALUES (?, ?, ?, ?, ?)
            """, (
                achievement_id, user_id, achievement_type,
                earned_at.isoformat(), points
            ))
            
            # Optional: Add points to user
            cursor.execute("""
                UPDATE users SET total_points = total_points + ? WHERE id = ?
            """, (points, user_id))

        conn.commit()
        conn.close()

    def save_user_profile(self, user_id, name, email, level, roles, industries):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("""
            INSERT OR REPLACE INTO user_profiles (user_id, name, email, level, roles, industries)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (user_id, name, email, level, json.dumps(roles), json.dumps(industries)))
        self.conn.commit()
        self.conn.close()

    def load_user_profile(self, user_id):
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.execute("SELECT * FROM user_profiles WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        self.conn.close()
        if row:
            return {
                "user_id": row[0],
                "name": row[1],
                "email": row[2],
                "level": row[3],
                "roles": json.loads(row[4]),
                "industries": json.loads(row[5])
            }
        return None


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
            'speaking_too_fast': "🐌 Try speaking a bit slower - your pace is quite fast",
            'too_many_filler_words': "🎯 Watch out for filler words like 'um', 'uh', 'like'",
            'good_eye_contact': "👀 Great eye contact! Keep it up",
            'be_specific': "📝 Try to be more specific - add concrete examples",
            'good_structure': "✅ Nice structure in your answer!",
            'confidence_boost': "💪 You're doing great! Stay confident",
            'almost_done': "⏰ You have about 15 seconds left to wrap up"
        }
        
    def start_session(self):
        """Start real-time interview session"""
        self.is_active = True
        self.speech_manager.start_recording()
        self.vision_analyzer.start_analysis()
        
    def stop_session(self):
        """Stop real-time interview session"""
        self.is_active = False
        transcription = self.speech_manager.stop_recording()
        self.vision_analyzer.stop_analysis()
        
        return transcription
    
    def process_frame(self, frame):
        """Process video frame and return annotated frame"""
        if not self.is_active:
            return frame
            
        with self.coaching_lock:
        # Analyze frame
            metrics = self.vision_analyzer.analyze_frame(frame)
            
            if metrics:
                self.metrics_queue.put(metrics)
                
                # Generate real-time feedback
                feedback = self.vision_analyzer.get_real_time_feedback()
                for fb in feedback:
                    self.feedback_queue.put(fb)
            
            # Annotate frame with feedback
            annotated_frame = self._annotate_frame(frame, metrics)
            
            return annotated_frame
    
    def _annotate_frame(self, frame, metrics):
        """Annotate frame with real-time feedback"""
        if not metrics:
            return frame
            
        # Add metrics overlay
        overlay = frame.copy()
        
        # Eye contact indicator
        eye_color = (0, 255, 0) if metrics.eye_contact_score > 70 else (0, 165, 255)
        cv2.circle(overlay, (30, 30), 10, eye_color, -1)
        cv2.putText(overlay, f"Eye Contact: {metrics.eye_contact_score:.0f}%", 
                   (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, eye_color, 2)
        
        # Posture indicator
        posture_color = (0, 255, 0) if metrics.posture_score > 70 else (0, 165, 255)
        cv2.circle(overlay, (30, 60), 10, posture_color, -1)
        cv2.putText(overlay, f"Posture: {metrics.posture_score:.0f}%", 
                   (50, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, posture_color, 2)
        
        # Gesture indicator
        gesture_color = (0, 255, 0) if metrics.gesture_appropriateness > 70 else (0, 165, 255)
        cv2.circle(overlay, (30, 90), 10, gesture_color, -1)
        cv2.putText(overlay, f"Gestures: {metrics.gesture_appropriateness:.0f}%", 
                   (50, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gesture_color, 2)
        
        return overlay
    
        
    def get_real_time_transcription(self):
        """Get current transcription"""
        return self.speech_manager.get_real_time_transcription()
    
    def get_real_time_feedback(self):
        """Get real-time feedback"""
        feedback = []
        while not self.feedback_queue.empty():
            try:
                feedback.append(self.feedback_queue.get_nowait())
            except queue.Empty:
                break
        return feedback
    
    def analyze_real_time_performance(self, transcription: str, elapsed_time: int) -> str:
        """Analyze current performance and provide real-time feedback"""
        if not transcription:
            return "🎤 Start speaking - I'm listening!"
        
        words = transcription.split()
        words_per_minute = len(words) / (elapsed_time / 60) if elapsed_time > 0 else 0
        
        # Check speaking pace
        if words_per_minute > 180:
            return self.coaching_messages['speaking_too_fast']
        
        # Check for filler words
        filler_words = ['um', 'uh', 'like', 'you know', 'basically', 'actually']
        filler_count = sum(transcription.lower().count(word) for word in filler_words)
        if filler_count > len(words) * 0.1:
            return self.coaching_messages['too_many_filler_words']
        
        # Check for specificity
        vague_words = ['something', 'things', 'stuff', 'various', 'many']
        if any(word in transcription.lower() for word in vague_words):
            return self.coaching_messages['be_specific']
        
        # Positive reinforcement
        if elapsed_time > 45:
            return self.coaching_messages['almost_done']
        
        return self.coaching_messages['confidence_boost']

class GamificationSystem:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def check_achievements(self, user_id: str, session: PracticeSession) -> List[str]:
        """Check and award achievements based on performance"""
        new_achievements = []
        user_sessions = self.db.get_user_sessions(user_id)
        
        # First interview achievement
        if len(user_sessions) == 1:
            new_achievements.append('first_interview')
        
        # Perfect score achievement
        overall_score = session.scores.get('overall_score', 0)
        if overall_score >= 90:
            new_achievements.append('perfect_score')
        
        # Week streak achievement
        if len(user_sessions) >= 7:
            recent_dates = [s.created_at.date() for s in user_sessions[-7:]]
            if len(set(recent_dates)) == 7:
                new_achievements.append('week_streak')
        
        # Improvement achievement
        if len(user_sessions) >= 5:
            first_scores = [s.scores.get('overall_score', 0) for s in user_sessions[-5:]]
            if first_scores[-1] - first_scores[0] >= 20:
                new_achievements.append('improvement_master')
        
        return new_achievements
    
    def get_user_achievements(self, user_id: str) -> List[str]:
        """Fetch existing achievements for the user from the database"""
        return self.db.get_user_achievements(user_id) or []
    
    def get_user_stats(self, user_id: str) -> Dict[str, int]:
        sessions = self.db.get_user_sessions(user_id)
        xp_per_session = 100
        total_xp = len(sessions) * xp_per_session

        level = total_xp // 500
        current_level_xp = total_xp % 500
        xp_to_next_level = 500 - current_level_xp

        return {
            "level": level,
            "total_xp": total_xp,
            "xp_to_next_level": xp_to_next_level,
            "current_level_xp": current_level_xp,
            "xp_for_current_level": 500
        }


class PersonalizedLearningPath:
    def __init__(self):
        self.skill_areas = [
            'communication_clarity',
            'technical_depth',
            'behavioral_examples',
            'confidence_presence',
            'question_relevance',
            'focus_areas',
            'target_improvements'
        ]
    
    def analyze_performance_gaps(self, user_sessions: List[PracticeSession]) -> Dict[str, float]:
        """Analyze user performance to identify gaps"""
        if not user_sessions:
            return {skill: 0.5 for skill in self.skill_areas}
        
        # Calculate average scores for each skill area
        skill_scores = {skill: [] for skill in self.skill_areas}
        
        for session in user_sessions:
            for skill in self.skill_areas:
                if skill in session.scores:
                    skill_scores[skill].append(session.scores[skill])
        
        # Calculate averages
        skill_averages = {}
        for skill, scores in skill_scores.items():
            skill_averages[skill] = sum(scores) / len(scores) if scores else 0.5
        
        return skill_averages
    
    def create_learning_plan(self, skill_scores: Dict[str, float], target_role: str) -> Dict[str, Any]:
        """Create personalized learning plan"""
        weak_areas = sorted(skill_scores.items(), key=lambda x: x[1])[:3]
        
        learning_plan = {
            'focus_areas': [area[0] for area in weak_areas],
            'recommended_practice': self.get_practice_recommendations(weak_areas),
            'target_improvements': {area[0]: min(area[1] + 0.3, 1.0) for area in weak_areas},
            'estimated_timeline': '2-3 weeks with daily practice'
        }
        
        return learning_plan
    
    def get_practice_recommendations(self, weak_areas: List[tuple]) -> List[str]:
        """Get specific practice recommendations for weak areas"""
        recommendations = []
        
        for area, score in weak_areas:
            if area == 'communication_clarity':
                recommendations.append("Practice speaking slowly and clearly")
                recommendations.append("Record yourself and listen for filler words")
            elif area == 'technical_depth':
                recommendations.append("Prepare detailed examples of your technical work")
                recommendations.append("Practice explaining complex concepts simply")
            elif area == 'behavioral_examples':
                recommendations.append("Use the STAR method for behavioral questions")
                recommendations.append("Prepare 5-7 strong examples from your experience")
            elif area == 'confidence_presence':
                recommendations.append("Practice power poses before interviews")
                recommendations.append("Work on maintaining eye contact")
            elif area == 'question_relevance':
                recommendations.append("Listen carefully to questions before answering")
                recommendations.append("Ask clarifying questions when needed")
        
        return recommendations
    
    def generate_daily_practice_plan(self, skill_scores: dict, learning_plan: dict) -> dict:
        """Generate daily practice plan based on the learning plan and current scores"""
        plan = {}
        focus_areas = learning_plan.get("focus_areas", [])
        target_improvements = learning_plan.get("target_improvements", {})

        for skill in focus_areas:
            current = skill_scores.get(skill, 0.0)
            target = target_improvements.get(skill, 1.0)
            gap = max(0, target - current)

            if gap > 0.05:  # >5% gap
                plan[skill] = f"Practice {skill.replace('_', ' ').title()} to improve by {gap * 100:.1f}%"

        return plan


class ResumeAnalyzer:
    def __init__(self):
        self.skills_keywords = {
            'technical': ['python', 'java', 'javascript', 'react', 'node.js', 'sql', 'aws', 'docker'],
            'leadership': ['lead', 'manage', 'mentor', 'direct', 'coordinate', 'supervise'],
            'analytical': ['analyze', 'data', 'metrics', 'optimize', 'improve', 'measure']
        }
    
    def parse_resume(self, resume_text: str) -> Dict[str, Any]:
        """Parse resume and extract key information"""
        resume_data = {
            'experiences': self.extract_experiences(resume_text),
            'skills': self.extract_skills(resume_text),
            'education': self.extract_education(resume_text),
            'achievements': self.extract_achievements(resume_text)
        }
        
        return resume_data
    
    def extract_experiences(self, text: str) -> List[Dict[str, str]]:
        """Extract work experiences from resume text"""
        experiences = []
        
        # Look for company names and roles (simplified pattern)
        company_pattern = r'([A-Z][a-z]+ ?[A-Z]?[a-z]*)\s*[-–—]\s*([A-Z][a-z]+ ?[A-Z]?[a-z]*)'
        matches = re.findall(company_pattern, text)
        
        for match in matches:
            experiences.append({
                'company': match[0],
                'role': match[1],
                'description': 'Experience at ' + match[0]
            })
        
        return experiences[:5]  # Limit to 5 most recent
    
    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from resume text"""
        found_skills = []
        text_lower = text.lower()
        
        for category, skills in self.skills_keywords.items():
            for skill in skills:
                if skill in text_lower:
                    found_skills.append(skill)
        
        return found_skills
    
    def extract_education(self, text: str) -> List[str]:
        """Extract education information"""
        education_keywords = ['university', 'college', 'degree', 'bachelor', 'master', 'phd']
        education = []
        
        for keyword in education_keywords:
            if keyword in text.lower():
                education.append(f"Education includes {keyword}")
        
        return education
    
    def extract_achievements(self, text: str) -> List[str]:
        """Extract achievements and accomplishments"""
        achievement_indicators = ['increased', 'improved', 'reduced', 'achieved', 'led', 'delivered']
        achievements = []
        
        sentences = text.split('.')
        for sentence in sentences:
            for indicator in achievement_indicators:
                if indicator in sentence.lower():
                    achievements.append(sentence.strip())
                    break
        
        return achievements[:3]  # Top 3 achievements
    
    def generate_resume_based_questions(self, resume_data: Dict[str, Any]) -> List[str]:
        """Generate questions based on resume content"""
        questions = []
        
        # Questions based on experiences
        for exp in resume_data['experiences']:
            questions.append(f"Tell me about your role at {exp['company']}")
            questions.append(f"What was your biggest achievement as {exp['role']}?")
        
        # Questions based on skills
        for skill in resume_data['skills'][:3]:
            questions.append(f"Describe a project where you used {skill}")
        
        # Questions based on achievements
        for achievement in resume_data['achievements']:
            questions.append(f"Tell me more about this achievement: {achievement[:50]}...")
        
        return questions

class CompanyResearcher:
    def __init__(self):
        self.company_data_cache = {}
    
    def research_company(self, company_name: str) -> Dict[str, Any]:
        """Research company information (simplified version)"""
        if company_name in self.company_data_cache:
            return self.company_data_cache[company_name]
        
        # In a real implementation, this would scrape company websites, news, etc.
        # For demo purposes, we'll return mock data
        company_data = {
            'values': ['innovation', 'customer-focus', 'integrity', 'teamwork'],
            'recent_news': [
                f"{company_name} launches new product line",
                f"{company_name} expands to new markets",
                f"{company_name} receives industry award"
            ],
            'culture': {
                'dress_code': 'business casual',
                'work_style': 'collaborative',
                'values': 'innovation and customer focus'
            },
            'interview_process': {
                'rounds': 3,
                'typical_duration': '45 minutes each',
                'includes_technical': True
            }
        }
        
        self.company_data_cache[company_name] = company_data
        return company_data
    
    def generate_company_questions(self, company_data: Dict[str, Any]) -> List[str]:
        """Generate company-specific questions"""
        questions = []
        
        # Culture-based questions
        for value in company_data['values']:
            questions.append(f"Give me an example of how you demonstrate {value}")
        
        # Recent news questions
        questions.append("What do you know about our recent developments?")
        questions.append("How do you see yourself contributing to our growth?")
        
        return questions

class MultiModalAnalyzer:
    def __init__(self):
        self.confidence_indicators = {
            'voice_pace': 0.0,
            'eye_contact': 0.0,
            'body_language': 0.0,
            'facial_expressions': 0.0
        }
    
    def analyze_comprehensive_performance(self, 
                                        transcription: str, 
                                        duration: int,
                                        simulated_video_data: Dict = None) -> Dict[str, Any]:
        """Comprehensive analysis of multiple modalities"""
        
        # Text analysis
        text_analysis = self.analyze_text_quality(transcription)
        
        # Simulated video analysis (in real implementation, use computer vision)
        video_analysis = self.simulate_video_analysis(duration)
        
        # Combine analyses
        comprehensive_score = {
            'text_quality': text_analysis['quality_score'],
            'communication_clarity': text_analysis['clarity_score'],
            'confidence_indicators': video_analysis['confidence_score'],
            'engagement_level': video_analysis['engagement_score'],
            'overall_impression': (text_analysis['quality_score'] + 
                                 video_analysis['confidence_score'] + 
                                 video_analysis['engagement_score']) / 3
        }
        
        return {
            'scores': comprehensive_score,
            'detailed_analysis': {
                'text_insights': text_analysis,
                'video_insights': video_analysis
            }
        }
    
    def analyze_text_quality(self, transcription: str) -> Dict[str, Any]:
        """Analyze text quality and content"""
        if not transcription:
            return {'quality_score': 0, 'clarity_score': 0}
        
        # Basic text analysis
        words = transcription.split()
        sentences = transcription.split('.')
        
        # Quality metrics
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        filler_words = ['um', 'uh', 'like', 'you know']
        filler_count = sum(transcription.lower().count(word) for word in filler_words)
        
        # Sentiment analysis
        blob = TextBlob(transcription)
        sentiment = blob.sentiment.polarity
        
        quality_score = max(0, min(100, 
            100 - (filler_count * 5) + (sentiment * 10) + 
            (min(avg_sentence_length, 20) * 2)
        ))
        
        clarity_score = max(0, min(100, 
            100 - (filler_count * 10) + (len(words) * 0.5)
        ))
        
        return {
            'quality_score': quality_score,
            'clarity_score': clarity_score,
            'sentiment': sentiment,
            'word_count': len(words),
            'filler_count': filler_count
        }
    
    def simulate_video_analysis(self, duration: int) -> Dict[str, Any]:
        """Simulate video analysis (placeholder for real computer vision)"""
        # In real implementation, this would use computer vision libraries
        # For demo, we'll simulate realistic scores
        
        confidence_score = np.random.uniform(60, 90)
        engagement_score = np.random.uniform(70, 95)
        
        return {
            'confidence_score': confidence_score,
            'engagement_score': engagement_score,
            'eye_contact_percentage': np.random.uniform(60, 85),
            'speaking_pace': 'appropriate' if duration > 30 else 'too fast',
            'body_language': 'confident' if confidence_score > 75 else 'needs improvement'
        }

class PredictiveAnalyzer:
    def __init__(self):
        self.success_factors = {
            'communication_clarity': 0.25,
            'technical_depth': 0.20,
            'behavioral_examples': 0.20,
            'confidence_presence': 0.15,
            'question_relevance': 0.20
        }
    
    def predict_interview_success(self, 
                                user_sessions: List[PracticeSession], 
                                target_company: str = None) -> Dict[str, Any]:
        """Predict interview success probability"""
        
        if not user_sessions:
            return {
                'success_probability': 0.5,
                'confidence_level': 'low',
                'key_factors': ['Need more practice data'],
                'recommendations': ['Complete more practice sessions']
            }
        
        # Calculate weighted score based on recent performance
        recent_sessions = user_sessions[-5:]  # Last 5 sessions
        
        weighted_scores = []
        for session in recent_sessions:
            session_score = 0
            for factor, weight in self.success_factors.items():
                if factor in session.scores:
                    session_score += session.scores[factor] * weight
            weighted_scores.append(session_score)
        
        # Calculate success probability
        avg_performance = sum(weighted_scores) / len(weighted_scores)
        success_probability = min(0.95, max(0.05, avg_performance / 100))
        
        # Determine confidence level
        if success_probability >= 0.8:
            confidence_level = 'high'
        elif success_probability >= 0.6:
            confidence_level = 'medium'
        else:
            confidence_level = 'low'
        
        # Identify key factors
        key_factors = self.identify_key_factors(recent_sessions)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(recent_sessions, success_probability)
        
        return {
            'success_probability': success_probability,
            'confidence_level': confidence_level,
            'key_factors': key_factors,
            'recommendations': recommendations,
            'improvement_areas': self.identify_improvement_areas(recent_sessions)
        }
    
    def identify_key_factors(self, sessions: List[PracticeSession]) -> List[str]:
        """Identify key factors affecting performance"""
        factor_scores = {}
        
        for session in sessions:
            for factor in self.success_factors:
                if factor in session.scores:
                    if factor not in factor_scores:
                        factor_scores[factor] = []
                    factor_scores[factor].append(session.scores[factor])
        
        # Calculate averages and identify strengths
        strengths = []
        for factor, scores in factor_scores.items():
            avg_score = sum(scores) / len(scores)
            if avg_score >= 80:
                strengths.append(factor.replace('_', ' ').title())
        
        return strengths or ['Building foundation skills']
    
    def identify_improvement_areas(self, sessions: List[PracticeSession]) -> List[str]:
        """Identify areas needing improvement"""
        factor_scores = {}
        
        for session in sessions:
            for factor in self.success_factors:
                if factor in session.scores:
                    if factor not in factor_scores:
                        factor_scores[factor] = []
                    factor_scores[factor].append(session.scores[factor])
        
        # Calculate averages and identify weak areas
        weak_areas = []
        for factor, scores in factor_scores.items():
            avg_score = sum(scores) / len(scores)
            if avg_score < 70:
                weak_areas.append(factor.replace('_', ' ').title())
        
        return weak_areas or ['Continue practicing all areas']
    
    def generate_recommendations(self, sessions: List[PracticeSession], success_prob: float) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []
        
        if success_prob < 0.6:
            recommendations.append("Focus on daily practice sessions")
            recommendations.append("Work on specific examples and storytelling")
        elif success_prob < 0.8:
            recommendations.append("Polish your strongest areas")
            recommendations.append("Practice with industry-specific questions")
        else:
            recommendations.append("You're well-prepared! Focus on confidence building")
            recommendations.append("Practice with mock interviews similar to your target company")
        
        return recommendations

class AIInterviewBuddy:
    def __init__(self):
        self.conversation_history = []
        self.difficulty_level = 'medium'
    
    def generate_follow_up_question(self, previous_answer: str, base_question: str) -> str:
        """Generate natural follow-up questions based on user's answer"""
        if not previous_answer:
            return base_question
        
        # Analyze the answer for follow-up opportunities
        words = previous_answer.lower().split()
        
        # Look for specific topics to drill down
        if 'project' in words:
            return "That's interesting! Can you tell me more about the specific challenges you faced in that project?"
        elif 'team' in words:
            return "How did you handle team dynamics and communication during that experience?"
        elif 'technical' in words or 'technology' in words:
            return "What was the most technically challenging aspect of that work?"
        elif 'leadership' in words or 'lead' in words:
            return "How did you motivate your team members and ensure everyone stayed aligned?"
        elif 'problem' in words:
            return "Walk me through your problem-solving process step by step."
        else:
            return "That's great! Can you give me a specific example of how you applied that in a real situation?"
    
    def adjust_difficulty(self, performance_score: float):
        """Adjust difficulty based on user performance"""
        if performance_score > 85:
            self.difficulty_level = 'hard'
        elif performance_score < 60:
            self.difficulty_level = 'easy'
        else:
            self.difficulty_level = 'medium'
    
    def provide_encouragement(self, performance_score: float) -> str:
        """Provide appropriate encouragement based on performance"""
        if performance_score >= 85:
            return "🌟 Excellent! You're really showing strong expertise in this area."
        elif performance_score >= 70:
            return "👍 Good job! You're on the right track. Let's keep building on this."
        elif performance_score >= 50:
            return "💪 You're making progress! Don't worry, this takes practice."
        else:
            return "🎯 Remember, every expert was once a beginner. You're learning!"

def analyze_interview_performance(transcription: str, question: str, question_type: str) -> Dict[str, float]:
    """Analyze interview performance and return scores"""
    if not transcription:
        return {
            'communication_clarity': 0,
            'technical_depth': 0,
            'behavioral_examples': 0,
            'confidence_presence': 0,
            'question_relevance': 0,
            'overall_score': 0
        }
    
    # Basic analysis metrics
    words = transcription.split()
    word_count = len(words)
    
    # Communication clarity (based on word count, filler words, etc.)
    filler_words = ['um', 'uh', 'like', 'you know', 'basically']
    filler_count = sum(transcription.lower().count(word) for word in filler_words)
    clarity_score = max(0, min(100, 100 - (filler_count * 10) + (word_count * 0.5)))
    
    # Technical depth (look for technical terms)
    technical_terms = ['system', 'architecture', 'database', 'algorithm', 'framework', 'api', 'performance', 'scalability']
    tech_score = min(100, sum(10 for term in technical_terms if term in transcription.lower()))
    
    # Behavioral examples (look for story structure)
    behavioral_indicators = ['when', 'situation', 'challenge', 'result', 'experience', 'example']
    behavioral_score = min(100, sum(15 for indicator in behavioral_indicators if indicator in transcription.lower()))
    
    # Confidence presence (sentence length, positive language)
    avg_sentence_length = word_count / len(transcription.split('.')) if '.' in transcription else word_count
    confidence_score = min(100, max(30, avg_sentence_length * 3))
    
    # Question relevance (keyword matching)
    question_words = set(question.lower().split())
    answer_words = set(transcription.lower().split())
    relevance_score = len(question_words.intersection(answer_words)) / len(question_words) * 100
    
    # Calculate overall score
    scores = {
        'communication_clarity': clarity_score,
        'technical_depth': tech_score if question_type == 'Technical' else behavioral_score,
        'behavioral_examples': behavioral_score,
        'confidence_presence': confidence_score,
        'question_relevance': relevance_score
    }
    
    overall_score = sum(scores.values()) / len(scores)
    scores['overall_score'] = overall_score
    
    return scores

def generate_detailed_feedback(scores: Dict[str, float], transcription: str) -> str:
    """Generate detailed feedback based on performance scores"""
    feedback = []
    
    # Communication clarity feedback
    if scores['communication_clarity'] >= 80:
        feedback.append("✅ Excellent communication clarity! Your speech was clear and well-structured.")
    elif scores['communication_clarity'] >= 60:
        feedback.append("👍 Good communication overall. Try to reduce filler words for even better clarity.")
    else:
        feedback.append("💡 Focus on speaking more clearly. Practice reducing 'um', 'uh', and other filler words.")
    
    # Technical depth feedback
    if scores['technical_depth'] >= 80:
        feedback.append("🔧 Strong technical depth! You demonstrated solid understanding of the concepts.")
    elif scores['technical_depth'] >= 60:
        feedback.append("📚 Good technical knowledge. Consider adding more specific examples or details.")
    else:
        feedback.append("🎯 Work on providing more technical details and specific examples in your answers.")
    
    # Behavioral examples feedback
    if scores['behavioral_examples'] >= 80:
        feedback.append("📖 Excellent use of examples! Your stories were well-structured and relevant.")
    elif scores['behavioral_examples'] >= 60:
        feedback.append("👥 Good examples provided. Try using the STAR method (Situation, Task, Action, Result).")
    else:
        feedback.append("🌟 Add more specific examples from your experience. Use the STAR method for better structure.")
    
    # Confidence presence feedback
    if scores['confidence_presence'] >= 80:
        feedback.append("💪 You showed great confidence! Your presence was strong and engaging.")
    elif scores['confidence_presence'] >= 60:
        feedback.append("🎭 Good confidence level. Work on maintaining energy throughout your answer.")
    else:
        feedback.append("🚀 Build your confidence by practicing more and preparing strong examples.")
    
    # Question relevance feedback
    if scores['question_relevance'] >= 80:
        feedback.append("🎯 Perfect! Your answer was highly relevant to the question asked.")
    elif scores['question_relevance'] >= 60:
        feedback.append("📍 Good relevance. Make sure to address all parts of the question directly.")
    else:
        feedback.append("🔍 Listen carefully to the question and make sure your answer addresses what's being asked.")
    
    return "\n".join(feedback)

# Initialize session state
if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

# NEW: Try to load existing profile
if 'profile' not in st.session_state:
    db = DatabaseManager()
    profile = db.load_user_profile(st.session_state.user_id)
    if profile:
        st.session_state.profile = profile

if 'db_manager' not in st.session_state:
    st.session_state.db_manager = DatabaseManager()
if 'current_session' not in st.session_state:
    st.session_state.current_session = None
if 'gamification' not in st.session_state:
    st.session_state.gamification = GamificationSystem(st.session_state.db_manager)
if 'learning_path' not in st.session_state:
    st.session_state.learning_path = PersonalizedLearningPath()
if 'real_time_coach' not in st.session_state:
    st.session_state.real_time_coach = RealTimeCoach()
if 'resume_analyzer' not in st.session_state:
    st.session_state.resume_analyzer = ResumeAnalyzer()
if 'company_researcher' not in st.session_state:
    st.session_state.company_researcher = CompanyResearcher()
if 'multimodal_analyzer' not in st.session_state:
    st.session_state.multimodal_analyzer = MultiModalAnalyzer()
if 'predictive_analyzer' not in st.session_state:
    st.session_state.predictive_analyzer = PredictiveAnalyzer()
if 'ai_buddy' not in st.session_state:
    st.session_state.ai_buddy = AIInterviewBuddy()
if 'questions_handler' not in st.session_state:
    st.session_state.questions_handler = QuestionsHandler("data/questions.json")

def main():
    st.set_page_config(
        page_title="InterviewPro - AI Interview Coach",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
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
        <h1>🎯 InterviewPro</h1>
        <p>Your AI-Powered Interview Coach</p>
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
        "📈Practice Summary": "summary",
        "📊 Analytics": "analytics",
        "📄 Extra Help": "extra_help",
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
        "📊 Analytics": show_analytics_achievements,
        "📄 Extra Help": show_extra_help,
        "🏢 Coding": show_coding_practice,
        "⚙️ Settings": show_settings
    }

    # Call the right function
    current_page = st.session_state.page
    page_router.get(current_page, lambda: st.error("Page not found"))()

    # ✅ Footer at the very end
    st.markdown("""
    <div class="footer">
        © 2025 InterviewPro | Made with ❤️ by Ashish Pathak
    </div>
    """, unsafe_allow_html=True)

def show_profile_setup():
    st.title("👤 Candidate Profile")
    
    db = st.session_state.db_manager
    user_id = st.session_state.user_id
    
    # 🔧 FIX: Always check database first for persistent data
    profile = db.load_user_profile(user_id)
    
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
            st.write(f"**Name:** {profile['name']}")
            st.write(f"**Email:** {profile['email']}")
            st.write(f"**Level:** {profile['level']}")
            st.write(f"**Roles:** {', '.join(profile['roles'])}")
            st.write(f"**Industries:** {', '.join(profile['industries'])}")
        
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
            profile = db.load_user_profile(user_id)
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
        db.save_user_profile(user_id, name, email, level, roles, industries)
        
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

def show_dashboard():
    st.title("📊 Your Interview Performance Dashboard")
    
    # Get user sessions
    user_sessions = st.session_state.db_manager.get_user_sessions(st.session_state.user_id)
    
    if not user_sessions:
        st.info("👋 Welcome to InterviewPro! Start with your first practice interview to see your progress here.")
        if st.button("🚀 Start First Interview"):
            st.session_state.page = "🎤 Practice Interview"
            st.rerun()
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Sessions",
            len(user_sessions),
            delta=f"+{len([s for s in user_sessions if s.created_at.date() == datetime.now().date()])} today"
        )
    
    with col2:
        avg_score = sum(s.scores.get('overall_score', 0) for s in user_sessions) / len(user_sessions)
        st.metric(
            "Average Score",
            f"{avg_score:.1f}%",
            delta=f"{avg_score - 50:.1f}%" if avg_score > 50 else f"{avg_score - 50:.1f}%"
        )
    
    with col3:
        recent_score = user_sessions[0].scores.get('overall_score', 0)
        st.metric(
            "Latest Score",
            f"{recent_score:.1f}%",
            delta=f"{recent_score - avg_score:.1f}%" if len(user_sessions) > 1 else None
        )
    
    with col4:
        streak = calculate_practice_streak(user_sessions)
        st.metric(
            "Practice Streak",
            f"{streak} days",
            delta="+1" if streak > 0 else "Start today!"
        )
    
    # Performance trend chart
    st.subheader("📈 Performance Trend")
    
    if len(user_sessions) > 1:
        df = pd.DataFrame([
            {
                'Date': s.created_at.date(),
                'Overall Score': s.scores.get('overall_score', 0),
                'Communication': s.scores.get('communication_clarity', 0),
                'Technical': s.scores.get('technical_depth', 0),
                'Behavioral': s.scores.get('behavioral_examples', 0)
            }
            for s in reversed(user_sessions[-10:])  # Last 10 sessions
        ])
        
        fig = px.line(df, x='Date', y=['Overall Score', 'Communication', 'Technical', 'Behavioral'],
                     title="Performance Trends (Last 10 Sessions)")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent activity
    st.subheader("🕐 Recent Activity")
    
    for session in user_sessions[:5]:  # Show last 5 sessions
        with st.expander(f"📅 {session.created_at.strftime('%Y-%m-%d %H:%M')} - {session.industry} ({session.question_type})"):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Question:** {session.question}")
                st.write(f"**Your Answer:** {session.transcription[:200]}...")
            
            with col2:
                st.metric("Score", f"{session.scores.get('overall_score', 0):.1f}%")
                st.write(f"**Duration:** {session.duration}s")
    
    # Predictive insights
    st.subheader("🔮 Interview Success Prediction")
    prediction = st.session_state.predictive_analyzer.predict_interview_success(user_sessions)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Success Probability",
            f"{prediction['success_probability']:.1%}",
            delta=f"Confidence: {prediction['confidence_level']}"
        )
    
    with col2:
        st.write("**Key Strengths:**")
        for factor in prediction['key_factors']:
            st.write(f"• {factor}")
    
    # Recommendations
    st.subheader("💡 Personalized Recommendations")
    for rec in prediction['recommendations']:
        st.info(f"📌 {rec}")

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
    index = session["current"]
    if index >= len(session["questions"]):
        st.warning("No more questions in this session. Returning to dashboard.")
        st.session_state.page = "🏠 Dashboard"
        st.rerun()
        return
    question_obj = session["questions"][index]
    question_text = get_question_text(question_obj)

    st.title("🎙️ Interview In Progress")
    st.subheader(f"Question {index + 1} of {len(session['questions'])}")
    st.info(f"**Q:** {question_text}")

    if session.get("target_company"):
        company_data = st.session_state.company_researcher.research_company(session["target_company"])
        st.write(f"**Company Context for {session['target_company']}:**")
        st.write(f"• Values: {', '.join(company_data['values'])}")
        st.write(f"• Culture: {company_data['culture']['work_style']}")

    if st.button("🎤 Start Answering"):
        coach = st.session_state.real_time_coach
        coach.speech_manager.start_recording()
        coach.vision_analyzer.start_analysis()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.warning("⚠️ Could not access webcam.")
        else:
            st.info("🔴 Webcam and Mic recording started. Please answer the question.")
            start_time = time.time()
            duration = 60
            stframe = st.empty()

            while time.time() - start_time < duration:
                ret, frame = cap.read()
                if not ret:
                    break
                metrics = coach.vision_analyzer.analyze_frame(frame)
                eye_color = (0, 255, 0) if metrics["eye_contact"] else (0, 0, 255)
                label = "Eye Contact: Good" if metrics["eye_contact"] else "Not Looking: Please look on scrren"
                cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, eye_color, 2)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(Image.fromarray(frame_rgb), channels="RGB")

            cap.release()
            coach.vision_analyzer.stop_analysis()

        transcription = coach.speech_manager.stop_recording()

        session["responses"].append({
            "question": question_text,
            "transcription": transcription,
            "duration": 60
        })

        session["current"] += 1
        if session["current"] >= len(session["questions"]):
            st.session_state.page = "practice_summary"
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

        # --- Step 5: Save to DB ---
        session_obj = PracticeSession(
            id=str(uuid.uuid4()),
            user_id=st.session_state.user_id,
            question=question,
            transcription=transcription,
            scores=scores,
            feedback=feedback_text,
            duration=responses["duration"],
            industry=session["industry"],
            question_type=session["question_type"],
            created_at=datetime.now()
        )
        st.session_state.db_manager.save_session(session_obj)
        sessions_to_save.append(session_obj)

    # --- Step 6: Show Summary & Download ---
    avg_score = sum(r['score'] for r in session["responses"]) / len(session["responses"])
    st.metric("📊 Average Interview Score", f"{avg_score:.1f}%")
    st.download_button("📥 Download Summary as Text", full_report.encode(), file_name="interview_summary.txt")

    if st.button("🏠 Return to Dashboard"):
        del st.session_state.practice_session
        st.session_state.page = "🏠 Dashboard"
        st.rerun()

def show_interview_results(session: PracticeSession):
    st.subheader("📊 Interview Results")
    
    # Overall score
    overall_score = session.scores.get('overall_score', 0)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Create a gauge chart for overall score
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = overall_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Overall Score"},
            delta = {'reference': 75, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 75], 'color': "yellow"},
                    {'range': [75, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed scores
    st.subheader("📈 Detailed Breakdown")
    
    score_data = [
        {'Skill': 'Communication Clarity', 'Score': session.scores.get('communication_clarity', 0)},
        {'Skill': 'Technical Depth', 'Score': session.scores.get('technical_depth', 0)},
        {'Skill': 'Behavioral Examples', 'Score': session.scores.get('behavioral_examples', 0)},
        {'Skill': 'Confidence Presence', 'Score': session.scores.get('confidence_presence', 0)},
        {'Skill': 'Question Relevance', 'Score': session.scores.get('question_relevance', 0)}
    ]
    
    df_scores = pd.DataFrame(score_data)
    
    fig = px.bar(df_scores, x='Skill', y='Score', 
                 title="Skill Breakdown",
                 color='Score',
                 color_continuous_scale='RdYlGn')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Feedback section
    st.subheader("💬 Detailed Feedback")
    st.markdown(f"""
    <div class="feedback-section">
        {session.feedback.replace(chr(10), '<br>')}
    </div>
    """, unsafe_allow_html=True)
    
    # AI Follow-up questions
    st.subheader("🤖 AI Follow-up Questions")
    follow_up = st.session_state.ai_buddy.generate_follow_up_question(session.transcription, session.question)
    st.info(f"**Follow-up:** {follow_up}")
    
    # Encouragement
    encouragement = st.session_state.ai_buddy.provide_encouragement(overall_score)
    st.success(encouragement)
    
    # Check for new achievements
    new_achievements = st.session_state.gamification.check_achievements(st.session_state.user_id, session)
    if new_achievements:
        st.balloons()
        st.success("🎉 New Achievement Unlocked!")
        for achievement in new_achievements:
            st.write(f"🏆 {ACHIEVEMENTS[achievement]['name']}: {ACHIEVEMENTS[achievement]['description']}")

def show_analytics_achievements():
    st.title("📊 Analytics & Achievements")
    
    # Create tabs for the merged content
    tab1, tab2, tab3 = st.tabs(["📈 Performance Analytics", "🏆 Achievements", "🎯 Learning Path"])
    
    with tab1:
        # Copy all content from original show_analytics() function
        user_sessions = st.session_state.db_manager.get_user_sessions(st.session_state.user_id)
        
        if not user_sessions:
            st.info("No practice sessions found. Complete some interviews to see your analytics!")
        else:
            # Time period selector
            time_period = st.selectbox(
                "Select Time Period:",
                ["Last 7 days", "Last 30 days", "Last 90 days", "All time"],
                key="analytics_time_period"
            )
            
            # Filter sessions based on time period
            filtered_sessions = filter_sessions_by_time(user_sessions, time_period)
            
            # Performance metrics
            st.subheader("📈 Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_score = sum(s.scores.get('overall_score', 0) for s in filtered_sessions) / len(filtered_sessions)
                st.metric("Average Score", f"{avg_score:.1f}%")
            
            with col2:
                best_score = max(s.scores.get('overall_score', 0) for s in filtered_sessions)
                st.metric("Best Score", f"{best_score:.1f}%")
            
            with col3:
                total_time = sum(s.duration for s in filtered_sessions)
                st.metric("Total Practice Time", f"{total_time // 60}m {total_time % 60}s")
            
            with col4:
                improvement = calculate_improvement(filtered_sessions)
                st.metric("Improvement", f"{improvement:+.1f}%")
            
            # Skill radar chart
            st.subheader("🎯 Skill Assessment")
            
            skill_averages = calculate_skill_averages(filtered_sessions)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=list(skill_averages.values()),
                theta=list(skill_averages.keys()),
                fill='toself',
                name='Your Skills'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=True,
                title="Skill Assessment Radar"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Industry and question type breakdown
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏭 Industry Breakdown")
                industry_data = {}
                for session in filtered_sessions:
                    if session.industry not in industry_data:
                        industry_data[session.industry] = []
                    industry_data[session.industry].append(session.scores.get('overall_score', 0))
                
                industry_avg = {industry: sum(scores)/len(scores) for industry, scores in industry_data.items()}
                
                fig = px.bar(
                    x=list(industry_avg.keys()),
                    y=list(industry_avg.values()),
                    title="Average Score by Industry"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("❓ Question Type Analysis")
                question_data = {}
                for session in filtered_sessions:
                    if session.question_type not in question_data:
                        question_data[session.question_type] = []
                    question_data[session.question_type].append(session.scores.get('overall_score', 0))
                
                question_avg = {qtype: sum(scores)/len(scores) for qtype, scores in question_data.items()}
                
                fig = px.pie(
                    values=list(question_avg.values()),
                    names=list(question_avg.keys()),
                    title="Performance by Question Type"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed session table
            st.subheader("📋 Session History")
            
            session_data = []
            for session in filtered_sessions:
                session_data.append({
                    'Date': session.created_at.strftime('%Y-%m-%d %H:%M'),
                    'Industry': session.industry,
                    'Question Type': session.question_type,
                    'Score': f"{session.scores.get('overall_score', 0):.1f}%",
                    'Duration': f"{session.duration}s"
                })
            
            df_sessions = pd.DataFrame(session_data)
            st.dataframe(df_sessions, use_container_width=True)
    
    with tab2:
        # Copy all content from original show_achievements() function
        st.subheader("🏆 Achievements")
        
        user_id = st.session_state.user_id
        gamification = st.session_state.gamification
        db = st.session_state.db_manager

        user_achievements = gamification.get_user_achievements(user_id)
        user_stats = gamification.get_user_stats(user_id)

        # ------------------------------
        # 🎮 XP & Level Display Section
        # ------------------------------
        st.subheader("🎮 Your Progress")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Level", user_stats['level'])
        with col2:
            st.metric("Total XP", user_stats['total_xp'])
        with col3:
            st.metric("XP to Next Level", user_stats['xp_to_next_level'])

        current_level_xp = user_stats['current_level_xp']
        xp_for_level = user_stats['xp_for_current_level']
        progress = current_level_xp / xp_for_level if xp_for_level > 0 else 0

        st.progress(progress)
        st.write(f"Progress to Level {user_stats['level'] + 1}: {current_level_xp}/{xp_for_level} XP")

        # ------------------------------
        # 🏆 Achievements Grid Section
        # ------------------------------
        st.subheader("🏆 Your Achievements")

        achievement_cols = st.columns(3)
        for i, (key, data) in enumerate(ACHIEVEMENTS.items()):
            col = achievement_cols[i % 3]
            with col:
                if key in user_achievements:
                    earned_date = user_achievements[key].strftime('%Y-%m-%d')
                    st.success(f"🏆 {data['name']}")
                    st.write(f"✅ {data['description']}")
                    st.write(f"Earned: {earned_date}")
                else:
                    st.info(f"🔒 {data['name']}")
                    st.write(f"❓ {data['description']}")

        # ------------------------------
        # 🏅 Leaderboard (Simulated)
        # ------------------------------
        st.subheader("🏅 Leaderboard")

        leaderboard_data = [
            {"Rank": 1, "User": "InterviewMaster", "Level": 12, "XP": 4500},
            {"Rank": 2, "User": "TechGuru", "Level": 10, "XP": 3800},
            {"Rank": 3, "User": "You", "Level": user_stats['level'], "XP": user_stats['total_xp']},
            {"Rank": 4, "User": "PracticeKing", "Level": 8, "XP": 2900},
            {"Rank": 5, "User": "SkillBuilder", "Level": 7, "XP": 2400}
        ]

        df_leaderboard = pd.DataFrame(leaderboard_data)
        st.dataframe(df_leaderboard, use_container_width=True)

    with tab3:

        st.title("🎯 Personalized Learning Path")
        
        user_sessions = st.session_state.db_manager.get_user_sessions(st.session_state.user_id)
        
        if not user_sessions:
            st.info("Complete some practice interviews to get your personalized learning path!")
            return
        
        # Analyze performance gaps
        skill_scores = st.session_state.learning_path.analyze_performance_gaps(user_sessions)
        
        # Create learning plan
        target_role = st.selectbox(
            "Target Role:",
            ["Software Engineer", "Data Scientist", "Product Manager", "Marketing Manager", "Other"]
        )
        
        learning_plan = st.session_state.learning_path.create_learning_plan(skill_scores, target_role)
        
        # Display current skill levels
        st.subheader("📊 Current Skill Levels")
        
        col1, col2 = st.columns(2)
        
        with col1:
            for skill, score in skill_scores.items():
                skill_name = skill.replace('_', ' ').title()
                progress = score
                st.metric(skill_name, f"{progress:.1f}%")
        
        with col2:
            # Skills progress chart
            fig = px.bar(
                x=list(skill_scores.keys()),
                y=[score for score in skill_scores.values()],
                title="Current Skill Levels",
                labels={'x': 'Skills', 'y': 'Score (%)'}
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Learning recommendations
        st.subheader("💡 Learning Recommendations")
        
        for i, recommendation in enumerate(learning_plan['recommended_practice'], 1):
            st.write(f"{i}. {recommendation}")
        
        # Focus areas
        st.subheader("🎯 Focus Areas")

        for area in learning_plan['focus_areas']:
            area_name = area.replace('_', ' ').title()
            raw_score = skill_scores.get(area, 0.0)
            current_score = raw_score 
            target_score = learning_plan['target_improvements'].get(area, 0.0) * 100

            st.write(f"**{area_name}**")
            safe_progress = max(0.0, min(raw_score, 1.0))  # ensure within [0.0, 1.0]
            st.progress(safe_progress)
            st.write(f"Current: {current_score:.1f}% → Target: {target_score:.1f}%")

        
        # Estimated timeline
        st.subheader("⏰ Estimated Timeline")
        st.info(f"Based on your current progress, estimated time to reach your goals: {learning_plan['estimated_timeline']}")
        
        # Daily practice suggestions
        st.subheader("📅 Daily Practice Plan")
        
        practice_plan = st.session_state.learning_path.generate_daily_practice_plan(skill_scores, learning_plan)
        
        for day, activities in practice_plan.items():
            with st.expander(f"📅 {day}"):
                st.write(f" {activities}")
            
def show_coding_practice():
    show_coding_practice_page(st.session_state)
    

def show_extra_help():
    st.title("Extra Help")
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
    
    dates = [session.created_at.date() for session in sessions]
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
    
    return [s for s in sessions if s.created_at >= cutoff]

def calculate_improvement(sessions):
    """Calculate improvement between first and last sessions"""
    if len(sessions) < 2:
        return 0
    
    first_score = sessions[-1].scores.get('overall_score', 0)
    last_score = sessions[0].scores.get('overall_score', 0)
    
    return last_score - first_score

def calculate_skill_averages(sessions):
    """Calculate average scores for each skill"""
    if not sessions:
        return {}
    
    skills = ['communication_clarity', 'technical_depth', 'behavioral_examples', 
              'confidence_presence', 'question_relevance']
    
    skill_averages = {}
    for skill in skills:
        scores = [s.scores.get(skill, 0) for s in sessions]
        skill_averages[skill.replace('_', ' ').title()] = sum(scores) / len(scores)
    
    return skill_averages


ACHIEVEMENTS = {
    "first_interview": {
        "name": "Getting Started",
        "description": "Complete your first practice interview",
        "xp_reward": 100
    },
    "perfect_score": {
        "name": "Perfect Performance",
        "description": "Score 95% or higher on an interview",
        "xp_reward": 500
    },
    "consistent_practice": {
        "name": "Dedicated Learner",
        "description": "Practice for 7 consecutive days",
        "xp_reward": 300
    },
    "improvement_streak": {
        "name": "Rising Star",
        "description": "Improve your score for 5 consecutive interviews",
        "xp_reward": 250
    },
    "technical_master": {
        "name": "Technical Expert",
        "description": "Score 90% or higher on 5 technical questions",
        "xp_reward": 400
    },
    "behavioral_ace": {
        "name": "Behavioral Expert",
        "description": "Score 90% or higher on 5 behavioral questions",
        "xp_reward": 400
    },
    "interview_marathon": {
        "name": "Interview Marathon",
        "description": "Complete 50 practice interviews",
        "xp_reward": 1000
    }
}


if __name__ == "__main__":
    main()
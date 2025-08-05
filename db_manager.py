import sqlite3
import hashlib
import uuid
from datetime import datetime
from dataclasses import dataclass, asdict
import json
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

DATABASE_FILE = "interview_coach.db"

# --- Dataclass Definitions ---
# These dataclasses represent the structure of your database tables
# and will be used for type hinting and easy data handling.

@dataclass
class User:
    id: str
    email: str
    target_role: Optional[str] = None
    industry: Optional[str] = None
    experience_level: Optional[str] = None
    total_points: int = 0
    achievements: Optional[str] = None # Stored as JSON string
    learning_path: Optional[str] = None # Stored as JSON string
    created_at: str = datetime.now().isoformat()

@dataclass
class PracticeSession:
    id: str
    user_id: str
    question: str
    transcription: str
    scores: str # Stored as JSON string
    feedback: str # Stored as JSON string
    duration: int # Stored in seconds
    industry: Optional[str] = None
    question_type: Optional[str] = None
    created_at: str = datetime.now().isoformat()

@dataclass
class UserProfile: # If this table is used for additional profile details
    user_id: str
    name: str
    email: str
    level: Optional[str] = None
    roles: Optional[str] = None # Stored as JSON string
    industries: Optional[str] = None # Stored as JSON string

@dataclass
class UserAchievement:
    id: str
    user_id: str
    achievement_type: str
    earned_at: str
    points: int

# --- Database Connection and Initialization ---

def get_db_connection():
    """Establishes and returns a database connection."""
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row # This allows accessing columns by name (e.g., row['column_name'])
    return conn

def initialize_db():
    """Creates tables if they do not exist."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            target_role TEXT,
            industry TEXT,
            experience_level TEXT,
            total_points INTEGER DEFAULT 0,
            achievements TEXT, -- For storing JSON string of achievement IDs/data
            learning_path TEXT, -- For storing JSON string of learning path progress
            created_at TIMESTAMP
        )
    ''')

    # Create user_profiles table (use if distinct from 'users' for additional fields)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id TEXT PRIMARY KEY,
            name TEXT,
            email TEXT,
            level TEXT,
            roles TEXT, -- For storing JSON string of roles
            industries TEXT, -- For storing JSON string of industries
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    # Create practice_sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS practice_sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            question TEXT NOT NULL,
            transcription TEXT NOT NULL,
            scores TEXT, -- For storing JSON string of scores (e.g., {"fluency": 0.8})
            feedback TEXT, -- For storing JSON string of detailed feedback
            duration INTEGER, -- In seconds
            industry TEXT,
            question_type TEXT,
            created_at TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    # Create user_achievements table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_achievements (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            achievement_type TEXT NOT NULL,
            earned_at TIMESTAMP NOT NULL,
            points INTEGER,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    conn.commit()
    conn.close()
    logger.info("Database initialized successfully.")


# --- User Management Functions ---

def create_user(user: User) -> bool:
    """Inserts a new user into the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO users (id, email, target_role, industry, experience_level, total_points, achievements, learning_path, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user.id, user.email, user.target_role, user.industry, user.experience_level,
              user.total_points, user.achievements, user.learning_path, user.created_at))
        conn.commit()
        logger.info(f"User {user.email} created successfully.")
        return True
    except sqlite3.IntegrityError:
        logger.warning(f"User with email {user.email} already exists.")
        return False
    except Exception as e:
        logger.error(f"Error creating user {user.email}: {e}")
        return False
    finally:
        conn.close()

def get_user_by_email(email: str) -> Optional[User]:
    """Retrieves a user by their email address."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
    user_data = cursor.fetchone()
    conn.close()
    if user_data:
        # Convert Row object to dictionary then to User dataclass
        return User(**dict(user_data))
    return None

def get_user_by_id(user_id: str) -> Optional[User]:
    """Retrieves a user by their ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    user_data = cursor.fetchone()
    conn.close()
    if user_data:
        return User(**dict(user_data))
    return None

def update_user(user: User) -> bool:
    """Updates an existing user's data."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            UPDATE users
            SET email = ?, target_role = ?, industry = ?, experience_level = ?, total_points = ?, achievements = ?, learning_path = ?
            WHERE id = ?
        ''', (user.email, user.target_role, user.industry, user.experience_level,
              user.total_points, user.achievements, user.learning_path, user.id))
        conn.commit()
        logger.info(f"User {user.id} updated successfully.")
        return True
    except Exception as e:
        logger.error(f"Error updating user {user.id}: {e}")
        return False
    finally:
        conn.close()

# --- Practice Session Management Functions ---

def save_practice_session(session_data: PracticeSession) -> bool:
    """Saves a completed practice session to the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO practice_sessions (id, user_id, question, transcription, scores, feedback, duration, industry, question_type, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (session_data.id, session_data.user_id, session_data.question,
              session_data.transcription, session_data.scores, session_data.feedback,
              session_data.duration, session_data.industry, session_data.question_type,
              session_data.created_at))
        conn.commit()
        logger.info(f"Practice session {session_data.id} saved successfully for user {session_data.user_id}.")
        return True
    except Exception as e:
        logger.error(f"Error saving practice session {session_data.id}: {e}")
        return False
    finally:
        conn.close()

def get_user_practice_sessions(user_id: str) -> List[PracticeSession]:
    """Retrieves all practice sessions for a given user."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM practice_sessions WHERE user_id = ? ORDER BY created_at DESC', (user_id,))
    sessions_data = cursor.fetchall()
    conn.close()
    return [PracticeSession(**dict(session_data)) for session_data in sessions_data]

# --- User Profile (Optional, if distinct from 'users' table and needed) ---
def save_user_profile(profile: UserProfile) -> bool:
    """Saves or updates a user's profile."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT OR REPLACE INTO user_profiles (user_id, name, email, level, roles, industries)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (profile.user_id, profile.name, profile.email, profile.level, profile.roles, profile.industries))
        conn.commit()
        logger.info(f"User profile for {profile.user_id} saved/updated successfully.")
        return True
    except Exception as e:
        logger.error(f"Error saving user profile for {profile.user_id}: {e}")
        return False
    finally:
        conn.close()

def get_user_profile(user_id: str) -> Optional[UserProfile]:
    """Retrieves a user profile by user ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM user_profiles WHERE user_id = ?', (user_id,))
    profile_data = cursor.fetchone()
    conn.close()
    if profile_data:
        return UserProfile(**dict(profile_data))
    return None

def save_user_achievement(user_id: str, achievement_type: str, earned_at: str, points: int = 50) -> bool:
    conn = get_db_connection()
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
            earned_at, points
        ))

        # Add points to user
        cursor.execute("""
            UPDATE users SET total_points = total_points + ? WHERE id = ?
        """, (points, user_id))

        conn.commit()
        conn.close()
        return True

    conn.close()
    return False

def get_user_achievements(user_id: str) -> List[UserAchievement]:
    """Retrieves all achievements for a given user."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM user_achievements WHERE user_id = ? ORDER BY earned_at DESC', (user_id,))
    achievements_data = cursor.fetchall()
    conn.close()
    return [UserAchievement(**dict(achievement_data)) for achievement_data in achievements_data]


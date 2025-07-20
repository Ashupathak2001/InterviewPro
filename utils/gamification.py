import json
import os
from datetime import datetime, timedelta
from typing import Dict
import logging

logger = logging.getLogger(__name__)

STATS_FILE = "data/user_stats.json"
ACHIEVEMENTS_FILE = "data/user_achievements.json"

def get_user_achievements() -> Dict:
    """Load user achievements """
    return None

def get_user_stats() -> Dict:
    """Load user statistics """
    try:
        if os.path.exists(STATS_FILE):
            with open(STATS_FILE) as f:
                return json.load(f)
        return None
    except Exception as e:
        logger.error(f"Error loading user stats: {e}")
        return None

def update_user_stats(score: float, topic: str) -> Dict:
    """Update user statistics with new practice session"""
    try:
        # Load existing stats or initialize
        try:
            with open(STATS_FILE) as f:
                stats = json.load(f)
        except:
            stats = {
                "total_sessions": 0,
                "average_score": 0,
                "streak_days": 0,
                "last_practice": None,
                "badges": [],
                "topic_scores": {}
            }
        
        # Update basic stats
        stats["total_sessions"] += 1
        stats["average_score"] = (
            (stats["average_score"] * (stats["total_sessions"] - 1) + score
        ) / stats["total_sessions"])
        
        # Update streak
        today = datetime.now().date()
        last_practice = (
            datetime.strptime(stats["last_practice"], "%Y-%m-%d").date()
            if stats["last_practice"] else None
        )
        
        if last_practice == today - timedelta(days=1):
            stats["streak_days"] += 1
        elif last_practice != today:
            stats["streak_days"] = 1
        
        stats["last_practice"] = str(today)
        
        # Update topic scores
        if topic not in stats["topic_scores"]:
            stats["topic_scores"][topic] = []
        stats["topic_scores"][topic].append(score)
        
        # Check for new badges
        badges = stats["badges"]
        if stats["total_sessions"] >= 5 and "Regular Practitioner" not in badges:
            badges.append("Regular Practitioner")
        if stats["total_sessions"] >= 20 and "Dedicated Learner" not in badges:
            badges.append("Dedicated Learner")
        if stats["average_score"] > 80 and "High Performer" not in badges:
            badges.append("High Performer")
        if len(stats["topic_scores"]) >= 3 and "Versatile Learner" not in badges:
            badges.append("Versatile Learner")
        if stats["streak_days"] >= 7 and "Weekly Streak" not in badges:
            badges.append("Weekly Streak")
        
        # Save updated stats
        os.makedirs("data", exist_ok=True)
        with open(STATS_FILE, 'w') as f:
            json.dump(stats, f)
        
        return stats
    except Exception as e:
        logger.error(f"Error updating user stats: {e}")
        return None
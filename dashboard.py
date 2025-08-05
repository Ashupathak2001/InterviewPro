import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import json
from typing import List

# Helper functions
def safe_parse_scores(scores_data) -> dict:
    """Safely parse scores from string or dict"""
    if isinstance(scores_data, dict):
        return scores_data
    try:
        return json.loads(scores_data) if scores_data else {}
    except (json.JSONDecodeError, TypeError):
        return {}

def calculate_practice_streak(sessions) -> int:
    """Calculate consecutive practice days"""
    if not sessions:
        return 0
    
    dates = sorted({datetime.fromisoformat(s.created_at).date() for s in sessions}, reverse=True)
    if not dates or dates[0] != datetime.now().date():
        return 0
    
    streak = 1
    for i in range(1, len(dates)):
        if (dates[i-1] - dates[i]).days == 1:
            streak += 1
        else:
            break
    return streak

# Dashboard components
def _render_welcome_message():
    """Show welcome for new users"""
    st.info("👋 Welcome to ForgeMe! Start with your first practice interview to see your progress here.")
    if st.button("🚀 Start First Interview"):
        st.session_state.page = "🎤 Practice Interview"
        st.rerun()

def _render_metrics(user_sessions):
    """Top metrics row"""
    col1, col2, col3, col4 = st.columns(4)
    
    # Total Sessions
    with col1:
        today_count = len([s for s in user_sessions 
                          if datetime.fromisoformat(s.created_at).date() == datetime.now().date()])
        st.metric("Total Sessions", len(user_sessions), delta=f"+{today_count} today")
    
    # Average Score
    with col2:
        avg_score = sum(safe_parse_scores(s.scores).get('overall_score', 0) 
                   for s in user_sessions) / max(1, len(user_sessions))
        st.metric("Average Score", f"{avg_score:.1f}%", delta=f"{avg_score - 50:.1f}%")
    
    # Latest Score
    with col3:
        recent_score = safe_parse_scores(user_sessions[0].scores).get('overall_score', 0)
        delta = f"{recent_score - avg_score:.1f}%" if len(user_sessions) > 1 else None
        st.metric("Latest Score", f"{recent_score:.1f}%", delta=delta)
    
    # Practice Streak
    with col4:
        streak = calculate_practice_streak(user_sessions)
        st.metric("Practice Streak", f"{streak} days", delta="+1" if streak > 0 else "Start today!")

def _render_trend_chart(user_sessions):
    """Performance trend visualization"""
    st.subheader("📈 Performance Trend")
    
    if len(user_sessions) > 1:
        session_data = []
        for s in reversed(user_sessions[-10:]):  # Last 10 sessions
            scores = safe_parse_scores(s.scores)
            session_data.append({
                'Date': datetime.fromisoformat(s.created_at).date(),
                'Overall Score': scores.get('overall_score', 0),
                'Communication': scores.get('communication_clarity', 0),
                'Technical': scores.get('technical_depth', 0),
                'Behavioral': scores.get('behavioral_examples', 0)
            })
        
        df = pd.DataFrame(session_data)
        fig = px.line(df, x='Date', 
                     y=['Overall Score', 'Communication', 'Technical', 'Behavioral'],
                     title="Performance Trends (Last 10 Sessions)")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def _render_recent_sessions(user_sessions):
    """Recent activity list"""
    st.subheader("🕐 Recent Activity")
    
    for session in user_sessions[:5]:  # Last 5 sessions
        with st.expander(
            f"📅 {datetime.fromisoformat(session.created_at).strftime('%Y-%m-%d %H:%M')} - "
            f"{session.industry} ({session.question_type})"
        ):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Question:** {session.question}")
                st.write(f"**Your Answer:** {session.transcription[:200]}...")
            
            with col2:
                scores = safe_parse_scores(session.scores)
                st.metric("Score", f"{scores.get('overall_score', 0):.1f}%")
                st.write(f"**Duration:** {session.duration}s")

# Main exported function
def show_dashboard():
    """Entry point for dashboard page"""
    st.title("📊 Performance Dashboard")
    
    try:
        # Get sessions from database
        from db_manager import get_user_practice_sessions  # Local import to avoid circular dependencies
        user_sessions = get_user_practice_sessions(st.session_state.user_id)
        
        if not user_sessions:
            _render_welcome_message()
            return
        
        # Render dashboard components
        _render_metrics(user_sessions)
        _render_trend_chart(user_sessions)
        _render_recent_sessions(user_sessions)
        
    except Exception as e:
        st.error("Failed to load dashboard data")
        if st.button("🔄 Refresh Dashboard"):
            st.rerun()
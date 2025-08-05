import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import List, Dict
import json

import logging
logger = logging.getLogger(__name__)

# Helper functions
def safe_parse_scores(scores_data) -> dict:
    """Safely parse scores from string or dict"""
    if isinstance(scores_data, dict):
        return scores_data
    try:
        return json.loads(scores_data) if scores_data else {}
    except (json.JSONDecodeError, TypeError):
        return {'overall_score': 0}

def filter_sessions_by_time(sessions: List, time_period: str) -> List:
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

def calculate_improvement(sessions: List) -> float:
    """Calculate score improvement between first and last sessions"""
    if len(sessions) < 2:
        return 0.0
    first = safe_parse_scores(sessions[-1].scores).get('overall_score', 0)
    last = safe_parse_scores(sessions[0].scores).get('overall_score', 0)
    return last - first

def calculate_skill_averages(sessions: List) -> Dict[str, float]:
    """Calculate average scores for each skill category"""
    skills = ['communication_clarity', 'technical_depth', 'behavioral_examples',
              'confidence_presence', 'question_relevance']
    return {
        skill.replace('_', ' ').title(): 
        sum(safe_parse_scores(s.scores).get(skill, 0) for s in sessions) / len(sessions)
        for skill in skills
    }

# Analytics Tab Components
def _render_performance_metrics(filtered_sessions: List):
    """Render the top metrics row"""
    st.subheader("📈 Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_score = sum(safe_parse_scores(s.scores).get('overall_score', 0) 
                   for s in filtered_sessions) / len(filtered_sessions)
        st.metric("Average Score", f"{avg_score:.1f}%")
    
    with col2:
        best_score = max(safe_parse_scores(s.scores).get('overall_score', 0) 
                        for s in filtered_sessions)
        st.metric("Best Score", f"{best_score:.1f}%")
    
    with col3:
        total_time = sum(s.duration for s in filtered_sessions)
        st.metric("Total Practice Time", f"{total_time//60}m {total_time%60}s")
    
    with col4:
        improvement = calculate_improvement(filtered_sessions)
        st.metric("Improvement", f"{improvement:+.1f}%")

def _render_skill_assessment(filtered_sessions: List):
    """Render the skill radar chart"""
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
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        title="Skill Assessment Radar"
    )
    st.plotly_chart(fig, use_container_width=True)

def _render_industry_analysis(filtered_sessions: List):
    """Render industry and question type breakdowns"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏭 Industry Breakdown")
        industry_data = {
            s.industry: safe_parse_scores(s.scores).get('overall_score', 0) 
            for s in filtered_sessions
        }
        fig = px.bar(
            x=list(industry_data.keys()),
            y=list(industry_data.values()),
            title="Average Score by Industry"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("❓ Question Type Analysis")
        question_data = {
            s.question_type: safe_parse_scores(s.scores).get('overall_score', 0)
            for s in filtered_sessions
        }
        fig = px.pie(
            values=list(question_data.values()),
            names=list(question_data.keys()),
            title="Performance by Question Type"
        )
        st.plotly_chart(fig, use_container_width=True)

def _render_session_history(filtered_sessions: List):
    """Render the session history table"""
    st.subheader("📋 Session History")
    session_data = [{
        'Date': datetime.fromisoformat(s.created_at).strftime('%Y-%m-%d %H:%M'),
        'Industry': s.industry,
        'Question Type': s.question_type,
        'Score': f"{safe_parse_scores(s.scores).get('overall_score', 0):.1f}%",
        'Duration': f"{s.duration}s"
    } for s in filtered_sessions]
    st.dataframe(pd.DataFrame(session_data), use_container_width=True)


# Main function
def show_analytics():
    """Main analytics/achievements view with tabs"""
    st.title("📊 Analytics & Achievements")
    
    # 1. Verify user session exists
    if 'user_id' not in st.session_state:
        st.error("Please login to view analytics")
        return
    
    # 2. Initialize loading state
    loading_placeholder = st.empty()
    loading_placeholder.info("Loading analytics data...")
    
    try:
        # 3. Load user sessions with error handling
        try:
            from db_manager import get_user_practice_sessions
            user_sessions = get_user_practice_sessions(st.session_state.user_id)
        except Exception as db_error:
            logger.error(f"Database error: {db_error}")
            loading_placeholder.error("Failed to load practice history")
            if st.button("Retry Database Connection"):
                st.rerun()
            return
        
        # Clear loading state
        loading_placeholder.empty()
        
        # 4. Setup tabs
        # tab1, tab2, tab3 = st.tabs(["📈 Performance Analytics", "🏆 Achievements", "🎯 Learning Path"])
        tab1 = st.tabs(["📈 Performance Analytics"])[0]
        
        with tab1:
            if not user_sessions:
                st.info("Complete some interviews to see analytics!")
            else:
                try:
                    time_period = st.selectbox(
                        "Select Time Period:",
                        ["Last 7 days", "Last 30 days", "Last 90 days", "All time"],
                        key="analytics_time_period"
                    )
                    filtered = filter_sessions_by_time(user_sessions, time_period)
                    
                    # Render components with individual error handling
                    try:
                        _render_performance_metrics(filtered)
                    except Exception as e:
                        st.error("Could not display performance metrics")
                        logger.error(f"Metrics error: {e}")
                    
                    try:
                        _render_skill_assessment(filtered)
                    except Exception as e:
                        st.error("Could not generate skill assessment")
                    
                    try:
                        _render_industry_analysis(filtered)
                    except Exception as e:
                        st.error("Could not analyze industry data")
                    
                    try:
                        _render_session_history(filtered)
                    except Exception as e:
                        st.error("Could not load session history")
                        
                except Exception as tab_error:
                    st.error("Failed to load performance analytics")
                    logger.error(f"Tab1 error: {tab_error}")

    except Exception as e:
        st.error("A system error occurred")
        logger.critical(f"Analytics page error: {e}")
        
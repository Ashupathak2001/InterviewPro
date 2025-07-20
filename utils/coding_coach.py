# coding_practice_module.py

import streamlit as st
import pandas as pd
import re
import os

DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "question_bank.csv")

# List of GFG Topics from Knowledge Base
# GFG_TOPICS = [
#     "Data Structures", "Algorithms", "System Design", "Foundational Courses",
#     "Data Science", "Practice Problem", "Python", "Machine Learning",
#     "Data Science Using Python", "Django", "DevOps", "JavaScript", "Java",
#     "C", "C++", "ReactJS", "NodeJS", "Web Development", "Web Design",
#     "Web Browser", "CP Live", "Aptitude", "Puzzles", "Projects", "DSA",
#     "Design Patterns", "Software Development", "SEO", "Product Management",
#     "SAP", "Programming"
# ]

def extract_link_from_text(text):
    url_pattern = r'https?://[^\s<>"{}|\\^`[\]]+'
    urls = re.findall(url_pattern, str(text))
    return urls[0] if urls else "https://www.geeksforgeeks.org/ "

def clean_problem_name(text):
    clean_text = re.sub(r'https?://[^\s<>"{}|\\^`[\]]+', '', str(text))
    return clean_text.strip()

def classify_difficulty(problem_name):
    problem_lower = problem_name.lower()
    easy_keywords = ['two sum', 'reverse', 'palindrome', 'valid', 'merge', 'binary search']
    hard_keywords = ['dp', 'dynamic programming', 'backtracking', 'tree', 'graph', 'matrix']
    
    if any(keyword in problem_lower for keyword in easy_keywords):
        return 'Easy'
    elif any(keyword in problem_lower for keyword in hard_keywords):
        return 'Hard'
    else:
        return 'Medium'

def load_questions_from_excel(file):
    try:
        df = pd.read_excel(file)
        if len(df.columns) >= 2:
            df.columns = ['Topic', 'Problem']
            df['Link'] = df['Problem'].apply(extract_link_from_text)
            df['Problem_Name'] = df['Problem'].apply(clean_problem_name)
            df['Difficulty'] = df['Problem_Name'].apply(classify_difficulty)
            df['Topic'] = df['Topic'].astype(str).str.strip()
            return df
        else:
            st.error("Excel file should have at least 2 columns: Topic and Problem")
            return None
    except Exception as e:
        st.error(f"Error loading Excel file: {e}")
        return None

def select_random_coding_question(df, attempted, skipped):
    available_questions = df[~df.index.isin(attempted) & ~df.index.isin(skipped)]
    if len(available_questions) == 0:
        skipped_questions = df[df.index.isin(skipped)]
        if len(skipped_questions) > 0:
            return skipped_questions.sample(n=1).iloc[0]
        return None
    return available_questions.sample(n=1).iloc[0]

def get_ai_help(question_data, help_type):
    responses = {
        "explanation": f"To solve {question_data['Problem_Name']}, you would typically use a {question_data['Topic']} approach.",
        "hints": f"Key insights for {question_data['Problem_Name']}: Think about how you might use {question_data['Topic']} concepts.",
        "optimization": f"For optimizing {question_data['Problem_Name']}, consider these approaches...",
        "similar": f"Similar problems to {question_data['Problem_Name']} include..."
    }
    return responses.get(help_type, "No help available for this type.")

def show_coding_practice_page(session_state):
    st.title("🎯 Coding Practice")

    # Initialize session_state keys if not present
    if 'coding_questions_df' not in session_state:
        session_state.coding_questions_df = None
    if 'coding_current_question' not in session_state:
        session_state.coding_current_question = None
    if 'coding_attempted_questions' not in session_state:
        session_state.coding_attempted_questions = set()
    if 'coding_solved_questions' not in session_state:
        session_state.coding_solved_questions = set()
    if 'coding_skipped_questions' not in session_state:
        session_state.coding_skipped_questions = set()
    if 'coding_topic_filter' not in session_state:
        session_state.coding_topic_filter = "All"
    if 'coding_remaining_questions' not in session_state:
        session_state.coding_remaining_questions = []

    col1, col2 = st.columns([2, 1])

    with col1:
        # Load existing CSV if available
        if session_state.coding_questions_df is None:
            if os.path.exists(CSV_PATH):
                with st.spinner("Loading saved question bank..."):
                    session_state.coding_questions_df = pd.read_csv(CSV_PATH)
                    st.success("✅ Loaded previous question bank successfully!")

        # File upload section
        if session_state.coding_questions_df is None:
            st.header("📁 Upload Question Bank")
            uploaded_file = st.file_uploader(
                "Upload Excel file with format: Topic | Problem (with embedded GFG links)",
                type=['xlsx', 'xls'],
                help="Excel should have 2 columns: Topic and Problem (with embedded links)"
            )

            if uploaded_file:
                with st.spinner("Processing Excel file..."):
                    df = load_questions_from_excel(uploaded_file)
                    if df is not None:
                        os.makedirs(DATA_DIR, exist_ok=True)
                        df.to_csv(CSV_PATH, index=False)
                        session_state.coding_questions_df = df
                        session_state.coding_current_question = select_random_coding_question(
                            df, set(), set()
                        )
                        st.success(f"✅ Loaded and saved {len(df)} questions successfully!")
                        st.rerun()

        # Only proceed if questions are loaded
        if session_state.coding_questions_df is not None:
            df = session_state.coding_questions_df

            # Tab Navigation
            tab1, tab2, tab3 = st.tabs(["📚 Practice", "📊 Overview", "Recent Acticity"])

            with tab1:
                st.subheader("📘 Choose Topic & Start Practice")

                # Topic selection
                topics = df['Topic'].dropna().astype(str).unique().tolist()
                topic_options = ["All"] + sorted(topics)
                selected_topic = st.selectbox("Choose Topic", topic_options)

                if st.button("Start Session", type="primary"):
                    session_state.coding_topic_filter = selected_topic
                    filtered_df = df if selected_topic == "All" else df[df['Topic'] == selected_topic]
                    session_state.coding_remaining_questions = filtered_df.sample(frac=1).index.tolist()[:5]
                    session_state.coding_current_question = df.loc[session_state.coding_remaining_questions.pop(0)]
                    session_state.coding_attempted_questions = set()
                    session_state.coding_solved_questions = set()
                    session_state.coding_skipped_questions = set(session_state.coding_remaining_questions)
                    st.rerun()

                # Display current question
                if session_state.coding_current_question is not None:
                    question = session_state.coding_current_question
                    question_index = question.name

                    difficulty_class = f"difficulty-{question['Difficulty'].lower()}"
                    st.markdown(f"""
                    <div style="border:1px solid #ccc; padding: 10px; border-radius: 8px;">
                        <h3>{question['Problem_Name']}</h3>
                        <p><strong>Topic:</strong> {question['Topic']}</p>
                        <p><strong>Difficulty:</strong> <span class="{difficulty_class}">{question['Difficulty']}</span></p>
                    </div>
                    """, unsafe_allow_html=True)

                    col1_btn, col2_btn, col3_btn = st.columns(3)

                    with col1_btn:
                        if st.button("🚀 Start Problem", type="primary"):
                            session_state.coding_attempted_questions.add(question_index)
                            st.markdown(f"**Opening GFG Link:** [Click here to solve]({question['Link']})")
                            

                    with col2_btn:
                        if st.button("⏭️ Skip for Later"):
                            session_state.coding_skipped_questions.add(question_index)
                            if session_state.coding_remaining_questions:
                                next_idx = session_state.coding_remaining_questions.pop(0)
                                session_state.coding_current_question = df.loc[next_idx]
                            else:
                                session_state.coding_current_question = None
                                st.info("✅ No more questions left in this session.")
                            st.rerun()

                    with col3_btn:
                        if question_index in session_state.coding_attempted_questions:
                            if st.button("✅ Mark as Solved"):
                                session_state.coding_solved_questions.add(question_index)
                                if session_state.coding_remaining_questions:
                                    next_idx = session_state.coding_remaining_questions.pop(0)
                                    session_state.coding_current_question = df.loc[next_idx]
                                else:
                                    session_state.coding_current_question = None
                                    st.success("🎉 You've completed this session!")
                                st.rerun()

                    # AI Help section
                    if question_index in session_state.coding_attempted_questions:
                        st.subheader("🤖 AI Learning Assistant")

                        help_type = st.selectbox(
                            "Choose AI assistance type:",
                            ["explanation", "hints", "optimization", "similar"],
                            format_func=lambda x: {
                                "explanation": "📚 Explain approach and algorithm",
                                "hints": "💡 Provide hints and insights",
                                "optimization": "🔧 Optimization techniques",
                                "similar": "🎯 Similar problems to practice"
                            }[x]
                        )

                        if st.button("Get AI Help"):
                            with st.spinner("Getting AI assistance..."):
                                help_content = get_ai_help(question, help_type)
                                st.markdown(f"""
                                <div style="background:#f0f0f0; padding:10px; border-radius:5px;">
                                    <h4>🤖 AI Assistant Response:</h4>
                                    <p>{help_content}</p>
                                </div>
                                """, unsafe_allow_html=True)

            with tab2:
                st.subheader("📈 Question Bank Overview")

                # Topic distribution
                topic_counts = df['Topic'].value_counts()
                st.write("**Topics**")
                for topic, count in topic_counts.items():
                    st.write(f"- {topic}: {count} questions")

                # Difficulty distribution
                difficulty_counts = df['Difficulty'].value_counts()
                st.write("\n**Difficulty Levels**")
                for difficulty, count in difficulty_counts.items():
                    st.write(f"- {difficulty}: {count} questions")

            with tab3:
                # Recent activity
                st.title("Recent Activity")
                if session_state.coding_attempted_questions:
                    st.write("\n**Recent Activity**")
                    recent_questions = df.loc[list(session_state.coding_attempted_questions)[-3:]]
                    for _, q in recent_questions.iterrows():
                        status = "✅" if q.name in session_state.coding_solved_questions else "⏳"
                        st.write(f"{status} {q['Problem_Name']}")


        else:
            st.info("Please upload an Excel file to start practicing.")
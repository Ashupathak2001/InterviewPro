import streamlit as st
import pandas as pd
import re
import os
import google.generativeai as genai
from dotenv import load_dotenv
import openpyxl

# Load environment variables from .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

DATA_DIR = "data"
QUESTION_BANK_CSV = os.path.join(DATA_DIR, "question_bank.csv")
PROBLEM_LINKS_CSV = os.path.join(DATA_DIR, "problem_links.csv")

def extract_links_from_excel(file_path):
    """
    Extracts data from an Excel file, including hyperlinks, using openpyxl.
    """
    wb = openpyxl.load_workbook(file_path)
    ws = wb.active
    data = []
    # Skip header row (min_row=2)
    for row in ws.iter_rows(min_row=2, values_only=False):
        topic_cell = row[0]
        problem_cell = row[1]
        
        topic = topic_cell.value
        problem_text = problem_cell.value
        
        # Prioritize the embedded hyperlink
        hyperlink = problem_cell.hyperlink.target if problem_cell.hyperlink else None
        
        # If no embedded hyperlink, try to extract one from the text
        if not hyperlink:
            hyperlink = extract_link_from_text(problem_text)
            
        data.append({
            "Topic": topic,
            "Problem": problem_text,
            "Link": hyperlink
        })
    return pd.DataFrame(data)

def extract_link_from_text(text):
    """
    Extracts the first URL from a string.
    """
    url_pattern = r'https?://[^\s<>"{}|\\^`[\]]+'
    urls = re.findall(url_pattern, str(text))
    return urls[0] if urls else "https://www.geeksforgeeks.org/"

def clean_problem_name(text):
    """
    Removes URLs from a string to get a clean problem name.
    """
    clean_text = re.sub(r'https?://[^\s<>"{}|\\^`[\]]+', '', str(text))
    return clean_text.strip()

def classify_difficulty(problem_name):
    """
    Classifies a problem's difficulty based on keywords.
    """
    problem_lower = problem_name.lower()
    easy_keywords = ['two sum', 'reverse', 'palindrome', 'valid', 'merge', 'binary search']
    hard_keywords = ['dp', 'dynamic programming', 'backtracking', 'tree', 'graph', 'matrix']
    
    if any(keyword in problem_lower for keyword in easy_keywords):
        return 'Easy'
    elif any(keyword in problem_lower for keyword in hard_keywords):
        return 'Hard'
    else:
        return 'Medium'

def process_uploaded_excel(file):
    """
    Loads, processes, and prepares the data from an uploaded Excel file,
    creating two CSV files.
    """
    try:
        df = extract_links_from_excel(file)
        if 'Topic' in df.columns and 'Problem' in df.columns:
            df['Problem_Name'] = df['Problem'].apply(clean_problem_name)
            df['Difficulty'] = df['Problem_Name'].apply(classify_difficulty)
            df['Topic'] = df['Topic'].astype(str).str.strip()

            # Create the 'data' directory if it doesn't exist
            os.makedirs(DATA_DIR, exist_ok=True)

            # Save the main DataFrame to a CSV file
            df.to_csv(QUESTION_BANK_CSV, index=False)

            # Create a new DataFrame with just Problem_Name and Link
            links_df = df[['Problem_Name', 'Link']].copy()
            # Save this new DataFrame to a separate CSV file
            links_df.to_csv(PROBLEM_LINKS_CSV, index=False)

            return df
        else:
            st.error("Excel file should have 'Topic' and 'Problem' columns.")
            return None
    except Exception as e:
        st.error(f"Error loading or processing Excel file: {e}")
        return None

def get_ai_help(question_data, help_type):
    """
    Generates AI-powered assistance for a coding problem.
    """
    prompt_map = {
        "explanation": "Explain the approach and algorithm to solve this coding problem.",
        "hints": "Give useful hints to help solve this problem.",
        "optimization": "How can this solution be optimized further?",
        "similar": "Suggest similar problems and briefly explain why they are related."
    }

    prompt = f"""
You are an expert coding tutor.

Problem Name: {question_data['Problem_Name']}
Topic: {question_data['Topic']}
Difficulty: {question_data['Difficulty']}

{prompt_map.get(help_type, "Give some help regarding this problem.")}
"""
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Failed to get AI help: {e}"

def show_coding_practice_page():
    """
    Main Streamlit page for coding practice.
    """
    st.title("👨‍💻 Coding Practice")

    if 'coding_questions_df' not in st.session_state:
        st.session_state.coding_questions_df = None

    if st.session_state.coding_questions_df is None:
        if os.path.exists(QUESTION_BANK_CSV):
            with st.spinner("Loading saved question bank..."):
                st.session_state.coding_questions_df = pd.read_csv(QUESTION_BANK_CSV)
                st.success("✅ Loaded previous question bank successfully!")
                st.rerun()

    if st.session_state.coding_questions_df is None:
        st.header("📂 Upload Question Bank")
        uploaded_file = st.file_uploader(
            "Upload Excel file with format: Topic | Problem (with embedded GFG links)",
            type=['xlsx', 'xls'],
            help="Excel should have 2 columns: 'Topic' and 'Problem' (with embedded links)"
        )
        if uploaded_file:
            with st.spinner("Processing Excel file..."):
                df = process_uploaded_excel(uploaded_file)
                if df is not None and not df.empty:
                    st.session_state.coding_questions_df = df
                    st.success(f"✅ Loaded and saved {len(df)} questions successfully!")
                    st.success("✅ Also created 'problem_links.csv' with problem names and links.")
                    st.rerun()
                elif df is not None and df.empty:
                    st.error("The uploaded file contains no questions.")
    
    if st.session_state.coding_questions_df is not None and not st.session_state.coding_questions_df.empty:
        df = st.session_state.coding_questions_df
        tab1, tab2, tab3 = st.tabs(["📚 Practice", "📊 Overview", "📌 Skipped Questions"])

        with tab1:
            st.subheader("📘 Choose Topic & Start Practice")
            topics = df['Topic'].dropna().astype(str).unique().tolist()
            topic_options = ["All"] + sorted(topics)
            selected_topic = st.selectbox("Choose Topic", topic_options, key="topic_select")

            # NEW: Add a number input to control the session length
            session_length = st.number_input(
                "Number of questions in this session:", 
                min_value=1, 
                max_value=len(df), 
                value=5,
                key="session_length_input"
            )

            if st.button("Start Session", type="primary"):
                if 'coding_questions_df' not in st.session_state or st.session_state.coding_questions_df is None:
                    st.error("Please upload a question bank first.")
                    st.stop()
                
                st.session_state.coding_topic_filter = selected_topic
                filtered_df = df if selected_topic == "All" else df[df['Topic'] == selected_topic]
                
                if not filtered_df.empty:
                    all_questions_in_session = filtered_df.sample(frac=1).index.tolist()
                    
                    # NEW: Use the user-defined session length
                    st.session_state.coding_remaining_questions = all_questions_in_session[:session_length]
                    
                    if st.session_state.coding_remaining_questions:
                        st.session_state.coding_current_question = df.loc[st.session_state.coding_remaining_questions.pop(0)]
                        st.session_state.coding_attempted_questions = set()
                        st.session_state.coding_solved_questions = set()
                        st.session_state.coding_skipped_questions = set()
                        st.rerun()
                    else:
                        st.info("No questions left for this topic.")
                        st.session_state.coding_current_question = None
                else:
                    st.info("No questions found for the selected topic.")
                    st.session_state.coding_current_question = None

            if st.session_state.coding_current_question is not None:
                question = st.session_state.coding_current_question
                question_index = question.name

                st.markdown(f"""
                <div style="border:1px solid #ccc; padding: 10px; border-radius: 8px;">
                    <h3>{question['Problem_Name']}</h3>
                    <p><strong>Topic:</strong> {question['Topic']}</p>
                    <p><strong>Difficulty:</strong> <span style="color:{'#4CAF50' if question['Difficulty'] == 'Easy' else '#FFC107' if question['Difficulty'] == 'Medium' else '#F44336'}">{question['Difficulty']}</span></p>
                </div>
                """, unsafe_allow_html=True)

                col1_btn, col2_btn, col3_btn = st.columns(3)

                with col1_btn:
                    if st.button("🚀 Start Problem", type="primary"):
                        st.session_state.coding_attempted_questions.add(question_index)
                        st.markdown(f"""
<div style="
    background-color: #4CAF50; 
    padding: 10px 15px; 
    border-radius: 5px; 
    display: inline-block; 
    margin-top: 10px;
">
    <a href="{question['Link']}" target="_blank" style="
        color: white; 
        text-decoration: none; 
        font-weight: bold;
        font-size: 16px;
    " title="Open the problem on GeeksforGeeks">
        🚀 Open GFG Problem
    </a>
</div>
""", unsafe_allow_html=True)

                with col2_btn:
                    if st.button("⏭️ Skip"):
                        st.session_state.coding_skipped_questions.add(question_index)
                        if st.session_state.coding_remaining_questions:
                            next_idx = st.session_state.coding_remaining_questions.pop(0)
                            st.session_state.coding_current_question = df.loc[next_idx]
                        else:
                            st.session_state.coding_current_question = None
                            st.info("✅ No more questions left in this session.")
                        st.rerun()

                with col3_btn:
                    if question_index in st.session_state.coding_attempted_questions:
                        if st.button("✅ Mark as Solved"):
                            st.session_state.coding_solved_questions.add(question_index)
                            if st.session_state.coding_remaining_questions:
                                next_idx = st.session_state.coding_remaining_questions.pop(0)
                                st.session_state.coding_current_question = df.loc[next_idx]
                            else:
                                st.session_state.coding_current_question = None
                                st.success("🎉 You've completed this session!")
                            st.rerun()

                if question_index in st.session_state.coding_attempted_questions:
                    st.subheader("🤖 AI Learning Assistant")
                    help_type = st.selectbox(
                        "Choose AI assistance type:",
                        ["explanation", "hints", "optimization", "similar"],
                        key=f"ai_help_select_{question_index}",
                        format_func=lambda x: {
                            "explanation": "📚 Explain approach and algorithm",
                            "hints": "💡 Provide hints and insights",
                            "optimization": "🔧 Optimization techniques",
                            "similar": "🎯 Similar problems to practice"
                        }[x]
                    )
                    if st.button("Get AI Help", key=f"get_ai_help_{question_index}"):
                        with st.spinner("Getting AI assistance..."):
                            help_content = get_ai_help(question, help_type)
                            st.markdown(f"""
                            <div style="background:#f0f0f0; padding:10px; border-radius:5px;">
                                <h4>🤖 AI Assistant Response:</h4>
                                <p>{help_content}</p>
                            </div>
                            """, unsafe_allow_html=True)

        with tab2:
            st.subheader("📊 Question Bank Overview")
            st.markdown("### Topic Breakdown")
            topic_counts = df['Topic'].value_counts().reset_index()
            topic_counts.columns = ['Topic', 'Count']
            st.dataframe(topic_counts, use_container_width=True, hide_index=True)
            
            st.markdown("### Difficulty Distribution")
            difficulty_counts = df['Difficulty'].value_counts().reset_index()
            difficulty_counts.columns = ['Difficulty', 'Count']
            st.dataframe(difficulty_counts, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("### Visual Summary")
            st.bar_chart(topic_counts.set_index('Topic'))

        with tab3:
            st.subheader("📌 Skipped Questions - AI Review")
            if st.session_state.coding_skipped_questions:
                for idx in sorted(list(st.session_state.coding_skipped_questions)):
                    if idx in df.index:
                        question = df.loc[idx]
                        with st.expander(f"❓ {question['Problem_Name']} ({question['Topic']}, {question['Difficulty']})"):
                            help_type = st.selectbox(
                                f"Select AI Help Type for: {question['Problem_Name']}",
                                ["explanation", "hints", "optimization", "similar"],
                                key=f"skipped_help_type_{idx}",
                                format_func=lambda x: {
                                    "explanation": "📚 Explanation",
                                    "hints": "💡 Hints",
                                    "optimization": "🔧 Optimization",
                                    "similar": "🎯 Similar Problems"
                                }[x]
                            )
                            if st.button(f"Get AI Help for {question['Problem_Name']}", key=f"get_help_{idx}"):
                                with st.spinner("Generating AI help..."):
                                    help_content = get_ai_help(question, help_type)
                                    st.markdown(f"""
                                    <div style="background:#f9f9f9; padding:10px; border-left: 5px solid #888;">
                                        <strong>🤖 AI Assistant:</strong> {help_content}
                                    </div>
                                    """, unsafe_allow_html=True)
                    else:
                        st.warning(f"Question with index {idx} was skipped but no longer exists in the question bank.")
            else:
                st.info("🎉 No skipped questions to review.")

    else:
        st.info("Please upload an Excel file to start your coding practice session.")

# Run the app
if __name__ == '__main__':
    show_coding_practice_page()
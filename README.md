# 🔥 ForgeMe – AI-Powered Mock Interview Simulator

ForgeMe is a powerful AI-based mock interview platform that helps candidates prepare for real-world job interviews by simulating HR and technical interview rounds. It offers **real-time feedback** on verbal responses (speech) and non-verbal cues (eye contact, facial focus), helping you forge a more confident and job-ready version of yourself.

---

## 🎯 Features

### 🧠 AI-Powered Answer Evaluation
- Converts speech to text using `SpeechRecognition`.
- Uses **Gemini LLM** to analyze:
  - Answer relevance
  - Confidence and fluency
  - Key phrase and topic coverage
  - Improvement suggestions

### 👀 Real-Time Webcam Analysis
- Captures and analyzes **eye contact** via **OpenCV** and **MediaPipe**.
- Flags distraction, eyes movement, or camera avoidance in real time.

### 🎤 Speech Metrics
- Tracks speaking time, hesitations, and filler words.
- Highlights areas of improvement in tone and pace.

### 🧾 Candidate Profile Management
- Candidate must complete a short profile (name, email, career level, domain).
- Profiles and history stored securely in **SQLite**.

### 🖥️ Coding Question Practice 
- Pulls coding questions randomly from a spreadsheet dataset.
- Candidate answers in a timed environment.
- (Planned) LLM-based code quality feedback.

### ATS Score Checker and Company Research
- Candidate check Resume ATS Score based on ROles
- Company Research Easy
- Extra Help - Job Description Based Questions and Tips

### 📊 Performance Dashboard
- Visual charts for speaking activity and webcam consistency.
- Stores and displays past mock sessions.
- Personalized tips after each session.

---

## 🚀 Demo

> Coming Soon: [Link to Hugging Face / Streamlit Cloud Deployment]

---

## 🧱 Tech Stack

| Layer       | Tools Used                                      |
|-------------|-------------------------------------------------|
| UI          | Streamlit (custom components, layout, theming)  |
| Speech      | SpeechRecognition                               |
| Vision      | OpenCV, MediaPipe (eye tracking)                |
| LLM         | Cohere API, Gemini API                          |
| Database    | SQLite (user profiles, session history)         |
| Deployment  | Docker (PyAudio, OpenCV, MediaPipe support)     |

---

## 📦 Installation

> ⚠️ Requires Python 3.10+


# Clone the repository
git clone https://github.com/yourusername/forgeme.git
cd forgeme

# Install dependencies
pip install -r requirements.txt

# Start the app
streamlit run app.py

# Optional Build
docker build -t forgeme .
docker run -p 8501:8501 forgeme

# Configuration
GEMINI_API_KEY=your_key_here

# ✨ Inspiration
Interview prep is often stressful and unstructured. ForgeMe solves this by providing an intelligent, structured, and real-time simulated experience — so you're not just practicing, you're evolving.

# 👨‍💻 Author
Ashish Pathak 
🔗 LinkedIn •
💻 GitHub •
📧 Email: ashupathak22@gmail.com

📝 License
This project is licensed under the MIT License.



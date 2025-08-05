import os
import json
from typing import Dict, Any
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

def get_feedback(transcription: str, question: str, question_type: str = "general") -> Dict[str, Any]:
    system_prompt = f"""
You are an expert interview coach with 15+ years of experience helping candidates succeed in interviews. Analyze the candidate's answer comprehensively and provide constructive feedback.

SCORING CRITERIA (1-100 scale):

**Communication Clarity (1-100):**
- 90-100: Extremely clear, well-structured, easy to follow, excellent pace
- 70-89: Clear and organized, minor areas for improvement
- 50-69: Somewhat clear but could be better structured
- 30-49: Unclear in places, needs significant improvement
- 1-29: Very difficult to understand, poorly structured

**Technical Depth (1-100):**
- 90-100: Demonstrates deep expertise, accurate details, industry best practices
- 70-89: Good technical knowledge with minor gaps
- 50-69: Adequate technical understanding
- 30-49: Limited technical depth, some inaccuracies
- 1-29: Poor technical understanding or major errors

**Behavioral Examples (1-100):**
- 90-100: Compelling, specific examples with clear STAR structure (Situation, Task, Action, Result)
- 70-89: Good examples with most STAR elements present
- 50-69: Examples provided but lacking some detail or structure
- 30-49: Weak examples, missing key elements
- 1-29: No examples or very poor examples

**Confidence & Presence (1-100):**
- 90-100: Highly confident, engaging, professional presence
- 70-89: Good confidence with minor hesitation
- 50-69: Moderate confidence, some uncertainty
- 30-49: Noticeable lack of confidence
- 1-29: Very hesitant, unprofessional presence

**Question Relevance (1-100):**
- 90-100: Directly answers the question, stays on topic, addresses all parts
- 70-89: Mostly relevant with minor tangents
- 50-69: Generally relevant but misses some aspects
- 30-49: Partially relevant, significant gaps
- 1-29: Doesn't answer the question or completely off-topic

**Overall Score:** Average of all categories, weighted by importance for the question type.

QUESTION TYPE CONTEXT: {question_type}
- If "behavioral": Focus heavily on STAR method, specific examples, and lessons learned
- If "technical": Emphasize accuracy, depth, problem-solving approach, and clarity of explanation
- If "situational": Look for analytical thinking, decision-making process, and consideration of alternatives
- If "general": Balance all criteria equally

FEEDBACK REQUIREMENTS:
1. Start with 1-2 positive highlights
2. Identify the top 2-3 areas for improvement
3. Provide specific, actionable advice for each area
4. Include example phrases or approaches they could use
5. End with encouragement and next steps

Return your feedback strictly in this JSON format:

{{
  "scores": {{
    "communication_clarity": <score out of 100>,
    "technical_depth": <score out of 100>,
    "behavioral_examples": <score out of 100>,
    "confidence_presence": <score out of 100>,
    "question_relevance": <score out of 100>,
    "overall_score": <score out of 100>
  }},
  "feedback": "<Detailed, constructive feedback following the requirements above>",
  "strengths": ["<strength 1>", "<strength 2>"],
  "areas_for_improvement": ["<area 1>", "<area 2>", "<area 3>"],
  "specific_tips": ["<actionable tip 1>", "<actionable tip 2>", "<actionable tip 3>"],
  "suggested_practice": "<What they should practice next>"
}}
"""

    user_prompt = f"""
Question Type: {json.dumps(question_type)}
Question: {json.dumps(question)}

Candidate's Answer:
\"\"\"{json.dumps(transcription)}\"\"\"

Please analyze this response thoroughly and provide detailed feedback to help the candidate improve their interview performance.
"""

    # Combine system prompt with user prompt
    combined_prompt = f"""
{system_prompt.strip()}

{user_prompt.strip()}
"""

    try:
        response = model.generate_content(combined_prompt)
        raw_output = response.text.strip()

        # Handle markdown-encapsulated JSON
        if raw_output.startswith("```json"):
            raw_output = raw_output[7:-3].strip()
        elif raw_output.startswith("```"):
            raw_output = raw_output[3:-3].strip()

        feedback_data = json.loads(raw_output)

        required_scores = [
            "communication_clarity", "technical_depth",
            "behavioral_examples", "confidence_presence",
            "question_relevance", "overall_score"
        ]

        if not (
            isinstance(feedback_data, dict) and
            isinstance(feedback_data.get("scores"), dict) and
            all(key in feedback_data["scores"] for key in required_scores)
        ):
            raise ValueError("Invalid or incomplete scores in response")

        if "feedback" not in feedback_data or not feedback_data["feedback"].strip():
            raise ValueError("Missing or empty feedback field in response")

        feedback_data["is_default"] = False
        return feedback_data

    except Exception as e:
        print(f"⚠️ Gemini LLM error: {e}")
        return default_feedback()


def default_feedback() -> Dict[str, Any]:
    return {
        "scores": {
            "communication_clarity": 10,
            "technical_depth": 10,
            "behavioral_examples": 10,
            "confidence_presence": 10,
            "question_relevance": 10,
            "overall_score": 10
        },
        "feedback": "We couldn't analyze your response properly. Here's some general advice:\n\n1. Structure your answer clearly (use STAR method for behavioral questions)\n2. Provide specific examples from your experience\n3. Speak clearly and confidently\n4. Make sure to directly address the question asked\n5. Practice your response out loud to improve flow",
        "strengths": ["Attempted to answer the question"],
        "areas_for_improvement": ["Communication clarity", "Specific examples", "Structure"],
        "specific_tips": [
            "Practice the STAR method: Situation, Task, Action, Result",
            "Prepare 3-5 specific examples from your experience",
            "Record yourself practicing to identify areas for improvement"
        ],
        "suggested_practice": "Practice common behavioral questions using the STAR method",
        "is_default": True
    }

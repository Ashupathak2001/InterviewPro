import json
import logging
import os
from typing import Dict, Any
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")
logger = logging.getLogger(__name__)

def analyze_resume_text(resume_text: str, target_role: str = None) -> Dict[str, Any]:
    """
    Analyzes a resume using the Gemini API and returns structured feedback as a JSON object.
    """
    role_line = f"The user is targeting the role of **{target_role}**." if target_role else ""
    
    system_prompt = f"""
You are a career advisor analyzing a candidate's resume.
{role_line}

Here is the resume content:
\"\"\"
{resume_text}
\"\"\"

Give the response as a JSON with the following structure:
{{
  "overall_score": 0-100,
  "strengths": [...],
  "improvements": [...],
  "detailed_scores": {{
    "formatting": 0-100,
    "technical_skills": 0-100,
    "experience_relevance": 0-100,
    "clarity": 0-100,
    "impact": 0-100
  }},
  "suggestions": [...]
}}

Respond strictly in JSON format, with no additional text or Markdown.
"""

    try:
        response = model.generate_content(
            contents=system_prompt,
            generation_config=genai.GenerationConfig(response_mime_type="application/json")
        )
        
        # The API's response.text is the JSON string
        return json.loads(response.text)
    
    except Exception as e:
        logger.error(f"Error during LLM analysis or JSON parsing: {e}")
        # Return a default, empty dictionary in case of an error
        return {
            "overall_score": 0,
            "strengths": [],
            "improvements": [],
            "detailed_scores": {},
            "suggestions": []
        }
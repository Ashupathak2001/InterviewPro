import requests
import json
import logging
logger = logging.getLogger(__name__)

def analyze_resume_text(resume_text: str, target_role: str = None) -> dict:
    role_line = f"The user is targeting the role of **{target_role}**." if target_role else ""
    
    prompt = f"""
You are a career advisor analyzing a candidate's resume.
{role_line}

Here is the resume content:
\"\"\"
{resume_text}
\"\"\"

Give the response as a JSON with:
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

Respond strictly in JSON format.
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "tinyllama", "prompt": prompt, "stream": False}
    )
    
    try:
        return json.loads(response.json()["response"])
    except Exception as e:
        logger.error(f"LLM parsing error: {e}")
        return {
            "overall_score": 0,
            "strengths": [],
            "improvements": [],
            "detailed_scores": {},
            "suggestions": []
        }

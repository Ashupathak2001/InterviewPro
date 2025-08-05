import logging
from typing import List, Tuple
import json
import os
import google.generativeai as genai
import dotenv
dotenv.load_dotenv()

logger = logging.getLogger(__name__)

# Configure the Gemini API key from an environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")


def generate_jd_questions_and_tips(job_description_text: str) -> Tuple[List[str], List[str]]:
    """
    Generate tailored interview questions and tips based on job description text using Gemini LLM.

    Args:
        job_description_text (str): The text of the job description.

    Returns:
        Tuple[List[str], List[str]]: Questions and tips derived from the job description.
    """
    try:
        # Use json.dumps to safely format the job description
        job_description_cleaned = json.dumps(job_description_text.strip())

        # Use structured prompt design
        system_prompt = (
            "You are a professional interview coach. Analyze the job description and generate:"
            "\n1. 5 to 7 tailored interview questions specific to the role."
            "\n2. 3 to 4 actionable tips to help the candidate prepare."
            "\n\nReturn ONLY a valid JSON in this format:\n"
            "{\n"
            '  "questions": ["..."],\n'
            '  "tips": ["..."]\n'
            "}"
        )

        user_prompt = f"Job Description:\n{job_description_cleaned}"

        response = model.generate_content([system_prompt, user_prompt])

        raw_output = response.text.strip()

        # Handle code block wrapping
        if raw_output.startswith("```json"):
            raw_output = raw_output[7:-3].strip()
        elif raw_output.startswith("```"):
            raw_output = raw_output[3:-3].strip()

        # Try parsing as JSON
        try:
            parsed_output = json.loads(raw_output)
            questions = parsed_output.get("questions", [])
            tips = parsed_output.get("tips", [])

            if not isinstance(questions, list) or not isinstance(tips, list):
                raise ValueError("Invalid format: 'questions' or 'tips' not a list.")

            return questions, tips

        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON parsing failed: {e}\nOutput: {raw_output}")
            return [], []

    except Exception as e:
        logger.error(f"❌ Error generating questions and tips: {e}")
        return [], []

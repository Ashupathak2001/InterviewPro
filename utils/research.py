import requests
import os
from typing import Dict, Any
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")


def generate_company_insights(company_name: str) -> Dict[str, Any]:
    """
    Generates a structured company overview using the Gemini API.
    """
    system_prompt = f'''
**Company Overview for {company_name}**
**Brief Overview:**
- Provide a concise summary of the company's mission, vision, and values.
**Key Products or Services:**
- List the main products or services offered by the company.
**Common Roles/Departments:**
- Identify typical roles or departments candidates might apply to.
**Common Interview Questions:**
- Provide a list of common interview questions candidates might face.
**Latest Industry Trends:**
- Discuss recent trends in the industry that may impact the company.
**Strategic Questions to Ask the Interviewer:**
- Suggest insightful questions candidates can ask during the interview.

Ensure the response is well-structured, informative, and relevant to the company.
'''
    try:
        response = model.generate_content(
            contents=system_prompt
        )
        return {
            "company_name": company_name,
            "insights": response.text.strip()
        }
    except Exception as e:
        # It's good practice to handle potential errors from the API call
        print(f"An error occurred while generating insights: {e}")
        return {
            "company_name": company_name,
            "insights": "Could not retrieve company insights at this time."
        }
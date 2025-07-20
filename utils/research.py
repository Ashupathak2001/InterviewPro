import requests
def generate_company_insights(company_name: str) -> str:
    prompt = f'''
You are a career assistant helping a candidate prepare for an interview.

Provide a structured company overview for **{company_name}** that includes:
1. Brief Overview
2. Key Products or Services
3. Common Roles/Departments
4. Common Interview Questions
5. Latest Industry Trends affecting this company
6. Strategic Questions to Ask the Interviewer

Give the response in a clean Markdown-style format.
'''

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "tinyllama", "prompt": prompt, "stream": False}
    )
    return response.json()["response"].strip()

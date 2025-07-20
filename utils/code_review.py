import requests
import cohere
import os
from typing import Optional

# Constants
TINYLLAMA_API_URL = "http://localhost:11434/api/chat"
COHERE_MODEL = "command-r"
TINYLLAMA_MODEL = "tinyllama"
MAX_TOKENS = 500
TEMPERATURE = 0.3

# Initialize Cohere client
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY environment variable not set")
co = cohere.Client(COHERE_API_KEY)


def get_feedback_tinyllama(code: str) -> str:
    """
    Get code feedback from TinyLLaMA model.
    
    Args:
        code: The code to be reviewed
        
    Returns:
        Feedback as a string or error message
    """
    prompt = f"""You are a senior engineer. Review this Python code and provide detailed feedback:
    
    Code:
    ```
    {code}
    ```
    
    Provide feedback on:
    - Correctness
    - Readability
    - Performance
    - Potential improvements
    
    Format your response as bullet points."""

    payload = {
        "model": TINYLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful code reviewer."},
            {"role": "user", "content": prompt}
        ]
    }

    try:
        response = requests.post(TINYLLAMA_API_URL, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"Error communicating with TinyLLaMA API: {str(e)}"
    except (KeyError, ValueError) as e:
        return f"Error processing TinyLLaMA response: {str(e)}"


def get_feedback_cohere(code: str) -> str:
    """
    Get code feedback from Cohere model.
    
    Args:
        code: The code to be reviewed
        
    Returns:
        Feedback as a string or error message
    """
    prompt = f"""You are a senior engineer reviewing Python code. Analyze for:
    - Correctness
    - Readability
    - Complexity
    - Potential improvements
    
    Code:
    ```
    {code}
    ```
    
    Provide detailed feedback in bullet points."""

    try:
        response = co.generate(
            model=COHERE_MODEL,
            prompt=prompt,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE
        )
        return response.generations[0].text.strip()
    except cohere.CohereError as e:
        return f"Cohere API error: {str(e)}"
    except Exception as e:
        return f"Unexpected error with Cohere: {str(e)}"
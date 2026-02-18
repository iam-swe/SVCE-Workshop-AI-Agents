"""
Mood and intent detection utilities.
"""

import os
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI


def get_llm(temperature: float = 0.0) -> Any:
    """Get an LLM instance for detection tasks using Gemini 2.5 Flash."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Please set GOOGLE_API_KEY in your .env file")
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=temperature)


INTENT_DETECTOR_PROMPT = """Analyze the user's message and determine their requirement.

User message: "{message}"

Classify as one of:
- explain: explain, teach, help me understand, what is 
- learn: create a document, summarise, revision, material 

Output ONLY one word: explain, or learn"""

def detect_intent(message: str) -> str:
    """Detect user intent from their message.

    Args:
        message: The user's message text

    Returns:
        One of: 'explain', 'learn', or 'unknown'
    """
    try:
        llm = get_llm(temperature=0)
        response = llm.invoke([
            HumanMessage(content=INTENT_DETECTOR_PROMPT.format(message=message))
        ])
        intent = response.content.strip().lower()
        if intent in ["explain", "learn"]:
            return intent
        return "unknown"
    except Exception:
        return "unknown"

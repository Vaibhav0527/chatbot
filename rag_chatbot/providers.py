import os

from langchain_google_genai import ChatGoogleGenerativeAI


def build_chat_model(temperature: float = 0.0):
    """Build a Google Gemini chat model from the configured API key."""
    if os.getenv("GOOGLE_API_KEY"):
        model = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
        return ChatGoogleGenerativeAI(model=model, temperature=temperature)

    raise ValueError(
        "No GOOGLE_API_KEY found. Set GOOGLE_API_KEY in your environment."
    )

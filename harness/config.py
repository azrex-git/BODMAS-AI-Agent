"""
Configuration settings for the BODMAS Tutor Agent Harness.
"""

import os
from dotenv import load_dotenv

# Load local environment variables if .env exists
load_dotenv()

# Primary lightweight model: Gemma 4 MoE (4B active parameters)
DEFAULT_MODEL = os.getenv("GEMMA_MODEL", "gemma-4-26b-a4b-it")

# Supported models list for UI and benchmarking
SUPPORTED_MODELS = [
    {"id": "gemma-4-26b-a4b-it", "name": "Gemma 4 MoE (26B / 4B active) [Lightweight]", "is_default": True},
    {"id": "gemma-4-31b-it", "name": "Gemma 4 (31B Dense)", "is_default": False},
    {"id": "gemma-2-9b-it", "name": "Gemma 2 (9B)", "is_default": False},
    {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash", "is_default": False},
]

# API Key lookup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

def get_api_key(override_key: str | None = None) -> str:
    """Returns the effective API key, reloading from .env dynamically."""
    if override_key and override_key.strip():
        return override_key.strip()
    load_dotenv(override=True)
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if key in ("", "your_api_key_here", "AIzaSyYourActualGeminiApiKeyHere"):
        return ""
    return key

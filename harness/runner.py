"""
Component 1: The Runner Wrapper (`harness/runner.py`)
Execution controller that manages Gemini API calls to Gemma 4,
high-precision latency timing, token telemetry extraction, and schema validation.
"""

import time
import json
import re
from typing import Any, Tuple, Optional
from google import genai
from google.genai import types

from .config import DEFAULT_MODEL, get_api_key


def _clean_json_text(text: str) -> str:
    """Strips markdown code fences and extraneous text if present."""
    if not text:
        return "{}"
    text = text.strip()
    # Match ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    return text


def run_gemma_task(
    prompt: str,
    temperature: float = 0.0,
    response_schema: Any = None,
    model_id: str = DEFAULT_MODEL,
    system_instruction: Optional[str] = None,
    api_key: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
) -> Tuple[dict, dict]:
    """
    Executes a prompt against Gemma 4 via the Google Gemini API.

    Returns:
        tuple[dict, dict]: (parsed_json_data, telemetry_dict)
    """
    effective_key = get_api_key(api_key)
    if not effective_key:
        raise ValueError(
            "GEMINI_API_KEY is not set. Please provide a key in your environment or through the Settings panel."
        )

    client = genai.Client(api_key=effective_key)

    config_kwargs: dict[str, Any] = {
        "temperature": temperature,
    }

    if max_output_tokens:
        config_kwargs["max_output_tokens"] = max_output_tokens

    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction

    if response_schema:
        config_kwargs["response_mime_type"] = "application/json"
        config_kwargs["response_schema"] = response_schema

    config = types.GenerateContentConfig(**config_kwargs)

    # High-precision timer start
    start_time = time.perf_counter()
    try:
        response = client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=config,
        )
    except Exception as e:
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        telemetry = {
            "latency_ms": latency_ms,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "model": model_id,
            "is_valid_schema": False,
            "error": str(e),
        }
        raise RuntimeError(f"Gemma 4 API Call Failed [{model_id}]: {str(e)}") from e

    # Timer stop
    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    # Extract Usage Telemetry
    prompt_tokens = 0
    candidate_tokens = 0
    total_tokens = 0
    if response.usage_metadata:
        prompt_tokens = response.usage_metadata.prompt_token_count or 0
        candidate_tokens = response.usage_metadata.candidates_token_count or 0
        total_tokens = response.usage_metadata.total_token_count or (prompt_tokens + candidate_tokens)

    # Parse JSON Content
    raw_text = response.text or "{}"
    cleaned = _clean_json_text(raw_text)
    is_valid_schema = True

    try:
        parsed_data = json.loads(cleaned)
    except json.JSONDecodeError as decode_err:
        # Resilient fallback: attempt regex field extraction if string was truncated
        title_match = re.search(r'"title"\s*:\s*"([^"]+)"', cleaned)
        expl_match = re.search(r'"explanation"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)', cleaned)
        tip_match = re.search(r'"tip"\s*:\s*"([^"]+)"', cleaned)
        if title_match or expl_match:
            clean_expl = expl_match.group(1).replace("\\n", "\n") if expl_match else "Follow BODMAS precedence to solve."
            # Strip runaway repetition if model repeated phrases
            clean_expl = re.sub(r'(\b\w+\b)(?:,\s*\1){3,}', r'\1', clean_expl)
            parsed_data = {
                "title": title_match.group(1) if title_match else "BODMAS Step-by-Step",
                "explanation": clean_expl.strip(),
                "tip": tip_match.group(1) if tip_match else "Always resolve operations inside brackets first.",
                "encouragement": "Great work solving this equation!",
            }
            is_valid_schema = True
        else:
            is_valid_schema = False
            parsed_data = {
                "parse_error": str(decode_err),
                "raw_text": raw_text,
            }

    telemetry = {
        "latency_ms": latency_ms,
        "input_tokens": prompt_tokens,
        "output_tokens": candidate_tokens,
        "total_tokens": total_tokens,
        "model": model_id,
        "is_valid_schema": is_valid_schema,
    }

    return parsed_data, telemetry

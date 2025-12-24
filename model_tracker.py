
"""
Model usage tracking module.

This module provides a wrapper around LangChain's ChatGoogleGenerativeAI that
automatically tracks API calls and token usage per agent/model combination.
Usage statistics are persisted to a JSON file for reporting purposes.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Base dir (project root)
BASE_DIR = Path(__file__).resolve().parent

# Load .env if present (non-fatal if missing)
# This allows API keys to be stored in a .env file without breaking if missing
load_dotenv(BASE_DIR / ".env", override=False)

# File where model usage is aggregated across processes
# Format: {agent_name: {model_name: {"numApiCalls": int, "totalTokens": int}}}
_USAGE_FILE = BASE_DIR / "model_usage.json"


def _load_usage() -> Dict[str, Dict[str, int]]:
    """
    Load usage statistics from the JSON file.
    
    Returns:
        Dictionary mapping agent names to their model usage stats.
        Returns empty dict if file doesn't exist or is corrupted.
    """
    if _USAGE_FILE.exists():
        try:
            return json.loads(_USAGE_FILE.read_text())
        except Exception:
            # If file is corrupted, return empty dict and let it be overwritten
            return {}
    return {}


def _save_usage(data: Dict[str, Dict[str, int]]) -> None:
    """
    Save usage statistics to the JSON file.
    
    Args:
        data: Dictionary containing usage statistics to persist
    """
    _USAGE_FILE.write_text(json.dumps(data, indent=2))


def _estimate_tokens(text: str) -> int:
    """
    Very rough token estimate using word count.
    
    Note: This is a simplified approximation for assignment purposes only.
    Real tokenization would require the model's tokenizer.
    
    Args:
        text: Input text to estimate tokens for
        
    Returns:
        Estimated token count (at least 1)
    """
    words = text.split()
    return max(len(words), 1)


def _extract_text(result: Any) -> str:
    """
    Try to extract a text string from a LangChain AIMessage or any model response.
    
    Handles various response formats:
    - String content directly
    - List of content parts (may contain dicts with "text" keys)
    - Other object types (converted to string)
    
    Args:
        result: The result object from an LLM invocation
        
    Returns:
        Extracted text as a string
    """
    content = getattr(result, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Handle multi-part content (e.g., text + images)
        parts = []
        for part in content:
            if isinstance(part, dict):
                if "text" in part:
                    parts.append(part["text"])
            else:
                parts.append(str(part))
        return "\n".join(parts)
    if content is not None:
        return str(content)
    return str(result)


def _update_usage(agent_name: str, model_name: str, num_tokens: int) -> None:
    """
    Update usage statistics for a specific agent and model.
    
    Increments the API call count and adds to the total token count.
    This function is thread-safe at the file level (though not perfect for
    concurrent writes, it's sufficient for this use case).
    
    Args:
        agent_name: Name of the agent making the call (e.g., "CodeGenerator")
        model_name: Name of the model being used (e.g., "gemini-2.5-flash")
        num_tokens: Number of tokens used in this call
    """
    usage = _load_usage()

    # Initialize agent entry if it doesn't exist
    if agent_name not in usage:
        usage[agent_name] = {}

    # Get or create model stats for this agent
    model_stats = usage[agent_name].get(model_name, {"numApiCalls": 0, "totalTokens": 0})
    model_stats["numApiCalls"] += 1
    model_stats["totalTokens"] += int(num_tokens)

    # Update and persist
    usage[agent_name][model_name] = model_stats
    _save_usage(usage)


class TrackingChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    """
    Subclass of ChatGoogleGenerativeAI that tracks usage stats and handles API key.
    
    This wrapper automatically tracks all LLM invocations, recording:
    - Number of API calls per agent/model
    - Total tokens used per agent/model
    
    The agent_name parameter allows different parts of the system to be tracked separately.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the tracking LLM wrapper.
        
        Automatically handles API key lookup from environment variables if not
        explicitly provided. Checks GOOGLE_API_KEY and GEMINI_API_KEY env vars.
        """
        # Prefer explicit key passed via kwargs, otherwise pull from env
        google_key = (
            kwargs.get("google_api_key")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("GEMINI_API_KEY")
        )
        if google_key:
            kwargs["google_api_key"] = google_key
        # If no key, super() will raise a clear error which we catch in the tools
        super().__init__(*args, **kwargs)

    def invoke(self, *args, agent_name="UnknownAgent",  **kwargs):
        """
        Synchronous invocation with usage tracking.
        
        Args:
            *args: Arguments passed to the parent invoke method
            agent_name: Name of the agent making this call (for tracking)
            **kwargs: Keyword arguments passed to the parent invoke method
            
        Returns:
            The result from the LLM invocation
        """
        result = super().invoke(*args, **kwargs)
        text = _extract_text(result)
        tokens = _estimate_tokens(text)
        _update_usage(agent_name, self.model, tokens)
        return result

    async def ainvoke(self, *args, agent_name="UnknownAgent", **kwargs):
        """
        Asynchronous invocation with usage tracking.
        
        Args:
            *args: Arguments passed to the parent ainvoke method
            agent_name: Name of the agent making this call (for tracking)
            **kwargs: Keyword arguments passed to the parent ainvoke method
            
        Returns:
            The result from the LLM invocation
        """
        result = await super().ainvoke(*args, **kwargs)
        text = _extract_text(result)
        tokens = _estimate_tokens(text)
        _update_usage(agent_name, self.model, tokens)
        return result


def get_model_usage() -> Dict[str, Dict[str, int]]:
    """Public accessor to model usage JSON."""
    return _load_usage()

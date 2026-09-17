"""
LLM provider setup for DecisionFlow AI.

Everything provider-specific lives in this one file. Chains never import an
LLM class directly - they receive an already-configured `llm` object from
here, so the underlying provider can be changed later by editing only this
file (and the .env values).

Configuration is read from environment variables (see .env.example):

    LLM_API_KEY      - required. API key for your LLM provider.
    LLM_MODEL        - optional. Defaults to "gpt-4o-mini".
    LLM_BASE_URL     - optional. Set this to use an OpenAI-compatible
                        provider other than OpenAI itself (e.g. Groq,
                        Together AI, OpenRouter). Leave unset for OpenAI.
    LLM_TEMPERATURE  - optional. Defaults to 0.3.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load variables from a local .env file, if present.
load_dotenv()


def get_llm() -> ChatOpenAI:
    """
    Build and return the chat model used by every chain in the pipeline.

    Uses the OpenAI-compatible ChatOpenAI client. Setting LLM_BASE_URL lets
    you point this at any OpenAI-compatible endpoint (Groq, Together AI,
    OpenRouter, etc.) without changing any chain code.
    """
    api_key = os.getenv("LLM_API_KEY")
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
    base_url = os.getenv("LLM_BASE_URL")  # None means "use OpenAI's default endpoint"
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))

    if not api_key:
        raise ValueError(
            "LLM_API_KEY is not set. Create a .env file in the project root "
            "(copy .env.example) and set LLM_API_KEY to your provider's API key."
        )

    llm_kwargs = {
        "model": model_name,
        "api_key": api_key,
        "temperature": temperature,
    }
    if base_url:
        llm_kwargs["base_url"] = base_url

    return ChatOpenAI(**llm_kwargs)

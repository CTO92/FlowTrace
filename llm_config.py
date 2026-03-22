"""
FlowTrace Multi-LLM Configuration

Central factory for all LLM clients. Supports Grok (xAI), Claude (Anthropic),
Gemini (Google), and OpenAI. Traders can configure a default provider and
optionally assign different providers to individual agents.

Configuration lives in llm_config.json. API keys come from environment variables.
"""

import os
import json
import re
import logging
from typing import Optional

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "llm_config.json")

# Cache for instantiated clients
_langchain_cache = {}
_async_client_cache = {}
_sync_client_cache = {}
_config_cache = None


# ---------------------------------------------------------------------------
# Configuration Loading
# ---------------------------------------------------------------------------

def load_llm_config() -> dict:
    """Load and cache the LLM configuration."""
    global _config_cache
    if _config_cache is not None:
        return _config_cache

    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            _config_cache = json.load(f)
    else:
        _config_cache = {
            "default_provider": "xai",
            "default_model": "grok-beta",
            "providers": {},
            "agent_assignments": {},
        }

    return _config_cache


def save_llm_config(config: dict) -> None:
    """Save LLM configuration to disk."""
    global _config_cache
    _config_cache = config
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def reload_config() -> dict:
    """Force reload configuration from disk (clears all caches)."""
    global _config_cache, _langchain_cache, _async_client_cache, _sync_client_cache
    _config_cache = None
    _langchain_cache.clear()
    _async_client_cache.clear()
    _sync_client_cache.clear()
    return load_llm_config()


def _resolve_provider_and_model(agent_type: str = None) -> tuple[str, str]:
    """
    Determine which provider and model to use for a given agent type.
    Falls back to defaults if no per-agent assignment exists.
    Returns (provider_name, model_name).
    """
    config = load_llm_config()
    default_provider = config.get("default_provider", "xai")
    default_model = config.get("default_model", "grok-beta")

    if agent_type:
        assignment = config.get("agent_assignments", {}).get(agent_type)
        if assignment and isinstance(assignment, dict):
            return (
                assignment.get("provider", default_provider),
                assignment.get("model", default_model),
            )

    return default_provider, default_model


def _get_provider_config(provider_name: str) -> dict:
    """Get the configuration for a specific provider."""
    config = load_llm_config()
    return config.get("providers", {}).get(provider_name, {})


def _get_api_key(provider_name: str) -> Optional[str]:
    """Retrieve the API key for a provider from environment variables."""
    provider = _get_provider_config(provider_name)
    env_var = provider.get("api_key_env")
    if env_var:
        return os.getenv(env_var)
    return None


def is_llm_configured(agent_type: str = None) -> bool:
    """Check if an LLM is configured and has a valid API key for the given agent type."""
    provider, _ = _resolve_provider_and_model(agent_type)
    return _get_api_key(provider) is not None


# ---------------------------------------------------------------------------
# LangChain Chat Model Factory
# ---------------------------------------------------------------------------

def get_langchain_llm(agent_type: str = None, temperature: float = 0):
    """
    Get a LangChain chat model instance for the given agent type.
    Returns ChatOpenAI, ChatAnthropic, or ChatGoogleGenerativeAI
    depending on the configured provider.
    """
    provider_name, model_name = _resolve_provider_and_model(agent_type)
    cache_key = f"{provider_name}:{model_name}:{temperature}"

    if cache_key in _langchain_cache:
        return _langchain_cache[cache_key]

    provider = _get_provider_config(provider_name)
    api_key = _get_api_key(provider_name)

    if not api_key:
        raise ValueError(
            f"API key not set for provider '{provider_name}'. "
            f"Set the {provider.get('api_key_env', 'UNKNOWN')} environment variable."
        )

    if provider_name in ("xai", "openai"):
        from langchain_openai import ChatOpenAI
        kwargs = {
            "api_key": api_key,
            "model": model_name,
            "temperature": temperature,
        }
        base_url = provider.get("base_url")
        if base_url:
            kwargs["base_url"] = base_url
        llm = ChatOpenAI(**kwargs)

    elif provider_name == "anthropic":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
            anthropic_api_key=api_key,
            model=model_name,
            temperature=temperature,
        )

    elif provider_name == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=model_name,
            temperature=temperature,
        )

    else:
        raise ValueError(f"Unknown LLM provider: {provider_name}")

    _langchain_cache[cache_key] = llm
    logger.debug(f"Created LangChain LLM: {provider_name}/{model_name} (agent={agent_type})")
    return llm


def get_vision_llm(temperature: float = 0):
    """
    Get a LangChain chat model configured for vision tasks.
    Uses the provider's vision_model field.
    """
    config = load_llm_config()
    default_provider = config.get("default_provider", "xai")

    # Check for vision-specific assignment
    assignment = config.get("agent_assignments", {}).get("VisionModel")
    if assignment and isinstance(assignment, dict):
        provider_name = assignment.get("provider", default_provider)
    else:
        provider_name = default_provider

    provider = _get_provider_config(provider_name)
    if not provider.get("supports_vision", False):
        logger.warning(f"Provider '{provider_name}' does not support vision. Falling back to default.")
        provider_name = default_provider
        provider = _get_provider_config(provider_name)

    vision_model = provider.get("vision_model", provider.get("default_model"))
    api_key = _get_api_key(provider_name)

    if not api_key:
        raise ValueError(f"API key not set for vision provider '{provider_name}'.")

    if provider_name in ("xai", "openai"):
        from langchain_openai import ChatOpenAI
        kwargs = {"api_key": api_key, "model": vision_model, "temperature": temperature}
        base_url = provider.get("base_url")
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOpenAI(**kwargs)

    elif provider_name == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            anthropic_api_key=api_key,
            model=vision_model,
            temperature=temperature,
        )

    elif provider_name == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=vision_model,
            temperature=temperature,
        )

    raise ValueError(f"Unknown vision provider: {provider_name}")


# ---------------------------------------------------------------------------
# Unified Async Completion (for raw client usage)
# ---------------------------------------------------------------------------

async def async_chat_completion(
    messages: list[dict],
    agent_type: str = None,
    temperature: float = 0.2,
    json_mode: bool = False,
) -> str:
    """
    Unified async chat completion that works across all providers.
    Returns the raw content string from the LLM response.

    This replaces direct AsyncOpenAI client calls in grok_analysis.py
    and agent_debate.py, handling SDK differences internally.
    """
    provider_name, model_name = _resolve_provider_and_model(agent_type)
    api_key = _get_api_key(provider_name)

    if not api_key:
        raise ValueError(f"API key not set for provider '{provider_name}'.")

    if provider_name in ("xai", "openai"):
        return await _async_openai_completion(
            provider_name, api_key, model_name, messages, temperature, json_mode
        )
    elif provider_name == "anthropic":
        return await _async_anthropic_completion(
            api_key, model_name, messages, temperature, json_mode
        )
    elif provider_name == "google":
        return await _async_google_completion(
            api_key, model_name, messages, temperature, json_mode
        )
    else:
        raise ValueError(f"Unknown provider: {provider_name}")


async def _async_openai_completion(
    provider_name, api_key, model, messages, temperature, json_mode
) -> str:
    """OpenAI-compatible async completion (works for OpenAI and xAI/Grok)."""
    from openai import AsyncOpenAI

    cache_key = f"async:{provider_name}"
    if cache_key not in _async_client_cache:
        provider = _get_provider_config(provider_name)
        kwargs = {"api_key": api_key}
        base_url = provider.get("base_url")
        if base_url:
            kwargs["base_url"] = base_url
        _async_client_cache[cache_key] = AsyncOpenAI(**kwargs)

    client = _async_client_cache[cache_key]
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = await client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


async def _async_anthropic_completion(
    api_key, model, messages, temperature, json_mode
) -> str:
    """Anthropic Claude async completion."""
    import anthropic

    cache_key = "async:anthropic"
    if cache_key not in _async_client_cache:
        _async_client_cache[cache_key] = anthropic.AsyncAnthropic(api_key=api_key)

    client = _async_client_cache[cache_key]

    # Separate system message from conversation messages
    system_text = ""
    conversation = []
    for msg in messages:
        if msg["role"] == "system":
            system_text += msg["content"] + "\n"
        else:
            conversation.append(msg)

    # Anthropic doesn't have native JSON mode — reinforce in system prompt
    if json_mode and "json" not in system_text.lower():
        system_text += "\nYou MUST respond with valid JSON only. No markdown, no explanation."

    response = await client.messages.create(
        model=model,
        max_tokens=4096,
        temperature=temperature,
        system=system_text.strip(),
        messages=conversation,
    )

    content = response.content[0].text

    # Extract JSON if wrapped in code fences
    if json_mode:
        content = _extract_json(content)

    return content


async def _async_google_completion(
    api_key, model, messages, temperature, json_mode
) -> str:
    """Google Gemini async completion."""
    import google.generativeai as genai

    cache_key = "async:google"
    if cache_key not in _async_client_cache:
        genai.configure(api_key=api_key)
        _async_client_cache[cache_key] = True  # just mark as configured

    gen_model = genai.GenerativeModel(model)

    # Convert OpenAI-style messages to Gemini format
    prompt_parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            prompt_parts.append(f"System: {content}")
        elif role == "user":
            prompt_parts.append(f"User: {content}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}")

    if json_mode:
        prompt_parts.append("Respond with valid JSON only. No markdown, no explanation.")

    combined = "\n\n".join(prompt_parts)

    config = {"temperature": temperature}
    if json_mode:
        config["response_mime_type"] = "application/json"

    response = await gen_model.generate_content_async(
        combined,
        generation_config=config,
    )

    return response.text


# ---------------------------------------------------------------------------
# Unified Sync Completion (for update_knowledge_graph.py)
# ---------------------------------------------------------------------------

def sync_chat_completion(
    messages: list[dict],
    agent_type: str = None,
    temperature: float = 0.2,
    json_mode: bool = False,
) -> str:
    """
    Synchronous version of async_chat_completion.
    For use in non-async contexts like update_knowledge_graph.py.
    """
    provider_name, model_name = _resolve_provider_and_model(agent_type)
    api_key = _get_api_key(provider_name)

    if not api_key:
        raise ValueError(f"API key not set for provider '{provider_name}'.")

    if provider_name in ("xai", "openai"):
        from openai import OpenAI

        provider = _get_provider_config(provider_name)
        kwargs = {"api_key": api_key}
        base_url = provider.get("base_url")
        if base_url:
            kwargs["base_url"] = base_url

        client = OpenAI(**kwargs)
        req_kwargs = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            req_kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**req_kwargs)
        return response.choices[0].message.content

    elif provider_name == "anthropic":
        import anthropic

        system_text = ""
        conversation = []
        for msg in messages:
            if msg["role"] == "system":
                system_text += msg["content"] + "\n"
            else:
                conversation.append(msg)

        if json_mode and "json" not in system_text.lower():
            system_text += "\nYou MUST respond with valid JSON only."

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model_name,
            max_tokens=4096,
            temperature=temperature,
            system=system_text.strip(),
            messages=conversation,
        )
        content = response.content[0].text
        return _extract_json(content) if json_mode else content

    elif provider_name == "google":
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        gen_model = genai.GenerativeModel(model_name)

        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        if json_mode:
            prompt_parts.append("Respond with valid JSON only.")

        config = {"temperature": temperature}
        if json_mode:
            config["response_mime_type"] = "application/json"

        response = gen_model.generate_content(
            "\n\n".join(prompt_parts),
            generation_config=config,
        )
        return response.text

    raise ValueError(f"Unknown provider: {provider_name}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_json(content: str) -> str:
    """
    Extract JSON from content that may be wrapped in markdown code fences.
    Anthropic sometimes returns ```json\n{...}\n``` instead of raw JSON.
    """
    # Try to find JSON in code fences
    match = re.search(r"```(?:json)?\s*\n?([\s\S]*?)\n?```", content)
    if match:
        return match.group(1).strip()

    # Try to find raw JSON object
    match = re.search(r"\{[\s\S]*\}", content)
    if match:
        return match.group(0).strip()

    return content.strip()


def get_available_providers() -> dict:
    """
    Return a dict of providers that have API keys configured.
    Useful for UI display.
    """
    config = load_llm_config()
    available = {}
    for name, provider in config.get("providers", {}).items():
        env_var = provider.get("api_key_env", "")
        has_key = bool(os.getenv(env_var))
        available[name] = {
            "configured": has_key,
            "env_var": env_var,
            "default_model": provider.get("default_model"),
            "supports_vision": provider.get("supports_vision", False),
        }
    return available


def get_agent_assignments() -> dict:
    """Return the current per-agent provider assignments."""
    config = load_llm_config()
    return config.get("agent_assignments", {})

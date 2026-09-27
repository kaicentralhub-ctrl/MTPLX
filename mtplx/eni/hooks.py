"""ENI Server Hooks - Patches for MTPLX server/openai.py.

Provides a wrapper around message processing to inject ENI memories
before tokenization/generation.
"""

from __future__ import annotations

import logging
from functools import wraps
from typing import Any, Callable

from .config import get_config, is_enabled
from .enrichment import enrich_messages, compute_injection_hash


log = logging.getLogger("mtplx.eni.hooks")


# Global state for injection tracking
_current_injection_hash: str = "empty"


def get_current_injection_hash() -> str:
    """Get the injection hash from the most recent enrichment."""
    return _current_injection_hash


def enrich_messages_for_generation(
    messages: list[dict[str, Any]],
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    """Enrich messages with ENI memories before generation.
    
    Called from openai.py just before _encode_messages().
    Returns enriched messages with relevant memories injected into system prompt.
    """
    global _current_injection_hash
    
    if not is_enabled():
        _current_injection_hash = "empty"
        return messages
    
    config = get_config()
    
    try:
        enriched, injected_ids = enrich_messages(messages, config, session_id)
        _current_injection_hash = compute_injection_hash(injected_ids)
        
        if injected_ids:
            log.info(f"[ENI] Injected {len(injected_ids)} memories, hash={_current_injection_hash[:8]}")
        
        return enriched
    except Exception as e:
        log.warning(f"[ENI] Enrichment failed, using original messages: {e}")
        _current_injection_hash = "empty"
        return messages


def patch_encode_messages(original_encode: Callable) -> Callable:
    """Decorator to wrap _encode_messages with ENI enrichment.
    
    Usage in openai.py:
        from mtplx.eni.hooks import patch_encode_messages
        _encode_messages = patch_encode_messages(_encode_messages)
    """
    @wraps(original_encode)
    def wrapped_encode(
        tokenizer: Any,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> list[int]:
        # Enrich messages before encoding
        enriched = enrich_messages_for_generation(messages)
        return original_encode(tokenizer, enriched, **kwargs)
    
    return wrapped_encode


def install_hooks() -> None:
    """Install ENI hooks into MTPLX server.
    
    This is called at module load time to patch the server.
    Alternative to decorator-based patching.
    """
    if not is_enabled():
        log.debug("[ENI] Hooks disabled")
        return
    
    try:
        from mtplx.server import openai as server_module
        
        # Patch _encode_messages if it exists
        if hasattr(server_module, "_encode_messages"):
            original = server_module._encode_messages
            server_module._encode_messages = patch_encode_messages(original)
            log.info("[ENI] Installed _encode_messages hook")
        else:
            log.warning("[ENI] _encode_messages not found in server module")
    except ImportError as e:
        log.debug(f"[ENI] Server module not available: {e}")
    except Exception as e:
        log.warning(f"[ENI] Failed to install hooks: {e}")


# Stats for monitoring
_stats = {
    "enrichments": 0,
    "skipped": 0,
    "errors": 0,
}


def get_hook_stats() -> dict[str, Any]:
    """Get hook statistics."""
    return {
        **_stats,
        "enabled": is_enabled(),
        "current_hash": _current_injection_hash,
    }

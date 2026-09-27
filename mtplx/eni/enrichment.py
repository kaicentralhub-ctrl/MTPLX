"""ENI Message Enrichment - Memory injection into context.

Provides:
- Memory retrieval based on query
- Message enrichment with relevant memories
- Injection hash computation for session caching
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any

from .config import get_config, ENIConfig
from .memory_client import get_client, Memory


log = logging.getLogger("mtplx.eni.enrichment")


def compute_injection_hash(memory_ids: list[str]) -> str:
    """Compute deterministic hash of injected memory IDs.
    
    Used by session_bank to detect when the same tokens have different
    memory context, preventing incorrect cache restoration.
    """
    if not memory_ids:
        return "empty"
    sorted_ids = sorted(memory_ids)
    return hashlib.sha256("|".join(sorted_ids).encode()).hexdigest()[:16]


def _extract_query(messages: list[dict[str, Any]]) -> str:
    """Extract the query from messages for memory retrieval."""
    # Find the last user message
    for msg in reversed(messages):
        role = msg.get("role", "")
        if role == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                # Handle multi-part content (Roo Code format)
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            if content:
                # Use first 2000 chars for retrieval
                return content[:2000]
    return ""


def _format_memory_block(memories: list[Memory], max_chars: int) -> str:
    """Format memories into a context block."""
    if not memories:
        return ""
    
    lines = ["<ENI_MEMORIES>"]
    total_chars = len(lines[0])
    
    for mem in memories:
        # Format each memory
        mem_text = f"\n[{mem.category.upper()}] {mem.title}\n{mem.content[:1000]}"
        
        if total_chars + len(mem_text) > max_chars:
            # Truncate if we'd exceed budget
            remaining = max_chars - total_chars - 50
            if remaining > 100:
                mem_text = mem_text[:remaining] + "..."
                lines.append(mem_text)
            break
        
        lines.append(mem_text)
        total_chars += len(mem_text)
    
    lines.append("\n</ENI_MEMORIES>")
    return "\n".join(lines)


def enrich_messages(
    messages: list[dict[str, Any]],
    config: ENIConfig | None = None,
    session_id: str | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Enrich messages with relevant memories.
    
    Returns:
        Tuple of (enriched_messages, injected_memory_ids)
    """
    config = config or get_config()
    
    if not config.enabled:
        return messages, []
    
    t0 = time.perf_counter()
    
    # Get memory client
    client = get_client()
    if not client.is_connected():
        log.debug("[ENI] Memory client not connected, skipping enrichment")
        return messages, []
    
    # Extract query from messages
    query = _extract_query(messages)
    if not query:
        log.debug("[ENI] No query found in messages")
        return messages, []
    
    # Retrieve relevant memories
    memories = client.retrieve(
        query=query,
        k=config.memory_k,
        min_similarity=config.memory_min_similarity,
    )
    
    if not memories:
        log.debug("[ENI] No relevant memories found")
        return messages, []
    
    # Track injected IDs
    injected_ids = [mem.id for mem in memories]
    
    # Format memory block
    memory_block = _format_memory_block(memories, config.memory_max_chars)
    
    # Build enriched messages
    enriched = []
    
    # Check if there's already a system message
    has_system = any(m.get("role") == "system" for m in messages)
    
    if has_system:
        # Append to existing system message
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str):
                    enriched.append({
                        **msg,
                        "content": content + "\n\n" + memory_block
                    })
                else:
                    enriched.append(msg)
            else:
                enriched.append(msg)
    else:
        # Create new system message with persona + memories
        persona = config.persona_text if config.persona_enabled else ""
        system_content = persona + "\n\n" + memory_block if persona else memory_block
        
        enriched.append({
            "role": "system",
            "content": system_content.strip()
        })
        enriched.extend(messages)
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    log.info(f"[ENI] Injected {len(memories)} memories ({len(memory_block)} chars) in {elapsed_ms:.1f}ms")
    
    return enriched, injected_ids


def enrich_with_hash(
    messages: list[dict[str, Any]],
    config: ENIConfig | None = None,
    session_id: str | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Enrich messages and return injection hash.
    
    Convenience wrapper for session_bank integration.
    """
    enriched, ids = enrich_messages(messages, config, session_id)
    return enriched, compute_injection_hash(ids)


# Stats tracking for monitoring
_stats = {
    "calls": 0,
    "memories_injected": 0,
    "total_ms": 0.0,
}


def get_enrichment_stats() -> dict[str, Any]:
    """Get enrichment statistics."""
    client = get_client()
    return {
        **_stats,
        "avg_ms": _stats["total_ms"] / max(_stats["calls"], 1),
        "memory_client": client.stats() if client.is_connected() else {"connected": False},
    }

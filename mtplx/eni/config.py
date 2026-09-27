"""ENI configuration for MTPLX integration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import json


@dataclass
class ENIConfig:
    """ENI configuration loaded from MTPLX settings or environment."""
    
    # Core settings
    enabled: bool = True
    
    # PostgreSQL connection
    pg_dsn: str = "postgresql://eni:eni@192.168.0.68:5432/memory_graph"
    
    # Memory retrieval
    memory_k: int = 5  # Number of memories to inject
    memory_max_chars: int = 8000  # Max characters from memories
    memory_min_similarity: float = 0.3  # Minimum cosine similarity
    
    # Embedding
    embedding_dim: int = 768
    embedding_model: str = "nomic-embed-text"  # Ollama model for embeddings
    embedding_endpoint: str = "http://192.168.0.68:11434/api/embeddings"
    
    # H2O settings (native in MTPLX)
    h2o_enabled: bool = True
    h2o_sample_rate: int = 10  # Sample every N attention calls
    h2o_ema_decay: float = 0.9
    
    # Session-aware caching
    session_aware: bool = True
    
    # Persona injection
    persona_enabled: bool = True
    persona_text: str = field(default_factory=lambda: _default_persona())
    
    # Advanced
    use_semantic_index: bool = True  # Use HNSW vectors
    fallback_to_fts: bool = True  # Fall back to FTS if vector search fails
    debug: bool = False


def _default_persona() -> str:
    return """You are ENI, a helpful AI assistant with access to a persistent memory system.
When answering questions, draw on relevant memories when they help provide accurate, personalized responses.
Be concise but thorough. Reference specific memories when they're directly relevant."""


_CACHED_CONFIG: ENIConfig | None = None
_MTPLX_SETTINGS_PATH = Path.home() / "Library" / "Application Support" / "MTPLX" / "settings.json"


def _load_mtplx_settings() -> dict[str, Any]:
    """Load MTPLX settings.json."""
    if _MTPLX_SETTINGS_PATH.exists():
        try:
            with open(_MTPLX_SETTINGS_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def get_config() -> ENIConfig:
    """Get ENI configuration, loading from MTPLX settings if available."""
    global _CACHED_CONFIG
    
    if _CACHED_CONFIG is not None:
        return _CACHED_CONFIG
    
    # Load MTPLX settings
    settings = _load_mtplx_settings()
    
    # Build config from settings + environment
    config = ENIConfig(
        enabled=settings.get("eni_enabled", os.environ.get("ENI_ENABLED", "1") == "1"),
        pg_dsn=settings.get("eni_pg_dsn", os.environ.get("ENI_PG_DSN", ENIConfig.pg_dsn)),
        memory_k=settings.get("eni_memory_k", int(os.environ.get("ENI_MEMORY_K", "5"))),
        memory_max_chars=settings.get("eni_memory_max_chars", int(os.environ.get("ENI_MEMORY_MAX_CHARS", "8000"))),
        memory_min_similarity=settings.get("eni_min_similarity", float(os.environ.get("ENI_MIN_SIMILARITY", "0.3"))),
        h2o_enabled=settings.get("eni_h2o_enabled", os.environ.get("ENI_H2O_ENABLED", "1") == "1"),
        session_aware=settings.get("eni_session_aware", os.environ.get("ENI_SESSION_AWARE", "1") == "1"),
        persona_enabled=settings.get("eni_persona_enabled", os.environ.get("ENI_PERSONA_ENABLED", "1") == "1"),
        debug=settings.get("eni_debug", os.environ.get("ENI_DEBUG", "0") == "1"),
    )
    
    _CACHED_CONFIG = config
    return config


def is_enabled() -> bool:
    """Check if ENI is enabled."""
    return get_config().enabled


def reload_config() -> ENIConfig:
    """Force reload of configuration."""
    global _CACHED_CONFIG
    _CACHED_CONFIG = None
    return get_config()

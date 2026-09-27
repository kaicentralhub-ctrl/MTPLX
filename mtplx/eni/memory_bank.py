"""NEXUS L1 — Persistent Memory Bank.

Zero-latency memory recall for the top-K most important memories.
Pre-computes and permanently caches KV representations of high-value
memories so they're always "understood" without re-prefilling.

Physics:
- At startup, select top-K memories by PageRank × recency × access_count.
- Pre-prefill these memories once, store as permanent KV bank.
- Every request gets these for free (zero prefill cost).
- Delta memories (query-specific) get the standard enrichment path.
- Extends ``session_bank.py`` into a persistent ``memory_bank``.

Integration points:
- ``kv_cache.py`` — reuses serialization/deserialization
- ``session_bank.py`` — injects as permanent bank entries
- ``pxk.py`` — persistent prefix cache infrastructure
- ``memory_client.py`` — memory retrieval for selection

Expected impact: Zero-latency recall of top-100 memories. The most
important facts are always in context without re-computation.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger("mtplx.eni.memory_bank")


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class MemoryBankConfig:
    """Tunable parameters for the persistent memory bank."""
    enabled: bool = True
    # Selection
    max_bank_size: int = 100         # top-K memories to keep permanently loaded
    min_importance: float = 0.7      # only include memories above this importance
    min_access_count: int = 3        # only include memories accessed at least N times
    # Scoring weights for ranking
    pagerank_weight: float = 0.4
    importance_weight: float = 0.3
    recency_weight: float = 0.2
    access_weight: float = 0.1
    # Refresh
    refresh_interval_s: float = 3600.0  # rebuild bank every hour
    lazy_rebuild: bool = True           # only rebuild when requested
    # Performance
    max_total_tokens: int = 8192    # total token budget for all bank memories
    max_tokens_per_memory: int = 512 # per-memory token budget


_config = MemoryBankConfig()


def configure(cfg: MemoryBankConfig | None = None, **kwargs: Any) -> MemoryBankConfig:
    """Update global config."""
    global _config
    if cfg is not None:
        _config = cfg
    for k, v in kwargs.items():
        if hasattr(_config, k):
            setattr(_config, k, v)
    return _config


def get_config() -> MemoryBankConfig:
    return _config


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class BankEntry:
    """A memory entry in the persistent bank."""
    memory_id: str
    title: str
    content: str
    category: str
    importance: float
    pagerank: float = 0.0
    access_count: int = 0
    age_days: float = 0.0
    score: float = 0.0
    token_ids: list[int] = field(default_factory=list)
    kv_snapshot: Any = None  # serialized KV cache (bytes or MLX arrays)


@dataclass
class BankState:
    """State of the persistent memory bank."""
    entries: list[BankEntry] = field(default_factory=list)
    total_tokens: int = 0
    last_rebuild: float = 0.0
    hits: int = 0
    misses: int = 0


# ── Memory Selection ─────────────────────────────────────────────────────────

def score_memory(
    importance: float,
    pagerank: float,
    access_count: int,
    age_days: float,
    config: MemoryBankConfig | None = None,
) -> float:
    """Score a memory for bank inclusion.

    Combines PageRank, importance, recency, and access frequency
    into a single selection score.

    Parameters
    ----------
    importance : float
        Memory importance (0-1).
    pagerank : float
        Graph centrality score.
    access_count : int
        How many times this memory has been retrieved.
    age_days : float
        Age in days (lower = more recent).
    config : MemoryBankConfig, optional

    Returns
    -------
    Selection score (higher = more likely to be in bank).
    """
    cfg = config or _config

    # Recency score: exponential decay with 30-day half-life
    recency = max(0.1, 0.5 ** (age_days / 30.0))

    # Access score: log scale to prevent runaway
    access = min(np.log1p(access_count) / 10.0, 1.0)

    # Pagerank: already 0-1 normalized
    pr = min(max(pagerank, 0.0), 1.0)

    # Importance: already 0-1
    imp = min(max(importance, 0.0), 1.0)

    score = (
        cfg.pagerank_weight * pr
        + cfg.importance_weight * imp
        + cfg.recency_weight * recency
        + cfg.access_weight * access
    )

    return score


def select_bank_memories(
    memories: list[Any],
    config: MemoryBankConfig | None = None,
) -> list[BankEntry]:
    """Select top-K memories for the persistent bank.

    Parameters
    ----------
    memories : list[Memory]
        All available memories (from memory_client).
    config : MemoryBankConfig, optional

    Returns
    -------
    List of BankEntry objects, ranked by selection score.
    """
    cfg = config or _config

    candidates = []
    for mem in memories:
        importance = float(getattr(mem, "importance", 0.0))
        if importance < cfg.min_importance:
            continue

        access = int(getattr(mem, "access_count", 0))
        if access < cfg.min_access_count:
            continue

        entry = BankEntry(
            memory_id=getattr(mem, "id", str(id(mem))),
            title=getattr(mem, "title", ""),
            content=getattr(mem, "content", "")[:cfg.max_tokens_per_memory * 4],
            category=getattr(mem, "category", "fact"),
            importance=importance,
            pagerank=float(getattr(mem, "pagerank", 0.0)),
            access_count=access,
            age_days=float(getattr(mem, "age_days", 0.0)),
        )
        entry.score = score_memory(
            importance, entry.pagerank, access, entry.age_days, cfg
        )
        candidates.append(entry)

    # Sort by score descending
    candidates.sort(key=lambda e: e.score, reverse=True)

    # Take top-K within token budget
    selected = []
    total_tokens = 0

    for entry in candidates:
        est_tokens = len(entry.content) // 4
        if total_tokens + est_tokens > cfg.max_total_tokens:
            break
        selected.append(entry)
        total_tokens += est_tokens

        if len(selected) >= cfg.max_bank_size:
            break

    log.debug(
        f"[NEXUS-MB] Selected {len(selected)} memories for bank "
        f"({total_tokens} tokens) from {len(memories)} candidates"
    )

    return selected


# ── Persistent Bank ──────────────────────────────────────────────────────────

class PersistentMemoryBank:
    """Manages the persistent memory bank lifecycle.

    Usage::

        bank = PersistentMemoryBank(config)
        bank.build(memories, tokenizer)  # one-time setup
        # On each request:
        bank.inject_into_session(session_bank)  # zero-cost injection
    """

    def __init__(self, config: MemoryBankConfig | None = None) -> None:
        self.config = config or _config
        self._state = BankState()
        self._built = False

    def build(
        self,
        memories: list[Any],
        tokenizer: Any = None,
    ) -> int:
        """Build the persistent memory bank.

        Selects top-K memories, tokenizes content, and prepares
        KV cache snapshots (if tokenizer and model are available).

        Returns
        -------
        Number of memories in the bank.
        """
        t0 = time.monotonic()

        entries = select_bank_memories(memories, self.config)

        # Tokenize each entry
        total_tokens = 0
        for entry in entries:
            if tokenizer is not None:
                try:
                    text = f"{entry.title}\n{entry.content}"
                    entry.token_ids = tokenizer.encode(text)[:self.config.max_tokens_per_memory]
                    total_tokens += len(entry.token_ids)
                except Exception as e:
                    log.debug(f"[NEXUS-MB] Tokenization failed: {e}")
                    entry.token_ids = []
            else:
                # Estimate tokens from content length
                est = len(entry.content) // 4
                total_tokens += est

        self._state.entries = entries
        self._state.total_tokens = total_tokens
        self._state.last_rebuild = time.time()
        self._built = True

        elapsed = (time.monotonic() - t0) * 1000.0
        log.info(
            f"[NEXUS-MB] Bank built: {len(entries)} memories, "
            f"{total_tokens} tokens, {elapsed:.0f}ms"
        )

        return len(entries)

    def needs_rebuild(self) -> bool:
        """Check if the bank should be rebuilt."""
        if not self._built:
            return True
        if not self.config.lazy_rebuild:
            return False
        return (time.time() - self._state.last_rebuild) > self.config.refresh_interval_s

    def inject_into_session(self, session_bank: Any) -> bool:
        """Inject bank entries into the session bank as permanent entries.

        Parameters
        ----------
        session_bank : SessionBank
            MTPLX session bank to inject into.

        Returns
        -------
        True if injection succeeded.
        """
        if not self._built or not self._state.entries:
            return False

        try:
            from .kv_cache import inject_into_bank, deserialize_snapshot

            injected = 0
            for entry in self._state.entries:
                if entry.kv_snapshot is not None and entry.token_ids:
                    snapshot = deserialize_snapshot(entry.kv_snapshot)
                    success = inject_into_bank(
                        session_bank, snapshot,
                        logits=None,
                        token_ids=entry.token_ids,
                        runtime=None,
                    )
                    if success:
                        injected += 1

            log.debug(f"[NEXUS-MB] Injected {injected}/{len(self._state.entries)} bank entries")
            return injected > 0

        except Exception as e:
            log.warning(f"[NEXUS-MB] Injection failed: {e}")
            return False

    def get_context_block(self) -> str:
        """Get bank memories as a formatted context block.

        Alternative to KV injection — returns memories as text
        for direct prompt injection.
        """
        if not self._state.entries:
            return ""

        lines = ["<MEMORY_BANK>"]
        for entry in self._state.entries:
            lines.append(f"[{entry.category.upper()}] {entry.title}")
            lines.append(entry.content[:200])
        lines.append("</MEMORY_BANK>")
        return "\n".join(lines)

    @property
    def size(self) -> int:
        return len(self._state.entries)

    @property
    def total_tokens(self) -> int:
        return self._state.total_tokens

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "size": len(self._state.entries),
            "total_tokens": self._state.total_tokens,
            "hits": self._state.hits,
            "misses": self._state.misses,
            "last_rebuild": self._state.last_rebuild,
            "built": self._built,
        }


# ── Global Bank Instance ─────────────────────────────────────────────────────

_global_bank: PersistentMemoryBank | None = None


def get_memory_bank() -> PersistentMemoryBank | None:
    """Get the global memory bank instance."""
    return _global_bank


def init_memory_bank(
    memories: list[Any],
    tokenizer: Any = None,
    config: MemoryBankConfig | None = None,
) -> PersistentMemoryBank:
    """Initialize the global memory bank."""
    global _global_bank
    _global_bank = PersistentMemoryBank(config)
    _global_bank.build(memories, tokenizer)
    return _global_bank


def shutdown_memory_bank() -> None:
    """Shutdown and clear the global memory bank."""
    global _global_bank
    _global_bank = None


# ── Stats ────────────────────────────────────────────────────────────────────

_stats = {
    "banks_created": 0,
    "total_memories_banked": 0,
    "total_injections": 0,
}


def get_memory_bank_stats() -> dict[str, Any]:
    result = dict(_stats)
    if _global_bank is not None:
        result["current_bank"] = _global_bank.stats
    return result


def reset_memory_bank_stats() -> None:
    global _stats
    _stats = {
        "banks_created": 0,
        "total_memories_banked": 0,
        "total_injections": 0,
    }

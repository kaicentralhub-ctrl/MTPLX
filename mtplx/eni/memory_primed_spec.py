"""NEXUS L3 — Memory-Primed Speculative Decoding.

Boosts MTP draft token logits using memory graph content so that
factual, memory-consistent sequences are drafted with near-certainty.

Physics:
- Memory content is injected into the prompt as text → tokenized.
- We extract n-gram patterns from memory token sequences.
- During draft sampling, tokens that continue a memory-consistent
  sequence get their logits boosted.
- The verify step (target model) catches any hallucinations.
- When memories are accurate, draft acceptance skyrockets.

Integration point: ``_sample_draft_from_logits()`` in ``generation.py``
(line ~6861) and ``generate_mtp1()`` / ``generate_mtpk()``.

Expected impact: 40-80% improvement in MTP acceptance rate for
factual content (per the 8-tier roadmap Tier 3).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger("mtplx.eni.mem_spec")


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class MemorySpecConfig:
    """Tunable parameters for memory-primed speculative decoding."""
    enabled: bool = True
    # N-gram extraction
    min_ngram: int = 3              # minimum n-gram length to extract
    max_ngram: int = 12             # maximum n-gram length
    max_patterns: int = 500         # cap on total patterns (memory + speed)
    # Logit boosting
    boost_strength: float = 2.0     # additive logit boost for memory-consistent tokens
    decay_per_position: float = 0.15 # boost decays as draft position increases
    max_boost: float = 5.0          # clip boost to prevent over-steering
    # Matching
    prefix_match_min: int = 2       # minimum prefix match to trigger boost
    case_sensitive: bool = False    # token-level matching is always case-sensitive
    # Performance
    max_pattern_tokens: int = 50    # truncate patterns longer than this


_config = MemorySpecConfig()


def configure(cfg: MemorySpecConfig | None = None, **kwargs: Any) -> MemorySpecConfig:
    """Update global config."""
    global _config
    if cfg is not None:
        _config = cfg
    for k, v in kwargs.items():
        if hasattr(_config, k):
            setattr(_config, k, v)
    return _config


def get_config() -> MemorySpecConfig:
    return _config


# ── Pattern Trie ─────────────────────────────────────────────────────────────

class _TrieNode:
    """A node in the token-pattern trie.

    Each node maps a token_id to a child node. Terminal nodes carry
    a boost weight derived from the memory's importance.
    """
    __slots__ = ("children", "boost", "count")

    def __init__(self) -> None:
        self.children: dict[int, _TrieNode] = {}
        self.boost: float = 0.0
        self.count: int = 0


class MemoryPatternTrie:
    """Prefix trie of expected token sequences from memory content.

    Built once per request from the injected memory tokens. Used during
    draft sampling to boost tokens that continue known memory sequences.
    """

    def __init__(self) -> None:
        self.root = _TrieNode()
        self._pattern_count = 0
        self._total_tokens = 0

    def insert(self, token_seq: list[int], boost: float) -> None:
        """Insert a token sequence with an associated boost weight."""
        if not token_seq:
            return

        node = self.root
        for tok in token_seq:
            if tok not in node.children:
                node.children[tok] = _TrieNode()
            node = node.children[tok]
            node.count += 1
            # Accumulate boost (max wins — strongest memory dominates)
            node.boost = max(node.boost, boost)

        self._pattern_count += 1
        self._total_tokens += len(token_seq)

    def lookup_continuation(self, prefix: list[int]) -> dict[int, float]:
        """Given a token prefix, return {next_token_id: boost_weight}.

        Walks the trie following the prefix path, then returns all
        children of the terminal node as candidate continuations.
        """
        node = self.root
        for tok in prefix:
            if tok not in node.children:
                return {}
            node = node.children[tok]

        # Return all children as candidate next tokens
        return {tok: child.boost for tok, child in node.children.items()}

    @property
    def pattern_count(self) -> int:
        return self._pattern_count

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    def __len__(self) -> int:
        return self._pattern_count


# ── Pattern Extraction ───────────────────────────────────────────────────────

def extract_patterns_from_tokens(
    memory_token_seqs: list[tuple[list[int], float]],
    config: MemorySpecConfig | None = None,
) -> MemoryPatternTrie:
    """Extract n-gram patterns from memory token sequences.

    Parameters
    ----------
    memory_token_seqs : list[tuple[list[int], float]]
        List of (token_ids, importance_weight) for each memory.
    config : MemorySpecConfig, optional
        Override global config.

    Returns
    -------
    MemoryPatternTrie with all extracted patterns.
    """
    cfg = config or _config
    trie = MemoryPatternTrie()

    for token_seq, importance in memory_token_seqs:
        if not token_seq:
            continue

        # Truncate very long sequences
        seq = token_seq[:cfg.max_pattern_tokens]

        # Boost proportional to importance (scaled to [0.5, max_boost])
        boost = min(
            cfg.boost_strength * max(importance, 0.1),
            cfg.max_boost,
        )

        # Extract all n-grams in [min_ngram, max_ngram] range
        for n in range(cfg.min_ngram, min(cfg.max_ngram + 1, len(seq) + 1)):
            for i in range(len(seq) - n + 1):
                if trie.pattern_count >= cfg.max_patterns:
                    break
                trie.insert(seq[i:i + n], boost)

    log.debug(
        f"[NEXUS-MPS] Extracted {trie.pattern_count} patterns "
        f"({trie.total_tokens} tokens) from {len(memory_token_seqs)} memories"
    )

    return trie


# ── Logit Boosting ───────────────────────────────────────────────────────────

def boost_draft_logits(
    logits: np.ndarray,
    recent_tokens: list[int],
    trie: MemoryPatternTrie,
    draft_position: int = 0,
    config: MemorySpecConfig | None = None,
) -> np.ndarray:
    """Boost draft logits based on memory-consistent token continuations.

    Given the current logits from the draft model and the recent token
    context, look up the trie for matching continuations and boost
    those tokens' logits.

    Parameters
    ----------
    logits : np.ndarray
        Raw logits from the draft model, shape [vocab_size].
    recent_tokens : list[int]
        Recent token context (last N tokens generated so far).
    trie : MemoryPatternTrie
        The pattern trie built from memory content.
    draft_position : int
        Current position in the draft (0 = first draft token).
        Used for position-based boost decay.
    config : MemorySpecConfig, optional
        Override global config.

    Returns
    -------
    Modified logits (new array, input not mutated).
    """
    cfg = config or _config

    if not cfg.enabled or not recent_tokens or len(trie) == 0:
        return logits

    # Decay boost as draft position increases (later tokens are less certain)
    position_decay = max(1.0 - draft_position * cfg.decay_per_position, 0.1)

    # Look up continuation for the trailing context
    # Try progressively shorter prefixes (longest match wins)
    boosted = logits.copy()
    matched = False

    for prefix_len in range(
        min(cfg.max_ngram, len(recent_tokens)),
        cfg.prefix_match_min - 1,
        -1,
    ):
        prefix = recent_tokens[-prefix_len:]
        continuations = trie.lookup_continuation(prefix)

        if continuations:
            for tok_id, boost in continuations.items():
                if 0 <= tok_id < len(boosted):
                    adjusted_boost = boost * position_decay
                    boosted[tok_id] += min(adjusted_boost, cfg.max_boost)
                    matched = True
            break  # longest match wins

    if matched:
        log.debug(
            f"[NEXUS-MPS] Boosted {len(continuations)} candidate tokens "
            f"at draft_pos={draft_position}, decay={position_decay:.2f}"
        )

    return boosted


# ── Integration: Draft Sampler Hook ──────────────────────────────────────────

class MemoryPrimedDraftSampler:
    """Wraps the draft sampling function to inject memory-based boosts.

    Usage in generation.py::

        primed_sampler = MemoryPrimedDraftSampler(memory_tokens, importance)
        # ... in draft loop:
        boosted_logits = primed_sampler.boost(logits, generated_so_far, step)

    Or as a context manager for the full generation::

        with MemoryPrimedDraftSampler.activate(memory_tokens) as sampler:
            # generation loop runs with memory-primed drafts
            ...
    """

    def __init__(
        self,
        memory_token_seqs: list[tuple[list[int], float]] | None = None,
        config: MemorySpecConfig | None = None,
    ) -> None:
        self.config = config or _config
        self.trie = extract_patterns_from_tokens(
            memory_token_seqs or [], self.config
        )
        self._boost_count = 0
        self._total_boost_tokens = 0

    def boost(
        self,
        logits: np.ndarray,
        recent_tokens: list[int],
        draft_position: int = 0,
    ) -> np.ndarray:
        """Boost logits for one draft step."""
        result = boost_draft_logits(
            logits, recent_tokens, self.trie,
            draft_position=draft_position,
            config=self.config,
        )
        if result is not logits:
            self._boost_count += 1
        return result

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "patterns": len(self.trie),
            "boost_calls": self._boost_count,
            "avg_boost_tokens": (
                self._total_boost_tokens / self._boost_count
                if self._boost_count > 0 else 0
            ),
        }

    @classmethod
    def activate(
        cls,
        memory_token_seqs: list[tuple[list[int], float]],
        config: MemorySpecConfig | None = None,
    ) -> "_PrimedContext":
        """Context manager that activates memory-primed sampling globally."""
        return _PrimedContext(memory_token_seqs, config)


class _PrimedContext:
    """Context manager for global memory-primed draft sampling."""

    def __init__(
        self,
        memory_token_seqs: list[tuple[list[int], float]],
        config: MemorySpecConfig | None = None,
    ) -> None:
        self._seqs = memory_token_seqs
        self._config = config
        self._sampler: MemoryPrimedDraftSampler | None = None

    def __enter__(self) -> MemoryPrimedDraftSampler:
        self._sampler = MemoryPrimedDraftSampler(self._seqs, self._config)
        _set_active_sampler(self._sampler)
        return self._sampler

    def __exit__(self, *args: Any) -> None:
        _set_active_sampler(None)


# ── Global Active Sampler (for generation.py hooks) ──────────────────────────

_active_sampler: MemoryPrimedDraftSampler | None = None


def _set_active_sampler(sampler: MemoryPrimedDraftSampler | None) -> None:
    global _active_sampler
    _active_sampler = sampler


def get_active_sampler() -> MemoryPrimedDraftSampler | None:
    return _active_sampler


def boost_logits_if_active(
    logits: np.ndarray,
    recent_tokens: list[int],
    draft_position: int = 0,
) -> np.ndarray:
    """Drop-in hook for generation.py.

    Call this in ``_sample_draft_from_logits()`` before sampling::

        logits = boost_logits_if_active(logits, generated_tokens, step)

    If no sampler is active (no memories injected), returns logits unchanged.
    """
    if _active_sampler is None:
        return logits
    return _active_sampler.boost(logits, recent_tokens, draft_position)


# ── Memory Token Extraction from Enrichment ──────────────────────────────────

def extract_memory_token_seqs(
    memories: list[Any],
    tokenizer: Any,
) -> list[tuple[list[int], float]]:
    """Tokenize memory content for pattern extraction.

    Parameters
    ----------
    memories : list[Memory]
        Retrieved memories from enrichment.
    tokenizer : MTPLX tokenizer
        The tokenizer used for generation.

    Returns
    -------
    List of (token_ids, importance) tuples ready for trie insertion.
    """
    result = []

    for mem in memories:
        content = getattr(mem, "content", "")
        title = getattr(mem, "title", "")
        importance = float(getattr(mem, "importance", 0.5))

        # Tokenize title + content
        text = f"{title}\n{content}" if title else content
        if not text.strip():
            continue

        try:
            token_ids = tokenizer.encode(text)
            if isinstance(token_ids, list) and len(token_ids) >= 3:
                result.append((token_ids, importance))
        except Exception as e:
            log.debug(f"[NEXUS-MPS] Tokenization failed for memory: {e}")
            continue

    return result


# ── Stats ────────────────────────────────────────────────────────────────────

_stats = {
    "samplers_created": 0,
    "total_patterns": 0,
    "total_boost_calls": 0,
}


def get_memory_spec_stats() -> dict[str, Any]:
    return dict(_stats)


def reset_memory_spec_stats() -> None:
    global _stats
    _stats = {
        "samplers_created": 0,
        "total_patterns": 0,
        "total_boost_calls": 0,
    }

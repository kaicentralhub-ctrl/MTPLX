"""NEXUS L6 — Unified Inference API.

Clean abstraction that ties all NEXUS layers together into a single
inference surface. Provides a unified interface where memory retrieval,
graph attention bias, memory-primed speculation, causal verification,
difficulty routing, and persistent memory are all first-class parameters.

This is NOT an HTTP server — it's the internal API that existing servers
(``eni_serve.py``, ``eni_mtplx_serve.py``, ``openai.py``) call into.

Architecture::

    NEXUS API
      ├── Difficulty Router    (L5) → model selection
      ├── Memory Bank          (L1) → zero-latency top-K recall
      ├── Memory Retrieval     (L2) → query-specific HNSW search
      ├── Graph-Attention Bias (L2) → attention shaping from graph edges
      ├── Memory-Primed Spec   (L3) → draft token boosting
      ├── Causal Verification  (L4) → real-time contradiction checking
      └── Adaptive Sampling    (L3) → feedback-driven temperature
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .config import get_config as get_eni_config
from .memory_client import get_client, Memory
from .enrichment import enrich_messages, compute_injection_hash
from .graph_attention_bias import (
    build_from_retrieval, BiasMatrix, GraphBiasConfig,
    configure as configure_graph_bias,
)
from .memory_primed_spec import (
    MemoryPrimedDraftSampler, MemorySpecConfig,
    configure as configure_memory_spec,
)
from .causal_verification import (
    CausalVerifier, create_token_callback,
    CausalVerifyConfig, configure as configure_causal_verify,
)
from .memory_bank import (
    PersistentMemoryBank, get_memory_bank, init_memory_bank,
    MemoryBankConfig, configure as configure_memory_bank,
)
from .difficulty_router import (
    classify_query, route_query, DifficultyTier, ModelRoute,
    RouterConfig, configure as configure_router,
)
from .adaptive_sampling import (
    get_recommended_draft_temp, record_accept_rate_async,
)

log = logging.getLogger("mtplx.eni.nexus")


# ── Request/Response Types ───────────────────────────────────────────────────

@dataclass
class NexusRequest:
    """A unified inference request through NEXUS."""
    messages: list[dict[str, Any]]
    session_id: str | None = None
    # Routing
    force_tier: DifficultyTier | None = None
    # Memory
    memory_k: int = 5
    use_memory_bank: bool = True
    use_graph_bias: bool = True
    # Speculation
    use_memory_primed_spec: bool = True
    # Verification
    use_causal_verification: bool = True
    # Sampling
    use_adaptive_sampling: bool = True
    temperature: float | None = None
    max_tokens: int = 2048


@dataclass
class NexusResult:
    """Result from a NEXUS inference request."""
    # Routing
    route: ModelRoute
    difficulty_score: float
    # Memory
    memories_used: list[Memory] = field(default_factory=list)
    injection_hash: str = "empty"
    memory_bank_hit: bool = False
    # Graph bias
    bias_matrix: BiasMatrix | None = None
    # Speculation
    spec_sampler: MemoryPrimedDraftSampler | None = None
    # Verification
    verifier: CausalVerifier | None = None
    token_callback: Callable | None = None
    # Sampling
    recommended_temp: float | None = None
    # Generation (filled by caller)
    generated_text: str = ""
    tokens_generated: int = 0
    generation_ms: float = 0.0
    # Stats
    preparation_ms: float = 0.0


@dataclass
class NexusConfig:
    """Global NEXUS configuration."""
    enabled: bool = True
    # Layer toggles
    enable_routing: bool = True
    enable_memory: bool = True
    enable_memory_bank: bool = True
    enable_graph_bias: bool = True
    enable_primed_spec: bool = True
    enable_causal_verify: bool = True
    enable_adaptive_sampling: bool = True
    # Sub-configs
    router_config: RouterConfig = field(default_factory=RouterConfig)
    memory_bank_config: MemoryBankConfig = field(default_factory=MemoryBankConfig)
    graph_bias_config: GraphBiasConfig = field(default_factory=GraphBiasConfig)
    spec_config: MemorySpecConfig = field(default_factory=MemorySpecConfig)
    causal_config: CausalVerifyConfig = field(default_factory=CausalVerifyConfig)


_nexus_config = NexusConfig()


def configure(**kwargs: Any) -> NexusConfig:
    """Update global NEXUS config."""
    global _nexus_config
    for k, v in kwargs.items():
        if hasattr(_nexus_config, k):
            setattr(_nexus_config, k, v)
    return _nexus_config


def get_nexus_config() -> NexusConfig:
    return _nexus_config


# ── Preparation Pipeline ─────────────────────────────────────────────────────

def prepare(request: NexusRequest) -> NexusResult:
    """Prepare a request through all NEXUS layers.

    This is the core pipeline that runs BEFORE generation. It:
    1. Routes the query to the optimal model (L5)
    2. Retrieves relevant memories (L2)
    3. Builds graph-attention bias matrix (L2)
    4. Initializes memory-primed speculation (L3)
    5. Sets up causal verification (L4)
    6. Determines adaptive sampling parameters (L3)

    Returns a NexusResult with all prepared state ready for generation.
    """
    t0 = time.monotonic()
    cfg = _nexus_config

    # ── Step 1: Difficulty Routing (L5) ─────────────────────────────────
    query_text = _extract_query(request.messages)
    route = route_query(query_text) if cfg.enable_routing else None

    if route is None:
        from .difficulty_router import ModelRoute
        route = ModelRoute(
            tier=DifficultyTier.MEDIUM,
            model="eni-27b-mtplx",
            backend="local",
            estimated_cost=0.001,
            estimated_latency_ms=500,
        )

    result = NexusResult(
        route=route,
        difficulty_score=classify_query(query_text).score,
    )

    # ── Step 2: Memory Bank (L1) ────────────────────────────────────────
    if cfg.enable_memory_bank and request.use_memory_bank:
        bank = get_memory_bank()
        if bank is not None and bank.size > 0:
            result.memory_bank_hit = True
            # Bank memories are always in context via KV injection
            # (handled at the session_bank level)

    # ── Step 3: Memory Retrieval (L2) ───────────────────────────────────
    if cfg.enable_memory and request.memory_k > 0:
        try:
            messages, injected_ids = enrich_with_hash(
                request.messages,
                k=request.memory_k,
                session_id=request.session_id,
            )
            result.injection_hash = compute_injection_hash(injected_ids)

            # Get memory objects for downstream use
            client = get_client()
            if client and client.is_connected():
                from .memory_client import Memory
                # Fetch memory details for the injected IDs
                result.memories_used = _fetch_memories(injected_ids, client)

        except Exception as e:
            log.warning(f"[NEXUS] Memory retrieval failed: {e}")

    # ── Step 4: Graph-Attention Bias (L2) ───────────────────────────────
    if cfg.enable_graph_bias and request.use_graph_bias and result.memories_used:
        try:
            memory_ids = [m.id for m in result.memories_used]
            result.bias_matrix = build_from_retrieval(
                memories=result.memories_used,
                edges=None,  # auto-fetch from graph
                prompt_tokens=[],  # estimated at generation time
                memory_block_start=0,
            )
        except Exception as e:
            log.warning(f"[NEXUS] Graph bias construction failed: {e}")

    # ── Step 5: Memory-Primed Speculation (L3) ──────────────────────────
    if cfg.enable_primed_spec and request.use_memory_primed_spec and result.memories_used:
        try:
            from .memory_primed_spec import extract_memory_token_seqs
            # We need the tokenizer — defer to generation time
            # For now, create the sampler with raw content patterns
            memory_seqs = [
                (m.content.encode("utf-8")[:200], m.importance)
                for m in result.memories_used
                if m.content
            ]
            # Convert bytes to pseudo-token IDs for trie building
            pseudo_seqs = [
                ([int(b) for b in seq], imp) for seq, imp in memory_seqs
            ]
            result.spec_sampler = MemoryPrimedDraftSampler(pseudo_seqs)
        except Exception as e:
            log.warning(f"[NEXUS] Memory-primed spec init failed: {e}")

    # ── Step 6: Causal Verification (L4) ────────────────────────────────
    if cfg.enable_causal_verify and request.use_causal_verification:
        try:
            entity_names = {m.id: m.title for m in result.memories_used}
            result.verifier = CausalVerifier(entity_names=entity_names)
            result.token_callback = create_token_callback(result.verifier)
        except Exception as e:
            log.warning(f"[NEXUS] Causal verification init failed: {e}")

    # ── Step 7: Adaptive Sampling (L3) ──────────────────────────────────
    if cfg.enable_adaptive_sampling and request.use_adaptive_sampling:
        try:
            result.recommended_temp = get_recommended_draft_temp(
                request.temperature
            )
        except Exception as e:
            log.debug(f"[NEXUS] Adaptive sampling lookup failed: {e}")

    result.preparation_ms = (time.monotonic() - t0) * 1000.0

    log.debug(
        f"[NEXUS] Prepared request: tier={route.tier.value}, "
        f"memories={len(result.memories_used)}, "
        f"bank_hit={result.memory_bank_hit}, "
        f"bias={'yes' if result.bias_matrix else 'no'}, "
        f"spec={'yes' if result.spec_sampler else 'no'}, "
        f"verify={'yes' if result.verifier else 'no'}, "
        f"{result.preparation_ms:.1f}ms"
    )

    return result


def finalize(result: NexusResult, generated_text: str, tokens_generated: int) -> None:
    """Finalize a request after generation. Records feedback signals.

    Call this after generation completes to feed the adaptive
    sampling feedback loop.
    """
    result.generated_text = generated_text
    result.tokens_generated = tokens_generated

    # Record accept rate for adaptive sampling
    if result.spec_sampler is not None:
        try:
            record_accept_rate_async(
                injection_hash=result.injection_hash,
                accept_rate=result.spec_sampler._boost_count / max(tokens_generated, 1),
            )
        except Exception:
            pass

    # Log verification results
    if result.verifier is not None:
        stats = result.verifier.stats
        if stats["contradictions"] > 0:
            log.info(
                f"[NEXUS] Verification: {stats['contradictions']} contradictions "
                f"detected in {tokens_generated} tokens"
            )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _extract_query(messages: list[dict[str, Any]]) -> str:
    """Extract query text from messages."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            return content[:2000]
    return ""


def _fetch_memories(memory_ids: list[str], client: Any) -> list[Memory]:
    """Fetch memory details by ID."""
    if not memory_ids:
        return []
    try:
        with client._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, title, content, category, importance
                    FROM memories
                    WHERE id = ANY(%s)
                    """,
                    (memory_ids,),
                )
                return [
                    Memory(
                        id=row[0], title=row[1], content=row[2],
                        category=row[3], importance=float(row[4]),
                    )
                    for row in cur.fetchall()
                ]
    except Exception as e:
        log.debug(f"[NEXUS] Memory fetch failed: {e}")
        return []


# ── Stats ────────────────────────────────────────────────────────────────────

_stats = {
    "requests_prepared": 0,
    "avg_preparation_ms": 0.0,
    "tier_distribution": {"simple": 0, "medium": 0, "hard": 0},
    "memory_bank_hits": 0,
    "graph_bias_built": 0,
    "spec_samplers_created": 0,
    "verifiers_created": 0,
}


def get_nexus_stats() -> dict[str, Any]:
    return dict(_stats)


def reset_nexus_stats() -> None:
    global _stats
    _stats = {
        "requests_prepared": 0,
        "avg_preparation_ms": 0.0,
        "tier_distribution": {"simple": 0, "medium": 0, "hard": 0},
        "memory_bank_hits": 0,
        "graph_bias_built": 0,
        "spec_samplers_created": 0,
        "verifiers_created": 0,
    }

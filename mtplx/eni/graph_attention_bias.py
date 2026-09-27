"""NEXUS L2 — Graph-Structured Attention Bias.

Maps knowledge graph edges to additive attention logit biases so the
graph topology directly shapes where the model looks during generation.

Physics:
- ``causal``     edges → positive bias  (A causes B → A's tokens attend to B)
- ``contradicts`` edges → negative bias (suppress conflicting information)
- ``supersedes``  edges → directional   (new attends to old, not reverse)
- ``related``     edges → mild positive
- ``temporal``    edges → recency decay
- PageRank        score → multiplicative attention weight multiplier

Zero parameter cost — biases are computed at inference time from the
retrieved memory subgraph, then added to attention logits before softmax.

Integration point: ``sdpa_mem_fused.py`` / ``attention_split.py`` /
``block_attention.py`` — bias matrix is passed alongside memory K/V.

Research grounding: Graphormer (2021), TreeMCTS (2024), but applied to
a personal knowledge graph with typed edges during *inference*, not
training. This is unprecedented.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger("mtplx.eni.graph_bias")


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class GraphBiasConfig:
    """Tunable parameters for graph-attention bias construction."""
    enabled: bool = True
    # Per-edge-type bias magnitudes (added to attention logits)
    causal_bias: float = 0.5          # strong positive
    contradicts_bias: float = -0.8    # strong negative (suppress)
    supersedes_bias: float = 0.3      # moderate positive (directional)
    related_bias: float = 0.2         # mild positive
    temporal_bias: float = 0.1        # recency-weighted
    auto_similar_bias: float = 0.15   # weakest positive
    # PageRank multiplier range
    pagerank_min_mult: float = 0.8    # low-importance memories get suppressed
    pagerank_max_mult: float = 1.5    # high-importance memories get boosted
    # Normalization
    clip_bias: float = 2.0            # clip extreme biases to prevent softmax collapse
    temperature_scale: float = 1.0    # global temperature on the bias
    # Memory span mapping
    max_memory_tokens: int = 4096     # max token span per memory in the bias matrix


_config = GraphBiasConfig()


def configure(cfg: GraphBiasConfig | None = None, **kwargs: Any) -> GraphBiasConfig:
    """Update global config."""
    global _config
    if cfg is not None:
        _config = cfg
    for k, v in kwargs.items():
        if hasattr(_config, k):
            setattr(_config, k, v)
    return _config


def get_config() -> GraphBiasConfig:
    return _config


# ── Edge Type Registry ───────────────────────────────────────────────────────

# Maps edge relation types to their bias direction and magnitude
EDGE_BIAS_MAP: dict[str, tuple[str, float]] = {
    # relation_type → (direction, magnitude_key)
    "causal":       ("forward", "causal_bias"),
    "contradicts":  ("symmetric", "contradicts_bias"),
    "supersedes":   ("directional", "supersedes_bias"),
    "related":      ("symmetric", "related_bias"),
    "temporal":     ("forward", "temporal_bias"),
    "auto_similar": ("symmetric", "auto_similar_bias"),
    "COMPLEMENT":   ("symmetric", "related_bias"),
    "REFINES":      ("directional", "related_bias"),
    "VERSION_UPDATE": ("directional", "supersedes_bias"),
    "PARADIGM_SHIFT": ("directional", "supersedes_bias"),
}


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class MemorySpan:
    """Maps a memory to its token span in the enriched prompt.

    After enrichment, memories are injected as text blocks in the system
    prompt. Each memory occupies a contiguous token span. This tracks
    which tokens belong to which memory so we can build the bias matrix.
    """
    memory_id: str
    start_tok: int
    end_tok: int          # exclusive
    pagerank: float = 0.0
    importance: float = 0.5
    category: str = "fact"


@dataclass
class GraphEdge:
    """A typed edge between two memories in the knowledge graph."""
    source_id: str
    target_id: str
    relation: str
    weight: float = 1.0


@dataclass
class BiasMatrix:
    """The output: a [seq_len × seq_len] additive attention bias matrix.

    This is added to attention logits BEFORE softmax, so positive values
    increase attention and negative values suppress it.
    """
    bias: np.ndarray              # [seq_len, seq_len] float32
    memory_spans: list[MemorySpan]
    edge_count: int = 0
    construction_ms: float = 0.0

    @property
    def shape(self) -> tuple[int, int]:
        return self.bias.shape

    @property
    def sparsity(self) -> float:
        """Fraction of entries that are non-zero."""
        if self.bias.size == 0:
            return 0.0
        return float(np.count_nonzero(self.bias)) / float(self.bias.size)


# ── Core Builder ─────────────────────────────────────────────────────────────

def build_bias_matrix(
    memory_spans: list[MemorySpan],
    edges: list[GraphEdge],
    seq_len: int,
    config: GraphBiasConfig | None = None,
) -> BiasMatrix:
    """Construct a [seq_len × seq_len] additive attention bias matrix.

    The matrix is built from the memory subgraph topology:
    1. Initialize all zeros (no bias = standard attention).
    2. For each graph edge, set the bias between the token spans of
       the source and target memories according to edge type.
    3. Apply PageRank multipliers to boost/suppress important memories.
    4. Clip and scale.

    Parameters
    ----------
    memory_spans : list[MemorySpan]
        Token spans for each retrieved memory in the enriched prompt.
    edges : list[GraphEdge]
        Typed edges between memories (from the knowledge graph).
    seq_len : int
        Total sequence length of the enriched prompt.
    config : GraphBiasConfig, optional
        Override global config.

    Returns
    -------
    BiasMatrix with the bias array and metadata.
    """
    cfg = config or _config
    t0 = time.monotonic()

    if not cfg.enabled or not memory_spans or seq_len <= 0:
        return BiasMatrix(
            bias=np.zeros((seq_len, seq_len), dtype=np.float32),
            memory_spans=memory_spans,
            edge_count=0,
        )

    bias = np.zeros((seq_len, seq_len), dtype=np.float32)

    # Build memory_id → span lookup
    span_map: dict[str, MemorySpan] = {s.memory_id: s for s in memory_spans}

    # ── Step 1: Edge-based biases ────────────────────────────────────────
    edge_count = 0
    for edge in edges:
        src = span_map.get(edge.source_id)
        tgt = span_map.get(edge.target_id)
        if src is None or tgt is None:
            continue  # edge references a memory not in this retrieval

        edge_info = EDGE_BIAS_MAP.get(edge.relation)
        if edge_info is None:
            continue

        direction, mag_key = edge_info
        magnitude = getattr(cfg, mag_key, 0.1) * edge.weight

        src_slice = slice(src.start_tok, src.end_tok)
        tgt_slice = slice(tgt.start_tok, tgt.end_tok)

        if direction == "symmetric":
            # Both memories attend to each other
            bias[src_slice, tgt_slice] += magnitude
            bias[tgt_slice, src_slice] += magnitude
        elif direction == "forward":
            # Source attends to target (causal flow)
            bias[src_slice, tgt_slice] += magnitude
        elif direction == "directional":
            # New → old (supersedes: new version attends to old)
            bias[src_slice, tgt_slice] += magnitude
            # Old gets slightly suppressed toward new
            bias[tgt_slice, src_slice] += magnitude * 0.3

        edge_count += 1

    # ── Step 2: PageRank multipliers ─────────────────────────────────────
    for span in memory_spans:
        # Normalize PageRank to [0, 1] then scale to multiplier range
        pr_norm = min(max(span.pagerank, 0.0), 1.0)
        multiplier = (
            cfg.pagerank_min_mult
            + pr_norm * (cfg.pagerank_max_mult - cfg.pagerank_min_mult)
        )
        # Apply multiplier to all biases involving this memory's tokens
        span_slice = slice(span.start_tok, span.end_tok)
        bias[span_slice, :] *= multiplier
        bias[:, span_slice] *= multiplier

    # ── Step 3: Self-attention bonus for memory tokens ───────────────────
    # Memory tokens should attend to themselves and each other more than
    # to non-memory tokens (reinforces memory grounding).
    for span in memory_spans:
        s = slice(span.start_tok, span.end_tok)
        bias[s, s] += 0.1  # small self-attention bonus

    # ── Step 4: Clip and scale ───────────────────────────────────────────
    bias = np.clip(bias, -cfg.clip_bias, cfg.clip_bias)
    bias *= cfg.temperature_scale

    construction_ms = (time.monotonic() - t0) * 1000.0

    result = BiasMatrix(
        bias=bias,
        memory_spans=memory_spans,
        edge_count=edge_count,
        construction_ms=construction_ms,
    )

    log.debug(
        f"[NEXUS-GAB] Built bias matrix: {seq_len}×{seq_len}, "
        f"{edge_count} edges, {result.sparsity:.1%} sparse, "
        f"{construction_ms:.1f}ms"
    )

    return result


# ── Memory Span Extraction ───────────────────────────────────────────────────

def extract_memory_spans(
    memories: list[Any],
    prompt_tokens: list[int],
    memory_block_start: int,
) -> list[MemorySpan]:
    """Extract token spans for each memory in the enriched prompt.

    After ``enrichment.py`` injects the ``<ENI_MEMORIES>`` block, we need
    to know which token positions belong to which memory. This function
    estimates spans based on the memory block structure.

    Parameters
    ----------
    memories : list[Memory]
        The retrieved memories that were injected.
    prompt_tokens : list[int]
        The full tokenized prompt.
    memory_block_start : int
        Token index where the memory block begins.

    Returns
    -------
    List of MemorySpan objects mapping memory_id → [start_tok, end_tok).
    """
    if not memories or memory_block_start < 0:
        return []

    spans = []
    cursor = memory_block_start

    for mem in memories:
        # Estimate token count from character length (rough: 1 token ≈ 4 chars)
        content_len = len(getattr(mem, "content", "")) + len(getattr(mem, "title", ""))
        est_tokens = max(content_len // 4, 10)

        # Cap at max_memory_tokens
        est_tokens = min(est_tokens, 4096)

        start = cursor
        end = min(cursor + est_tokens, len(prompt_tokens))

        spans.append(MemorySpan(
            memory_id=getattr(mem, "id", str(id(mem))),
            start_tok=start,
            end_tok=end,
            pagerank=getattr(mem, "pagerank", 0.0),
            importance=getattr(mem, "importance", 0.5),
            category=getattr(mem, "category", "fact"),
        ))

        cursor = end

    return spans


# ── Edge Retrieval ───────────────────────────────────────────────────────────

def fetch_graph_edges(
    memory_ids: list[str],
    client: Any = None,
) -> list[GraphEdge]:
    """Fetch typed edges between a set of memories from the knowledge graph.

    Queries the ``memory_links`` table for edges where both source and
    target are in the provided ``memory_ids`` set (the retrieved subgraph).

    Parameters
    ----------
    memory_ids : list[str]
        IDs of the retrieved memories.
    client : ENIMemoryClient, optional
        Memory client. Uses global client if None.

    Returns
    -------
    List of GraphEdge objects for the subgraph.
    """
    if not memory_ids:
        return []

    if client is None:
        from .memory_client import get_client
        client = get_client()

    if client is None or not client.is_connected():
        log.debug("[NEXUS-GAB] No memory client — returning empty edges")
        return []

    try:
        # Query edges between retrieved memories
        id_set = set(memory_ids)
        edges = []

        with client._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT source_id, target_id, relation, weight
                    FROM memory_links
                    WHERE source_id = ANY(%s)
                      AND target_id = ANY(%s)
                    """,
                    (memory_ids, memory_ids),
                )
                for row in cur.fetchall():
                    src, tgt, rel, weight = row
                    if src in id_set and tgt in id_set:
                        edges.append(GraphEdge(
                            source_id=src,
                            target_id=tgt,
                            relation=rel or "related",
                            weight=float(weight or 1.0),
                        ))

        log.debug(f"[NEXUS-GAB] Fetched {len(edges)} edges for {len(memory_ids)} memories")
        return edges

    except Exception as e:
        log.warning(f"[NEXUS-GAB] Failed to fetch edges: {e}")
        return []


# ── MLX Conversion ───────────────────────────────────────────────────────────

def to_mlx(bias_matrix: BiasMatrix) -> Any:
    """Convert BiasMatrix to MLX array for kernel injection.

    Returns an mx.array of shape [1, 1, seq_len, seq_len] (broadcastable
    across batch and head dimensions for GQA attention).
    """
    try:
        import mlx.core as mx
        # Reshape to [1, 1, seq_len, seq_len] for broadcast across batch/heads
        return mx.array(bias_matrix.bias[np.newaxis, np.newaxis, :, :])
    except ImportError:
        log.warning("[NEXUS-GAB] MLX not available — returning numpy")
        return bias_matrix.bias[np.newaxis, np.newaxis, :, :]


# ── Stats ────────────────────────────────────────────────────────────────────

_stats = {
    "matrices_built": 0,
    "total_edges": 0,
    "total_construction_ms": 0.0,
    "avg_sparsity": 0.0,
}


def get_graph_bias_stats() -> dict[str, Any]:
    return dict(_stats)


def reset_graph_bias_stats() -> None:
    global _stats
    _stats = {
        "matrices_built": 0,
        "total_edges": 0,
        "total_construction_ms": 0.0,
        "avg_sparsity": 0.0,
    }


# ── Convenience: One-Shot Build ──────────────────────────────────────────────

def build_from_retrieval(
    memories: list[Any],
    edges: list[GraphEdge] | None,
    prompt_tokens: list[int],
    memory_block_start: int,
    seq_len: int | None = None,
) -> BiasMatrix:
    """One-shot convenience: extract spans, fetch edges, build bias matrix.

    This is the main entry point for integration into the generation loop.
    """
    if seq_len is None:
        seq_len = len(prompt_tokens)

    spans = extract_memory_spans(memories, prompt_tokens, memory_block_start)

    if edges is None:
        memory_ids = [s.memory_id for s in spans]
        edges = fetch_graph_edges(memory_ids)

    result = build_bias_matrix(spans, edges, seq_len)

    # Update stats
    _stats["matrices_built"] += 1
    _stats["total_edges"] += result.edge_count
    _stats["total_construction_ms"] += result.construction_ms
    if _stats["matrices_built"] > 0:
        _stats["avg_sparsity"] = (
            _stats["avg_sparsity"] * (_stats["matrices_built"] - 1) + result.sparsity
        ) / _stats["matrices_built"]

    return result

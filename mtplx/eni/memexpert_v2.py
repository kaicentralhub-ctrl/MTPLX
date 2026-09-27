"""NEXUS L2 — MemExpert v2: Memory Graph as Forward-Pass Participant.

Per-layer learned gate that decides when to consult the memory graph
during inference. When the gate fires, memory embeddings are retrieved
mid-forward-pass and fused with layer hidden states via cross-attention.

This is NOT prompt injection or RAG. The memory graph participates
as a native expert in the computation graph — analogous to how MoE
experts contribute to the forward pass, but the "expert" is an
external knowledge graph.

Physics (per layer L):
```
h_L → gate_L = σ(w_gate[L] · h_L + b_gate[L])
if gate_L > threshold:
    q = W_q · h_L
    memories = HNSW_search(q, top_k=3)
    mem_kv = project_to_kv(memories)
    h_L = h_L + cross_attend(q, mem_kv)
```

Integration point: ``_forward_ar_optional_hidden()`` in ``generation.py``
(line ~1937) — layer-by-layer forward pass is accessible.

Research grounding: MemExpert (2026-09-13, built + verified as MoE
expert). This v2 moves it from message-level to *forward-pass-level*.
Retroformer (2024), MemoryLLM (2024) do memory-augmented generation
but not with typed graph structure or learned per-layer gates.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger("mtplx.eni.memexpert_v2")


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class MemExpertConfig:
    """Tunable parameters for forward-pass memory expert."""
    enabled: bool = True
    # Gate parameters
    gate_threshold: float = 0.5       # fire gate when sigmoid output > threshold
    gate_init_bias: float = -1.0      # initial bias (starts gated OFF, learns to open)
    # Retrieval
    retrieval_top_k: int = 3          # memories to retrieve per gate firing
    retrieval_dim: int = 768          # memory embedding dimension
    # Cross-attention fusion
    hidden_dim: int = 4096            # model hidden dimension
    num_heads: int = 8                # cross-attention heads
    fusion_scale: float = 0.1         # scale of memory contribution (residual)
    # Performance
    max_gate_firings_per_token: int = 3  # cap memory lookups per decode step
    async_retrieval: bool = True       # fire retrieval on background thread
    cache_retrievals: bool = True       # cache retrieval results per query


_config = MemExpertConfig()


def configure(cfg: MemExpertConfig | None = None, **kwargs: Any) -> MemExpertConfig:
    """Update global config."""
    global _config
    if cfg is not None:
        _config = cfg
    for k, v in kwargs.items():
        if hasattr(_config, k):
            setattr(_config, k, v)
    return _config


def get_config() -> MemExpertConfig:
    return _config


# ── Gate Network ─────────────────────────────────────────────────────────────

class LayerGate:
    """Learned scalar gate for one layer.

    Computes gate_L = σ(w_gate · h_L + b_gate). When gate > threshold,
    the memory expert activates for this layer.

    Parameters are tiny: 1 weight vector + 1 bias per layer.
    For 28 layers × 4096 hidden = 114K params total (~450KB in FP32).
    """

    def __init__(self, hidden_dim: int, layer_idx: int) -> None:
        self.hidden_dim = hidden_dim
        self.layer_idx = layer_idx

        # Initialize with small random weights + negative bias (gated OFF)
        self.w_gate = np.random.randn(hidden_dim).astype(np.float32) * 0.01
        self.b_gate = _config.gate_init_bias

        # Training state (for future calibration)
        self._activation_count = 0
        self._total_count = 0

    def forward(self, hidden: np.ndarray) -> float:
        """Compute gate value from hidden state.

        Parameters
        ----------
        hidden : np.ndarray
            Layer hidden state, shape [hidden_dim] or [seq_len, hidden_dim].

        Returns
        -------
        Gate value in [0, 1] (sigmoid of linear projection).
        """
        if hidden.ndim == 1:
            score = np.dot(self.w_gate, hidden) + self.b_gate
        else:
            score = hidden @ self.w_gate + self.b_gate
            score = float(np.mean(score))  # average across positions

        gate = 1.0 / (1.0 + np.exp(-score))

        self._total_count += 1
        if gate > _config.gate_threshold:
            self._activation_count += 1

        return gate

    @property
    def activation_rate(self) -> float:
        if self._total_count == 0:
            return 0.0
        return self._activation_count / self._total_count

    def set_params(self, w_gate: np.ndarray, b_gate: float) -> None:
        """Load trained parameters."""
        self.w_gate = w_gate.astype(np.float32)
        self.b_gate = float(b_gate)


class GateNetwork:
    """Collection of per-layer gates for the entire model.

    Usage::

        gates = GateNetwork(num_layers=28, hidden_dim=4096)
        gate_values = gates.forward_all(layer_hiddens)
        active_layers = [i for i, g in enumerate(gate_values) if g > threshold]
    """

    def __init__(self, num_layers: int, hidden_dim: int) -> None:
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.gates = [LayerGate(hidden_dim, i) for i in range(num_layers)]

    def forward(self, layer_idx: int, hidden: np.ndarray) -> float:
        """Compute gate for one layer."""
        return self.gates[layer_idx].forward(hidden)

    def forward_all(self, layer_hiddens: list[np.ndarray]) -> list[float]:
        """Compute gates for all layers."""
        return [
            self.gates[i].forward(h)
            for i, h in enumerate(layer_hiddens)
            if i < self.num_layers
        ]

    def get_active_layers(self, gate_values: list[float]) -> list[int]:
        """Return indices of layers where gate fired."""
        return [
            i for i, g in enumerate(gate_values)
            if g > _config.gate_threshold
        ]

    @property
    def total_params(self) -> int:
        return self.num_layers * (self.hidden_dim + 1)

    @property
    def activation_rates(self) -> dict[int, float]:
        return {i: g.activation_rate for i, g in enumerate(self.gates)}


# ── Memory Retrieval (Mid-Forward-Pass) ──────────────────────────────────────

class MidPassRetriever:
    """Retrieves memories during the forward pass.

    When a gate fires, this retriever queries the memory graph using
    the current hidden state as a query vector. Results are cached
    to avoid redundant lookups.
    """

    def __init__(
        self,
        client: Any = None,
        config: MemExpertConfig | None = None,
    ) -> None:
        self.config = config or _config
        self._client = client
        self._cache: dict[str, list[tuple[np.ndarray, float]]] = {}
        self._retrieval_count = 0
        self._cache_hits = 0

    def retrieve(
        self,
        query_hidden: np.ndarray,
        top_k: int | None = None,
    ) -> list[tuple[np.ndarray, float]]:
        """Retrieve memories by hidden-state query.

        Parameters
        ----------
        query_hidden : np.ndarray
            Hidden state to use as query, shape [hidden_dim].
        top_k : int, optional
            Number of memories to retrieve.

        Returns
        -------
        List of (embedding, similarity) tuples.
        """
        k = top_k or self.config.retrieval_top_k

        # Check cache (round hidden to 4 decimal places for cache key)
        cache_key = hash(query_hidden.round(4).tobytes()[:64])
        if self.config.cache_retrievals and cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]

        self._retrieval_count += 1

        try:
            if self._client is None:
                from .memory_client import get_client
                self._client = get_client()

            if self._client is None or not self._client.is_connected():
                return []

            # Use the hidden state as an embedding query
            # In practice, we'd project hidden_dim → retrieval_dim first
            query_embedding = self._project_to_retrieval_dim(query_hidden)

            results = self._client.retrieve(
                query_embedding, k=k
            ) if hasattr(self._client, 'retrieve') else []

            # Convert to (embedding, similarity) tuples
            memories = []
            for mem in results:
                emb = getattr(mem, 'embedding', np.zeros(self.config.retrieval_dim))
                sim = getattr(mem, 'similarity', 0.0)
                memories.append((np.asarray(emb, dtype=np.float32), sim))

            if self.config.cache_retrievals:
                self._cache[cache_key] = memories

            return memories

        except Exception as e:
            log.debug(f"[NEXUS-ME] Retrieval failed: {e}")
            return []

    def _project_to_retrieval_dim(self, hidden: np.ndarray) -> np.ndarray:
        """Project hidden state to retrieval embedding dimension.

        In a full implementation, this would use a learned projection
        matrix W_proj: [hidden_dim → retrieval_dim]. For now, we use
        a simple average-pooling + truncation.
        """
        if hidden.ndim == 2:
            hidden = np.mean(hidden, axis=0)

        target_dim = self.config.retrieval_dim
        if len(hidden) >= target_dim:
            return hidden[:target_dim]
        else:
            return np.pad(hidden, (0, target_dim - len(hidden)))

    @property
    def stats(self) -> dict[str, Any]:
        total = self._retrieval_count + self._cache_hits
        return {
            "retrievals": self._retrieval_count,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": self._cache_hits / max(total, 1),
            "cached_queries": len(self._cache),
        }


# ── Cross-Attention Fusion ───────────────────────────────────────────────────

class MemoryCrossAttention:
    """Fuses memory embeddings with layer hidden states via cross-attention.

    When the gate fires and memories are retrieved, this module computes:
        output = h_L + scale * CrossAttention(Q=h_L, K=mem_embeddings, V=mem_embeddings)

    The scale parameter controls how much memory contributes to the
    hidden state (residual connection prevents catastrophic forgetting).
    """

    def __init__(self, hidden_dim: int, config: MemExpertConfig | None = None) -> None:
        self.config = config or _config
        self.hidden_dim = hidden_dim

        # Simple projection matrices (in practice, these would be trained)
        # For now: identity + scaling
        self.scale = self.config.fusion_scale

    def fuse(
        self,
        hidden: np.ndarray,
        memory_embeddings: list[np.ndarray],
    ) -> np.ndarray:
        """Fuse memory embeddings into hidden state.

        Parameters
        ----------
        hidden : np.ndarray
            Current layer hidden state, shape [hidden_dim].
        memory_embeddings : list[np.ndarray]
            Retrieved memory embeddings.

        Returns
        -------
        Fused hidden state, shape [hidden_dim].
        """
        if not memory_embeddings or hidden.size == 0:
            return hidden

        # Simple weighted average fusion
        # In full implementation: multi-head cross-attention with Q/K/V projections
        mem_stack = np.stack(memory_embeddings)  # [n_mem, mem_dim]

        # Project memory embeddings to hidden_dim if needed
        if mem_stack.shape[1] != self.hidden_dim:
            mem_stack = self._project_to_hidden(mem_stack)

        # Compute attention weights (dot product similarity)
        attn_scores = mem_stack @ hidden  # [n_mem]
        attn_weights = np.exp(attn_scores - np.max(attn_scores))
        attn_weights = attn_weights / (np.sum(attn_weights) + 1e-8)

        # Weighted sum of memory embeddings
        memory_context = np.sum(
            mem_stack * attn_weights[:, np.newaxis], axis=0
        )

        # Residual fusion
        fused = hidden + self.scale * memory_context

        return fused

    def _project_to_hidden(self, mem_stack: np.ndarray) -> np.ndarray:
        """Project memory embeddings to hidden dimension."""
        n_mem, mem_dim = mem_stack.shape
        if mem_dim == self.hidden_dim:
            return mem_stack

        if mem_dim > self.hidden_dim:
            return mem_stack[:, :self.hidden_dim]
        else:
            return np.pad(mem_stack, ((0, 0), (0, self.hidden_dim - mem_dim)))


# ── MemExpert v2 — Full Pipeline ─────────────────────────────────────────────

class MemExpertV2:
    """Complete memory expert for forward-pass integration.

    Combines gate network, mid-pass retrieval, and cross-attention
    fusion into a single drop-in module for the generation loop.

    Usage in generation.py::

        mem_expert = MemExpertV2(num_layers=28, hidden_dim=4096)

        # In the forward pass loop:
        for layer_idx, h_L in enumerate(layer_hiddens):
            h_L = mem_expert.process_layer(layer_idx, h_L)
            # ... continue with layer computation
    """

    def __init__(
        self,
        num_layers: int = 28,
        hidden_dim: int = 4096,
        client: Any = None,
        config: MemExpertConfig | None = None,
    ) -> None:
        self.config = config or _config
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.gates = GateNetwork(num_layers, hidden_dim)
        self.retriever = MidPassRetriever(client, config)
        self.fusion = MemoryCrossAttention(hidden_dim, config)

        self._firings_this_token = 0
        self._total_gate_evaluations = 0
        self._total_fusions = 0

    def process_layer(
        self,
        layer_idx: int,
        hidden: np.ndarray,
    ) -> np.ndarray:
        """Process one layer's hidden state through the memory expert.

        Parameters
        ----------
        layer_idx : int
            Current layer index.
        hidden : np.ndarray
            Layer hidden state to potentially augment.

        Returns
        -------
        (Possibly augmented) hidden state.
        """
        if not self.config.enabled:
            return hidden

        self._total_gate_evaluations += 1

        # Check gate
        gate_value = self.gates.forward(layer_idx, hidden)
        if gate_value <= self.config.gate_threshold:
            return hidden

        # Rate limit
        if self._firings_this_token >= self.config.max_gate_firings_per_token:
            return hidden

        # Retrieve memories
        memories = self.retriever.retrieve(hidden)
        if not memories:
            return hidden

        self._firings_this_token += 1

        # Fuse memories into hidden state
        mem_embeddings = [emb for emb, _ in memories]
        fused = self.fusion.fuse(hidden, mem_embeddings)

        self._total_fusions += 1

        log.debug(
            f"[NEXUS-ME] Layer {layer_idx}: gate={gate_value:.3f}, "
            f"memories={len(memories)}, fusion applied"
        )

        return fused

    def start_new_token(self) -> None:
        """Reset per-token firing counter. Call at each decode step."""
        self._firings_this_token = 0

    def load_gates(self, checkpoints: dict[int, tuple[np.ndarray, float]]) -> None:
        """Load trained gate parameters from checkpoint.

        Parameters
        ----------
        checkpoints : dict[int, tuple[np.ndarray, float]]
            Mapping of layer_idx → (w_gate, b_gate).
        """
        for layer_idx, (w, b) in checkpoints.items():
            if layer_idx < self.num_layers:
                self.gates.gates[layer_idx].set_params(w, b)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_gate_evaluations": self._total_gate_evaluations,
            "total_fusions": self._total_fusions,
            "fusion_rate": (
                self._total_fusions / max(self._total_gate_evaluations, 1)
            ),
            "retrieval": self.retriever.stats,
            "gate_activation_rates": self.gates.activation_rates,
            "total_params": self.gates.total_params,
        }


# ── Global Instance ──────────────────────────────────────────────────────────

_expert: MemExpertV2 | None = None


def get_memexpert() -> MemExpertV2 | None:
    """Get the global MemExpert v2 instance."""
    return _expert


def init_memexpert(
    num_layers: int = 28,
    hidden_dim: int = 4096,
    client: Any = None,
    config: MemExpertConfig | None = None,
) -> MemExpertV2:
    """Initialize the global MemExpert v2."""
    global _expert
    _expert = MemExpertV2(num_layers, hidden_dim, client, config)
    return _expert


def shutdown_memexpert() -> None:
    """Shutdown and clear the global MemExpert v2."""
    global _expert
    _expert = None


# ── Stats ────────────────────────────────────────────────────────────────────

_stats = {
    "experts_created": 0,
    "total_gate_evaluations": 0,
    "total_fusions": 0,
}


def get_memexpert_stats() -> dict[str, Any]:
    result = dict(_stats)
    if _expert is not None:
        result["current_expert"] = _expert.stats
    return result


def reset_memexpert_stats() -> None:
    global _stats
    _stats = {
        "experts_created": 0,
        "total_gate_evaluations": 0,
        "total_fusions": 0,
    }

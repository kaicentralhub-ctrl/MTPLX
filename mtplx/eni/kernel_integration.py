"""ENI Kernel Integration — Complete pipeline from query to Metal kernel.

This module provides the end-to-end integration:
    Query → Memory retrieval → Embedding projection → Kernel registry

Tier 2 upgrades (composable, independently toggleable):
    2.1 Quantization: Compress projected K/V — validates roundtrip, enables cache
    2.2 Layer Routing: Route memories to relevant layers only (50-70% savings)
    2.3 Hierarchical: Two-pass cluster-based retrieval for 10× capacity

Tier 3 multi-modal extension:
    3.1 Visual: CLIP/SigLIP image embeddings → K/V
    3.2 Code: AST-aware code embeddings with structural metadata
    3.3 Audio: Whisper transcription → text embedding → K/V

Usage:
    from mtplx.eni.kernel_integration import prepare_memory_for_kernel
    
    # At request start (before generation) — auto-detects multi-modal
    success = prepare_memory_for_kernel(messages)
    
    # During generation: Metal kernel automatically uses registered K/V
    
    # At request end (handled by openai.py)
    clear_mem_kv_registry()
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from .config import get_config, ENIConfig
from .memory_client import get_client, Memory
from .memory_projection import get_projector, project_memories_for_kernel
from .sdpa_mem_fused import register_mem_kv_for_request, clear_mem_kv_registry
from .memory_capacity import (
    get_capacity_config,
    get_hierarchical_retriever,
    quantize_kv_pair,
    dequantize_kv_pair,
    compute_layer_mask,
    filter_registry_by_routing,
    get_capacity_stats,
)
from .multimodal_memory import (
    get_multimodal_config,
    get_retriever as get_multimodal_retriever,
    get_multimodal_stats,
    Modality,
)

log = logging.getLogger("mtplx.eni.kernel_integration")


# ═══════════════════════════════════════════════════════════════════════════════
# STATS
# ═══════════════════════════════════════════════════════════════════════════════

_integration_stats = {
    "requests_processed": 0,
    "memories_projected": 0,
    "total_projection_ms": 0.0,
    "last_projection_ms": 0.0,
    "failures": 0,
    # Tier 2 stats
    "hierarchical_retrievals": 0,
    "flat_retrievals": 0,
    "layers_routed": 0,
    "layers_skipped": 0,
    "quant_roundtrips": 0,
    # Tier 3 stats
    "multimodal_retrievals": 0,
    "modalities_seen": {},
}


# ═══════════════════════════════════════════════════════════════════════════════
# QUERY EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_memory_for_kernel(
    messages: list[dict[str, Any]],
    config: ENIConfig | None = None,
) -> bool:
    """Full pipeline: retrieve memories → project to K/V → register for kernel.
    
    Pipeline stages with Tier 2 integration:
        1. Extract query from messages
        2. Get memory client
        3. Retrieve memories (Tier 2.3 hierarchical or flat fallback)
        4. Extract importance weights + categories
        5. Project embeddings to model K/V space (per-layer)
        6. Layer routing (Tier 2.2) — filter to relevant layers only
        7. Quantization roundtrip (Tier 2.1) — validate compression path
        8. Register K/V tensors for Metal kernel
    
    Args:
        messages: The conversation messages
        config: Optional ENI config override
        
    Returns:
        True if memories were successfully registered, False otherwise
    """
    global _integration_stats
    
    config = config or get_config()
    
    if not config.enabled:
        return False
    
    t0 = time.perf_counter()
    cap_config = get_capacity_config()
    
    try:
        # ── Step 1: Extract query ───────────────────────────────────────
        query = _extract_query(messages)
        if not query:
            log.debug("[ENI Kernel] No query found in messages")
            return False
        
        # ── Step 2: Get memory client ───────────────────────────────────
        client = get_client()
        if not client.is_connected():
            log.debug("[ENI Kernel] Memory client not connected")
            return False
        
        # ── Step 3: Retrieve memories ───────────────────────────────────
        # Tier 2.3 — hierarchical clustering: two-pass retrieval
        memories = None
        embeddings = None
        
        if cap_config.clustering.enabled:
            retriever = get_hierarchical_retriever()
            memories, embeddings = retriever.retrieve(client, query)
            if memories:
                _integration_stats["hierarchical_retrievals"] += 1
                log.debug(
                    f"[ENI Kernel] Hierarchical: {len(memories)} memories "
                    f"from {retriever._clusters_hit} clusters"
                )
        
        # Fallback to flat retrieval if hierarchical didn't return results
        if not memories:
            memories, embeddings = client.retrieve_with_embeddings(
                query,
                k=config.memory_k,
                min_similarity=config.memory_min_similarity,
            )
            _integration_stats["flat_retrievals"] += 1
        
        if not memories or embeddings is None:
            log.debug("[ENI Kernel] No memories retrieved")
            return False
        
        log.debug(
            f"[ENI Kernel] Retrieved {len(memories)} memories, "
            f"embeddings shape {embeddings.shape}"
        )
        
        # ── Step 4: Extract importance weights + categories ─────────────
        importance_weights = np.array(
            [m.importance for m in memories], dtype=np.float32
        )
        categories = [m.category for m in memories]
        
        # ── Step 5: Project embeddings to model K/V space ───────────────
        projector = get_projector()
        # Initialize with per-layer=True for layer specialization (Tier 1.2)
        if not projector._initialized:
            from .memory_projection import configure_projector
            configure_projector(per_layer=True, n_layers=64)
            projector = get_projector()
        
        layer_kv_map = projector.project_memories_for_request(embeddings)
        
        if not layer_kv_map:
            log.warning("[ENI Kernel] Projection failed")
            _integration_stats["failures"] += 1
            return False
        
        total_layers = len(layer_kv_map)
        
        # ── Step 6: Layer routing (Tier 2.2) ────────────────────────────
        # Filter to relevant layers based on memory categories.
        # Code memories → early layers, decision memories → late layers.
        # Layers not in the mask get no entry → attention_split.py falls
        # back to stock SDPA via the bail-and-None contract.
        if cap_config.layer_routing_enabled and categories:
            active_layers = compute_layer_mask(categories, n_layers=64)
            layer_kv_map = filter_registry_by_routing(layer_kv_map, active_layers)
            routed = len(layer_kv_map)
            skipped = total_layers - routed
            _integration_stats["layers_routed"] += routed
            _integration_stats["layers_skipped"] += skipped
            log.debug(
                f"[ENI Kernel] Layer routing: {routed}/{total_layers} active, "
                f"{skipped} skipped (cats={set(categories)})"
            )
        
        # ── Step 7: Quantization roundtrip (Tier 2.1) ──────────────────
        # Validates the Q4/Q8 compress→decompress path on one layer.
        # Real savings materialize when we add a projection cache that
        # stores quantized K/V across requests (same memories → skip
        # re-projection, just dequantize from cache).
        if cap_config.quantization.enabled and layer_kv_map:
            sample_layer = next(iter(layer_kv_map))
            k_orig, v_orig = layer_kv_map[sample_layer]
            q_k, q_v = quantize_kv_pair(
                k_orig, v_orig, cap_config.quantization
            )
            k_rt, v_rt = dequantize_kv_pair(q_k, q_v)
            # Verify roundtrip shape integrity
            assert k_rt.shape == k_orig.shape, (
                f"K shape mismatch: {k_rt.shape} vs {k_orig.shape}"
            )
            assert v_rt.shape == v_orig.shape, (
                f"V shape mismatch: {v_rt.shape} vs {v_orig.shape}"
            )
            _integration_stats["quant_roundtrips"] += 1
            log.debug(
                f"[ENI Kernel] Q{cap_config.quantization.bits} roundtrip "
                f"validated on layer {sample_layer}"
            )
        
        # ── Step 8: Register for Metal kernel ───────────────────────────
        register_mem_kv_for_request(layer_kv_map, importance_weights)
        
        # ── Stats ───────────────────────────────────────────────────────
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _integration_stats["requests_processed"] += 1
        _integration_stats["memories_projected"] += len(memories)
        _integration_stats["total_projection_ms"] += elapsed_ms
        _integration_stats["last_projection_ms"] = elapsed_ms
        
        log.info(
            f"[ENI Kernel] Registered {len(memories)} memories for "
            f"{len(layer_kv_map)} layers in {elapsed_ms:.1f}ms "
            f"(cats={set(categories)})"
        )
        return True
        
    except Exception as e:
        log.warning(f"[ENI Kernel] Integration failed: {e}")
        _integration_stats["failures"] += 1
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 3: MULTI-MODAL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_multimodal_for_kernel(
    messages: list[dict[str, Any]],
    config: ENIConfig | None = None,
) -> bool:
    """Multi-modal pipeline: retrieve mixed-modality memories → per-modality
    projection → merge K/V → register for Metal kernel.
    
    This extends prepare_memory_for_kernel() with modality-aware retrieval
    and projection. Memories from different modalities (text, visual, code,
    audio) get their own W_k/W_v projection weights, then merge into a
    single K/V registry that the Metal kernel consumes.
    
    Falls back to text-only prepare_memory_for_kernel() if multi-modal
    is disabled or if the schema hasn't been migrated yet.
    
    Args:
        messages: The conversation messages
        config: Optional ENI config override
        
    Returns:
        True if memories were successfully registered
    """
    global _integration_stats
    
    mm_config = get_multimodal_config()
    if not mm_config.enabled:
        return prepare_memory_for_kernel(messages, config)
    
    config = config or get_config()
    if not config.enabled:
        return False
    
    t0 = time.perf_counter()
    cap_config = get_capacity_config()
    
    try:
        # ── Step 1: Extract query ───────────────────────────────────────
        query = _extract_query(messages)
        if not query:
            return False
        
        # ── Step 2: Get memory client ───────────────────────────────────
        client = get_client()
        if not client.is_connected():
            return False
        
        # ── Step 3: Multi-modal retrieval ───────────────────────────────
        # Retrieves memories across all modalities, returns modality tags
        retriever = get_multimodal_retriever()
        memories, embeddings, modality_tags = retriever.retrieve_with_modalities(
            client, query, k=config.memory_k,
            min_similarity=config.memory_min_similarity,
        )
        
        if not memories or embeddings is None:
            # Fallback to standard text-only pipeline
            log.debug("[ENI Kernel MM] No multimodal results, falling back to text-only")
            return prepare_memory_for_kernel(messages, config)
        
        # Track modality distribution
        for tag in modality_tags:
            mod_name = tag.value
            _integration_stats["modalities_seen"][mod_name] = (
                _integration_stats["modalities_seen"].get(mod_name, 0) + 1
            )
        
        log.debug(
            f"[ENI Kernel MM] Retrieved {len(memories)} memories across "
            f"modalities: {set(t.value for t in modality_tags)}"
        )
        
        # ── Step 4: Extract importance weights ──────────────────────────
        importance_weights = np.array(
            [m.importance for m in memories], dtype=np.float32
        )
        
        # ── Step 5: Per-modality kernel projection ──────────────────────
        # Each modality gets its own W_k/W_v weights, then K/V tensors
        # are concatenated across modalities per-layer
        layer_kv_map = retriever.project_for_kernel(embeddings, modality_tags)
        
        if not layer_kv_map:
            log.warning("[ENI Kernel MM] Multi-modal projection failed")
            _integration_stats["failures"] += 1
            return False
        
        total_layers = len(layer_kv_map)
        
        # ── Step 6: Layer routing (Tier 2.2) ────────────────────────────
        # Multi-modal routing uses modality-specific layer ranges
        # (already applied in MultiModalKernelProjector.project()),
        # but we can further filter by memory categories
        categories = [m.category for m in memories]
        if cap_config.layer_routing_enabled and categories:
            active_layers = compute_layer_mask(categories, n_layers=64)
            layer_kv_map = filter_registry_by_routing(layer_kv_map, active_layers)
            routed = len(layer_kv_map)
            skipped = total_layers - routed
            _integration_stats["layers_routed"] += routed
            _integration_stats["layers_skipped"] += skipped
        
        # ── Step 7: Quantization roundtrip (Tier 2.1) ──────────────────
        if cap_config.quantization.enabled and layer_kv_map:
            sample_layer = next(iter(layer_kv_map))
            k_orig, v_orig = layer_kv_map[sample_layer]
            q_k, q_v = quantize_kv_pair(k_orig, v_orig, cap_config.quantization)
            k_rt, v_rt = dequantize_kv_pair(q_k, q_v)
            assert k_rt.shape == k_orig.shape
            assert v_rt.shape == v_orig.shape
            _integration_stats["quant_roundtrips"] += 1
        
        # ── Step 8: Register for Metal kernel ───────────────────────────
        register_mem_kv_for_request(layer_kv_map, importance_weights)
        
        # ── Stats ───────────────────────────────────────────────────────
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _integration_stats["requests_processed"] += 1
        _integration_stats["memories_projected"] += len(memories)
        _integration_stats["multimodal_retrievals"] += 1
        _integration_stats["total_projection_ms"] += elapsed_ms
        _integration_stats["last_projection_ms"] = elapsed_ms
        
        log.info(
            f"[ENI Kernel MM] Registered {len(memories)} multi-modal memories "
            f"for {len(layer_kv_map)} layers in {elapsed_ms:.1f}ms "
            f"(modalities={set(t.value for t in modality_tags)})"
        )
        return True
        
    except Exception as e:
        log.warning(f"[ENI Kernel MM] Multi-modal integration failed: {e}")
        _integration_stats["failures"] += 1
        # Fallback to text-only
        return prepare_memory_for_kernel(messages, config)


# ═══════════════════════════════════════════════════════════════════════════════
# STATS / RESET
# ═══════════════════════════════════════════════════════════════════════════════

def get_integration_stats() -> dict:
    """Get kernel integration statistics including Tier 2+3 stats."""
    stats = dict(_integration_stats)
    
    # Add average projection time
    if stats["requests_processed"] > 0:
        stats["avg_projection_ms"] = round(
            stats["total_projection_ms"] / stats["requests_processed"], 2
        )
    else:
        stats["avg_projection_ms"] = 0.0
    
    # Add projector stats
    from .memory_projection import get_projection_stats
    stats["projector"] = get_projection_stats()
    
    # Add kernel stats
    from .sdpa_mem_fused import get_mem_fused_stats
    stats["kernel"] = get_mem_fused_stats()
    
    # Add Tier 2 capacity stats
    stats["capacity"] = get_capacity_stats()
    
    # Add Tier 3 multi-modal stats
    stats["multimodal"] = get_multimodal_stats()
    
    return stats


def reset_integration_stats() -> None:
    """Reset integration statistics."""
    global _integration_stats
    _integration_stats = {
        "requests_processed": 0,
        "memories_projected": 0,
        "total_projection_ms": 0.0,
        "last_projection_ms": 0.0,
        "failures": 0,
        "hierarchical_retrievals": 0,
        "flat_retrievals": 0,
        "layers_routed": 0,
        "layers_skipped": 0,
        "quant_roundtrips": 0,
        "multimodal_retrievals": 0,
        "modalities_seen": {},
    }

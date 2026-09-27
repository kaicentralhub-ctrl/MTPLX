"""ENI Memory Integration for MTPLX.

Native memory-augmented generation for MTPLX inference engine.
Provides:
- PostgreSQL memory graph client
- Semantic memory retrieval (HNSW vectors)
- H2O-aware memory injection
- Session-aware caching with memory hashes
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import ENIConfig, get_config, is_enabled, reload_config
from .memory_client import ENIMemoryClient, get_client, close_client
from .enrichment import enrich_messages, compute_injection_hash, enrich_with_hash
from .hooks import (
    enrich_messages_for_generation,
    get_current_injection_hash,
    patch_encode_messages,
    install_hooks,
    get_hook_stats,
)
from .kv_cache import (
    maybe_inject_kv,
    capture_kv_async,
    store_kv,
    load_kv,
    get_kv_stats,
    KVCacheConfig,
    configure as configure_kv_cache,
)
from .adaptive_sampling import (
    get_recommended_draft_temp,
    record_accept_rate_async,
    extract_accept_rate_from_generated,
    get_adaptive_sampling_stats,
    AdaptiveSamplingConfig,
    configure as configure_adaptive_sampling,
)
from .sdpa_mem_fused import (
    sdpa_mem_fused_q1,
    register_mem_kv_for_request,
    get_mem_kv_for_layer,
    clear_mem_kv_registry,
    get_mem_fused_stats,
    mem_fused_bail_counts,
)
from .memory_projection import (
    MemoryProjector,
    ProjectionConfig,
    get_projector,
    configure_projector,
    project_memories_for_kernel,
    get_projection_stats,
)
from .kernel_integration import (
    prepare_memory_for_kernel,
    prepare_multimodal_for_kernel,
    get_integration_stats,
    reset_integration_stats,
)
from .memory_capacity import (
    QuantConfig,
    quantize_kv_pair,
    dequantize_kv_pair,
    memory_savings_ratio,
    LAYER_ROUTING_TABLE,
    compute_layer_mask,
    compute_per_memory_layer_masks,
    filter_registry_by_routing,
    ClusterConfig,
    HierarchicalRetriever,
    CapacityConfig,
    configure_capacity,
    get_capacity_config,
    get_hierarchical_retriever,
    get_capacity_stats,
)
from .multimodal_memory import (
    Modality,
    ModalityConfig,
    MODALITY_DEFAULTS,
    BaseEmbeddingAdapter,
    TextEmbeddingAdapter,
    VisualEmbeddingAdapter,
    CodeEmbeddingAdapter,
    AudioEmbeddingAdapter,
    AlignmentProjector,
    MultiModalKernelProjector,
    MultiModalIngester,
    MultiModalRetriever,
    MultiModalConfig,
    configure_multimodal,
    get_multimodal_config,
    get_ingester,
    get_retriever,
    get_multimodal_stats,
    migrate_multimodal_schema,
)
from .soma_hooks import (
    on_user_message,
    on_assistant_response,
    on_session_end,
    get_ipc_client as get_soma_ipc_client,
    SomaSignal,
    detect_correction,
)
from .graph_attention_bias import (
    build_bias_matrix,
    build_from_retrieval,
    extract_memory_spans,
    fetch_graph_edges,
    to_mlx as graph_bias_to_mlx,
    get_graph_bias_stats,
    GraphBiasConfig,
    MemorySpan,
    GraphEdge,
    BiasMatrix,
    configure as configure_graph_bias,
)
from .memory_primed_spec import (
    MemoryPrimedDraftSampler,
    MemoryPatternTrie,
    extract_patterns_from_tokens,
    boost_draft_logits,
    boost_logits_if_active,
    get_active_sampler as get_active_spec_sampler,
    extract_memory_token_seqs,
    get_memory_spec_stats,
    MemorySpecConfig,
    configure as configure_memory_spec,
)
from .causal_verification import (
    CausalVerifier,
    EntityDetector,
    ClaimExtractor,
    CausalGraph,
    create_token_callback,
    get_causal_verify_stats,
    CausalVerifyConfig,
    configure as configure_causal_verify,
)
from .memory_bank import (
    PersistentMemoryBank,
    init_memory_bank,
    get_memory_bank,
    shutdown_memory_bank,
    select_bank_memories,
    score_memory,
    get_memory_bank_stats,
    MemoryBankConfig,
    configure as configure_memory_bank,
)
from .memexpert_v2 import (
    MemExpertV2,
    LayerGate,
    GateNetwork,
    MidPassRetriever,
    MemoryCrossAttention,
    init_memexpert,
    get_memexpert,
    shutdown_memexpert,
    get_memexpert_stats,
    MemExpertConfig,
    configure as configure_memexpert,
)

__all__ = [
    # Config
    "ENIConfig",
    "get_config",
    "is_enabled",
    "reload_config",
    # Memory client
    "ENIMemoryClient",
    "get_client",
    "close_client",
    # Enrichment
    "enrich_messages",
    "compute_injection_hash",
    "enrich_with_hash",
    # Hooks (for server integration)
    "enrich_messages_for_generation",
    "get_current_injection_hash",
    "patch_encode_messages",
    "install_hooks",
    "get_hook_stats",
    # KV Cache Pre-Computation (Phase A / T1)
    "maybe_inject_kv",
    "capture_kv_async",
    "store_kv",
    "load_kv",
    "get_kv_stats",
    "KVCacheConfig",
    "configure_kv_cache",
    # Adaptive Sampling (Phase B / T7)
    "get_recommended_draft_temp",
    "record_accept_rate_async",
    "extract_accept_rate_from_generated",
    "get_adaptive_sampling_stats",
    "AdaptiveSamplingConfig",
    "configure_adaptive_sampling",
    # Memory-Fused SDPA (Phase C / T8)
    "sdpa_mem_fused_q1",
    "register_mem_kv_for_request",
    "get_mem_kv_for_layer",
    "clear_mem_kv_registry",
    "get_mem_fused_stats",
    "mem_fused_bail_counts",
    # Memory Projection (Phase C Integration)
    "MemoryProjector",
    "ProjectionConfig",
    "get_projector",
    "configure_projector",
    "project_memories_for_kernel",
    "get_projection_stats",
    # Kernel Integration (Phase C Complete Pipeline + Tier 3 MM)
    "prepare_memory_for_kernel",
    "prepare_multimodal_for_kernel",
    "get_integration_stats",
    "reset_integration_stats",
    # Memory Capacity (Tier 2)
    "QuantConfig",
    "quantize_kv_pair",
    "dequantize_kv_pair",
    "memory_savings_ratio",
    "LAYER_ROUTING_TABLE",
    "compute_layer_mask",
    "compute_per_memory_layer_masks",
    "filter_registry_by_routing",
    "ClusterConfig",
    "HierarchicalRetriever",
    "CapacityConfig",
    "configure_capacity",
    "get_capacity_config",
    "get_hierarchical_retriever",
    "get_capacity_stats",
    # Multi-Modal Memory (Tier 3)
    "Modality",
    "ModalityConfig",
    "MODALITY_DEFAULTS",
    "BaseEmbeddingAdapter",
    "TextEmbeddingAdapter",
    "VisualEmbeddingAdapter",
    "CodeEmbeddingAdapter",
    "AudioEmbeddingAdapter",
    "AlignmentProjector",
    "MultiModalKernelProjector",
    "MultiModalIngester",
    "MultiModalRetriever",
    "MultiModalConfig",
    "configure_multimodal",
    "get_multimodal_config",
    "get_ingester",
    "get_retriever",
    "get_multimodal_stats",
    "migrate_multimodal_schema",
    # SOMA Integration Hooks
    "on_user_message",
    "on_assistant_response",
    "on_session_end",
    "get_soma_ipc_client",
    "SomaSignal",
    "detect_correction",
    # Graph-Attention Bias (NEXUS L2 / Tier 2)
    "build_bias_matrix",
    "build_from_retrieval",
    "extract_memory_spans",
    "fetch_graph_edges",
    "graph_bias_to_mlx",
    "get_graph_bias_stats",
    "GraphBiasConfig",
    "MemorySpan",
    "GraphEdge",
    "BiasMatrix",
    "configure_graph_bias",
    # Memory-Primed Speculative Decoding (NEXUS L3 / Tier 3)
    "MemoryPrimedDraftSampler",
    "MemoryPatternTrie",
    "extract_patterns_from_tokens",
    "boost_draft_logits",
    "boost_logits_if_active",
    "get_active_spec_sampler",
    "extract_memory_token_seqs",
    "get_memory_spec_stats",
    "MemorySpecConfig",
    "configure_memory_spec",
    # Causal Chain Verification (NEXUS L4 / Tier 6)
    "CausalVerifier",
    "EntityDetector",
    "ClaimExtractor",
    "CausalGraph",
    "create_token_callback",
    "get_causal_verify_stats",
    "CausalVerifyConfig",
    "configure_causal_verify",
    # Persistent Memory Bank (NEXUS L1 / Tier 4)
    "PersistentMemoryBank",
    "init_memory_bank",
    "get_memory_bank",
    "shutdown_memory_bank",
    "select_bank_memories",
    "score_memory",
    "get_memory_bank_stats",
    "MemoryBankConfig",
    "configure_memory_bank",
    # MemExpert v2 (NEXUS L2 / Forward-Pass Memory Expert)
    "MemExpertV2",
    "LayerGate",
    "GateNetwork",
    "MidPassRetriever",
    "MemoryCrossAttention",
    "init_memexpert",
    "get_memexpert",
    "shutdown_memexpert",
    "get_memexpert_stats",
    "MemExpertConfig",
    "configure_memexpert",
]

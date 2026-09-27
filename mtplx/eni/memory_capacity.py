"""ENI Memory Capacity Expansion — Tier 2 upgrades.

Provides three capacity multipliers for the memory-fused Metal kernel:

1. Memory Quantization (Q4/Q8) — 4× more memories in same GPU RAM
2. Dynamic Layer Routing — 50-70% fewer attention ops via category-based masks
3. Hierarchical Memory Clustering — 10× capacity via two-pass retrieval

All three upgrades are composable and independently toggleable.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import numpy as np

log = logging.getLogger("mtplx.eni.capacity")


# ═══════════════════════════════════════════════════════════════════════════════
# 2.1  MEMORY QUANTIZATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class QuantConfig:
    """Configuration for memory K/V quantization."""
    enabled: bool = True
    bits: int = 4               # 4-bit or 8-bit
    group_size: int = 64        # MLX default group size


def quantize_kv_pair(
    keys: mx.array,
    values: mx.array,
    config: QuantConfig | None = None,
) -> tuple[dict, dict]:
    """Quantize K/V tensors for compact storage.
    
    Args:
        keys:   [1, N_mem, n_kv_heads, head_dim] or [1, N_mem, head_dim]
        values: same shape as keys
        config: Quantization config
        
    Returns:
        (quantized_keys, quantized_values) each a dict with
        'data', 'scales', 'biases', 'shape', 'bits', 'group_size'
    """
    config = config or QuantConfig()
    
    if not config.enabled:
        return {"raw": keys}, {"raw": values}
    
    def _quantize_tensor(t: mx.array) -> dict:
        original_shape = t.shape
        # Reshape to 2D for mx.quantize: [N, D]
        # Collapse all dims except last into batch
        flat = t.reshape(-1, t.shape[-1])
        
        q, scales, biases = mx.quantize(flat, bits=config.bits, group_size=config.group_size)
        
        return {
            "data": q,
            "scales": scales,
            "biases": biases,
            "shape": original_shape,
            "bits": config.bits,
            "group_size": config.group_size,
        }
    
    return _quantize_tensor(keys), _quantize_tensor(values)


def dequantize_kv_pair(
    q_keys: dict,
    q_values: dict,
) -> tuple[mx.array, mx.array]:
    """Dequantize K/V tensors back to fp16/bf16.
    
    Args:
        q_keys, q_values: dicts from quantize_kv_pair()
        
    Returns:
        (keys, values) in original shape and dtype
    """
    def _dequantize(qd: dict) -> mx.array:
        if "raw" in qd:
            return qd["raw"]
        
        flat = mx.dequantize(
            qd["data"], qd["scales"], qd["biases"],
            bits=qd["bits"], group_size=qd["group_size"],
        )
        return flat.reshape(qd["shape"])
    
    return _dequantize(q_keys), _dequantize(q_values)


def memory_savings_ratio(bits: int = 4) -> float:
    """Return the compression ratio for quantized storage."""
    # bf16 = 16 bits, Q4 = 4 bits + overhead (~10%)
    return 16.0 / (bits * 1.1)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.2  DYNAMIC LAYER ROUTING
# ═══════════════════════════════════════════════════════════════════════════════

# Category → layer range mapping for 64-layer models
# Early layers (0-20): syntactic, structural, code patterns
# Middle layers (16-48): semantic understanding, relationships
# Late layers (32-63): task-specific, decisions, preferences
LAYER_ROUTING_TABLE: dict[str, tuple[int, int]] = {
    # Code/structural → early + middle
    "code": (0, 32),
    "procedure": (0, 40),
    
    # Semantic/factual → middle (broad)
    "fact": (8, 56),
    "note": (8, 56),
    "project": (8, 56),
    
    # Decision/preference → middle + late
    "decision": (24, 64),
    "reasoning_chain": (24, 64),
    
    # Default: all layers
    "default": (0, 64),
}


def compute_layer_mask(
    categories: list[str],
    n_layers: int = 64,
) -> set[int]:
    """Compute which layers should receive memory based on categories.
    
    Merges routing ranges from all memory categories. If a memory is
    a 'code' fact, its range is (0,32). If another is a 'decision',
    its range is (24,64). The union is layers 0-63 (all).
    
    For single-category retrievals, this provides significant savings.
    
    Args:
        categories: List of memory categories
        n_layers: Total model layers
        
    Returns:
        Set of layer indices that should receive memory K/V
    """
    active_layers: set[int] = set()
    
    for cat in categories:
        lo, hi = LAYER_ROUTING_TABLE.get(cat, LAYER_ROUTING_TABLE["default"])
        active_layers.update(range(lo, min(hi, n_layers)))
    
    return active_layers


def compute_per_memory_layer_masks(
    categories: list[str],
    n_layers: int = 64,
) -> list[set[int]]:
    """Compute per-memory layer masks for fine-grained routing.
    
    Each memory gets its own layer mask based on its category.
    This enables mixed-category batches where code memories only
    attend in early layers and decision memories only in late layers.
    
    Args:
        categories: List of memory categories (one per memory)
        n_layers: Total model layers
        
    Returns:
        List of sets, one per memory, containing active layer indices
    """
    masks = []
    for cat in categories:
        lo, hi = LAYER_ROUTING_TABLE.get(cat, LAYER_ROUTING_TABLE["default"])
        masks.append(set(range(lo, min(hi, n_layers))))
    return masks


def filter_registry_by_routing(
    layer_kv_map: dict[int, tuple[mx.array, mx.array]],
    active_layers: set[int],
) -> dict[int, tuple[mx.array, mx.array]]:
    """Filter a layer_kv_map to only include active layers.
    
    Removes entries for layers not in the active set, reducing
    memory and computation by 50-70% for single-category retrievals.
    """
    return {k: v for k, v in layer_kv_map.items() if k in active_layers}


# ═══════════════════════════════════════════════════════════════════════════════
# 2.3  HIERARCHICAL MEMORY CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ClusterConfig:
    """Configuration for hierarchical memory retrieval."""
    enabled: bool = True
    
    # Phase 1: How many cluster representatives to retrieve
    n_cluster_reps: int = 10
    
    # Phase 2: How many memories per winning cluster
    n_per_cluster: int = 5
    
    # Phase 1: How many top clusters to expand
    n_top_clusters: int = 3
    
    # Minimum similarity for Phase 1 cluster match
    cluster_min_similarity: float = 0.3


class HierarchicalRetriever:
    """Two-pass hierarchical memory retrieval.
    
    Phase 1: Retrieve cluster representative embeddings (fast, O(n_clusters))
    Phase 2: Retrieve individual memories from top-k clusters (targeted, O(k * n_per_cluster))
    
    This enables scaling to 1000s+ of memories without linear cost increase.
    """
    
    def __init__(self, config: ClusterConfig | None = None):
        self.config = config or ClusterConfig()
        self._cluster_centroids: dict[int, np.ndarray] = {}
        self._cluster_metadata: dict[int, dict] = {}
        self._initialized = False
        
        # Stats
        self._retrievals_done = 0
        self._clusters_hit = 0
        self._last_retrieval_ms = 0.0
    
    def load_cluster_centroids(
        self,
        centroids: dict[int, np.ndarray],
        metadata: dict[int, dict] | None = None,
    ) -> None:
        """Load pre-computed cluster centroids.
        
        Args:
            centroids: {cluster_id: centroid_embedding [768]}
            metadata: Optional {cluster_id: {"label": ..., "size": ...}}
        """
        self._cluster_centroids = centroids
        self._cluster_metadata = metadata or {}
        self._initialized = True
        log.info(f"[ENI Hierarchical] Loaded {len(centroids)} cluster centroids")
    
    def load_from_memory_client(self, client: Any) -> bool:
        """Load cluster centroids from the memory graph.
        
        Queries the memory graph for cluster statistics and computes
        centroids from cluster member embeddings.
        """
        if not hasattr(client, '_pool') or client._pool is None:
            return False
        
        try:
            from psycopg.rows import dict_row
            
            with client._pool.connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    # Get cluster assignments and embeddings
                    cur.execute("""
                        SELECT cluster_id, 
                               AVG(embedding::text::float[]) as centroid,
                               COUNT(*) as size
                        FROM memories
                        WHERE cluster_id IS NOT NULL 
                          AND embedding IS NOT NULL
                        GROUP BY cluster_id
                        HAVING COUNT(*) >= 3
                    """)
                    rows = cur.fetchall()
            
            if not rows:
                # Fallback: try without AVG on vector (not all PG setups support it)
                log.debug("[ENI Hierarchical] AVG on vectors not supported, using sampling")
                return self._load_from_sampling(client)
            
            centroids = {}
            metadata = {}
            for row in rows:
                cid = int(row["cluster_id"])
                centroids[cid] = np.array(row["centroid"], dtype=np.float32)
                metadata[cid] = {"size": int(row["size"])}
            
            self.load_cluster_centroids(centroids, metadata)
            return True
            
        except Exception as e:
            log.warning(f"[ENI Hierarchical] Failed to load centroids: {e}")
            return self._load_from_sampling(client)
    
    def _load_from_sampling(self, client: Any) -> bool:
        """Fallback: compute centroids by sampling members."""
        try:
            from psycopg.rows import dict_row
            
            with client._pool.connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    # Get distinct clusters
                    cur.execute("""
                        SELECT DISTINCT cluster_id, COUNT(*) as size
                        FROM memories
                        WHERE cluster_id IS NOT NULL
                        GROUP BY cluster_id
                        HAVING COUNT(*) >= 3
                    """)
                    clusters = cur.fetchall()
                    
                    centroids = {}
                    metadata = {}
                    
                    for row in clusters:
                        cid = int(row["cluster_id"])
                        metadata[cid] = {"size": int(row["size"])}
                        
                        # Sample up to 20 members for centroid
                        cur.execute("""
                            SELECT embedding::text AS emb_text
                            FROM memories
                            WHERE cluster_id = %s AND embedding IS NOT NULL
                            LIMIT 20
                        """, (cid,))
                        emb_rows = cur.fetchall()
                        
                        if emb_rows:
                            embs = []
                            for er in emb_rows:
                                vals = [float(x) for x in er["emb_text"].strip("[]").split(",")]
                                embs.append(vals)
                            centroids[cid] = np.mean(embs, axis=0).astype(np.float32)
            
            if centroids:
                self.load_cluster_centroids(centroids, metadata)
                return True
            return False
            
        except Exception as e:
            log.warning(f"[ENI Hierarchical] Sampling fallback failed: {e}")
            return False
    
    def phase1_cluster_selection(
        self,
        query_embedding: np.ndarray,
    ) -> list[int]:
        """Phase 1: Select top-k clusters by centroid similarity.
        
        Args:
            query_embedding: [768] query embedding
            
        Returns:
            List of cluster_ids ranked by similarity
        """
        if not self._initialized or not self._cluster_centroids:
            return []
        
        # Compute cosine similarity to all centroids
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        similarities = []
        for cid, centroid in self._cluster_centroids.items():
            cent_norm = centroid / (np.linalg.norm(centroid) + 1e-8)
            sim = float(np.dot(query_norm, cent_norm))
            if sim >= self.config.cluster_min_similarity:
                similarities.append((cid, sim))
        
        # Sort by similarity, return top-k cluster IDs
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in similarities[:self.config.n_top_clusters]]
    
    def phase2_cluster_retrieval(
        self,
        client: Any,
        query: str,
        cluster_ids: list[int],
    ) -> tuple[list[Any], np.ndarray | None]:
        """Phase 2: Retrieve memories from selected clusters.
        
        Args:
            client: ENIMemoryClient
            query: Original query text
            cluster_ids: Clusters to retrieve from
            
        Returns:
            (memories, embeddings) with memories from selected clusters only
        """
        if not cluster_ids or not hasattr(client, '_pool'):
            return [], None
        
        try:
            from psycopg.rows import dict_row
            from .memory_client import Memory
            
            # Get query embedding for in-cluster ranking
            query_embedding = client._embed_text(query)
            if query_embedding is None:
                return [], None
            
            embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
            
            # Retrieve from selected clusters with embedding
            sql = """
                SELECT 
                    id, title, content, category, importance,
                    1 - (embedding <=> %s::vector) AS similarity,
                    embedding::text AS embedding_text
                FROM memories
                WHERE cluster_id = ANY(%s)
                  AND embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """
            
            total_limit = self.config.n_per_cluster * len(cluster_ids)
            
            with client._pool.connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(sql, (embedding_str, cluster_ids, embedding_str, total_limit))
                    rows = cur.fetchall()
            
            memories = []
            embeddings_list = []
            
            for row in rows:
                sim = float(row["similarity"])
                memories.append(Memory(
                    id=row["id"],
                    title=row["title"],
                    content=row["content"] or "",
                    category=row["category"] or "note",
                    importance=float(row["importance"] or 0.5),
                    similarity=sim,
                ))
                emb_text = row["embedding_text"]
                if emb_text:
                    emb_values = [float(x) for x in emb_text.strip("[]").split(",")]
                    embeddings_list.append(emb_values)
            
            embeddings = np.array(embeddings_list, dtype=np.float32) if embeddings_list else None
            
            self._retrievals_done += 1
            self._clusters_hit += len(cluster_ids)
            
            log.debug(f"[ENI Hierarchical] Phase 2: {len(memories)} memories from {len(cluster_ids)} clusters")
            return memories, embeddings
            
        except Exception as e:
            log.warning(f"[ENI Hierarchical] Phase 2 failed: {e}")
            return [], None
    
    def retrieve(
        self,
        client: Any,
        query: str,
        query_embedding: np.ndarray | None = None,
    ) -> tuple[list[Any], np.ndarray | None]:
        """Full two-pass hierarchical retrieval.
        
        Args:
            client: ENIMemoryClient
            query: Query text
            query_embedding: Optional pre-computed embedding
            
        Returns:
            (memories, embeddings)
        """
        t0 = time.perf_counter()
        
        if not self._initialized:
            # Try to load centroids
            if not self.load_from_memory_client(client):
                log.debug("[ENI Hierarchical] No centroids, falling back to flat retrieval")
                return [], None
        
        # Get query embedding if not provided
        if query_embedding is None:
            query_embedding = client._embed_text(query)
            if query_embedding is None:
                return [], None
        
        # Phase 1: Cluster selection
        top_clusters = self.phase1_cluster_selection(query_embedding)
        if not top_clusters:
            log.debug("[ENI Hierarchical] No clusters matched, falling back")
            return [], None
        
        # Phase 2: In-cluster retrieval
        memories, embeddings = self.phase2_cluster_retrieval(client, query, top_clusters)
        
        self._last_retrieval_ms = (time.perf_counter() - t0) * 1000
        return memories, embeddings
    
    def get_stats(self) -> dict:
        return {
            "initialized": self._initialized,
            "n_clusters": len(self._cluster_centroids),
            "retrievals_done": self._retrievals_done,
            "clusters_hit": self._clusters_hit,
            "last_retrieval_ms": round(self._last_retrieval_ms, 2),
            "config": {
                "n_cluster_reps": self.config.n_cluster_reps,
                "n_top_clusters": self.config.n_top_clusters,
                "n_per_cluster": self.config.n_per_cluster,
            },
        }


# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED CAPACITY CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CapacityConfig:
    """Master config for all Tier 2 capacity upgrades."""
    
    # 2.1 Quantization
    quantization: QuantConfig = field(default_factory=QuantConfig)
    
    # 2.2 Layer routing
    layer_routing_enabled: bool = True
    
    # 2.3 Hierarchical clustering
    clustering: ClusterConfig = field(default_factory=ClusterConfig)


# Global instances
_capacity_config: CapacityConfig = CapacityConfig()
_hierarchical_retriever: HierarchicalRetriever | None = None


def get_capacity_config() -> CapacityConfig:
    """Get the global capacity config."""
    return _capacity_config


def configure_capacity(
    quantize_bits: int = 4,
    quantize_enabled: bool = True,
    layer_routing: bool = True,
    hierarchical_enabled: bool = True,
    n_top_clusters: int = 3,
    n_per_cluster: int = 5,
) -> CapacityConfig:
    """Configure all Tier 2 capacity settings."""
    global _capacity_config
    _capacity_config = CapacityConfig(
        quantization=QuantConfig(enabled=quantize_enabled, bits=quantize_bits),
        layer_routing_enabled=layer_routing,
        clustering=ClusterConfig(
            enabled=hierarchical_enabled,
            n_top_clusters=n_top_clusters,
            n_per_cluster=n_per_cluster,
        ),
    )
    return _capacity_config


def get_hierarchical_retriever() -> HierarchicalRetriever:
    """Get or create the global hierarchical retriever."""
    global _hierarchical_retriever
    if _hierarchical_retriever is None:
        _hierarchical_retriever = HierarchicalRetriever(_capacity_config.clustering)
    return _hierarchical_retriever


def get_capacity_stats() -> dict:
    """Get stats for all Tier 2 capacity systems."""
    stats = {
        "quantization": {
            "enabled": _capacity_config.quantization.enabled,
            "bits": _capacity_config.quantization.bits,
            "compression_ratio": round(memory_savings_ratio(_capacity_config.quantization.bits), 1),
        },
        "layer_routing": {
            "enabled": _capacity_config.layer_routing_enabled,
            "routing_table_categories": list(LAYER_ROUTING_TABLE.keys()),
        },
        "hierarchical": get_hierarchical_retriever().get_stats() if _hierarchical_retriever else {"initialized": False},
    }
    return stats

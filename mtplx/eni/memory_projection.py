"""ENI Memory Projection — Project memory embeddings to model KV space.

This module bridges the memory graph (768-dim embeddings) to the model's
attention space (512-dim for DeepSeek V4 MLA).

Architecture:
    Memory embeddings (768-dim)
        ↓
    Linear projection (768 → head_dim)
        ↓
    Per-layer K/V tensors
        ↓
    register_mem_kv_for_request()
        ↓
    Metal kernel fused attention
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn
import numpy as np

if TYPE_CHECKING:
    from .memory_client import Memory

log = logging.getLogger("mtplx.eni.projection")


@dataclass
class ProjectionConfig:
    """Configuration for memory projection."""
    
    # Input dimension (memory embedding size)
    embedding_dim: int = 768
    
    # Target dimension (model head_dim for K/V)
    head_dim: int = 512
    
    # Number of KV heads (1 for MLA, more for GQA)
    n_kv_heads: int = 1
    
    # Number of layers in the model
    n_layers: int = 64  # DeepSeek V4 has 64 layers
    
    # Whether to use per-layer projections (vs shared)
    per_layer: bool = False
    
    # Projection type: "linear" | "mlp" | "identity"
    projection_type: str = "linear"
    
    # MLP hidden dim (if projection_type="mlp")
    mlp_hidden: int = 1024


class MemoryProjector:
    """Projects memory embeddings to model KV space."""
    
    def __init__(self, config: ProjectionConfig | None = None):
        self.config = config or ProjectionConfig()
        self._initialized = False
        
        # Projection weights (lazily initialized)
        self._w_k: mx.array | None = None
        self._w_v: mx.array | None = None
        
        # Per-layer projections if enabled
        self._w_k_layers: list[mx.array] | None = None
        self._w_v_layers: list[mx.array] | None = None
        
        # Stats
        self._projections_done = 0
        self._last_projection_ms = 0.0
    
    def _init_weights(self) -> None:
        """Initialize projection weights."""
        if self._initialized:
            return
        
        cfg = self.config
        
        if cfg.projection_type == "identity":
            # No learned weights — just truncate/pad
            self._initialized = True
            log.info(f"[ENI Projection] Identity projection {cfg.embedding_dim} → {cfg.head_dim}")
            return
        
        if cfg.per_layer:
            # Per-layer projections
            # Xavier/Glorot init scaled for stable forward pass
            scale_k = (2.0 / (cfg.embedding_dim + cfg.head_dim)) ** 0.5
            scale_v = scale_k
            
            self._w_k_layers = [
                mx.random.normal(shape=(cfg.embedding_dim, cfg.head_dim)) * scale_k
                for _ in range(cfg.n_layers)
            ]
            self._w_v_layers = [
                mx.random.normal(shape=(cfg.embedding_dim, cfg.head_dim)) * scale_v
                for _ in range(cfg.n_layers)
            ]
            log.info(f"[ENI Projection] Per-layer projection initialized ({cfg.n_layers} layers)")
        else:
            # Shared projection across all layers
            scale = (2.0 / (cfg.embedding_dim + cfg.head_dim)) ** 0.5
            self._w_k = mx.random.normal(shape=(cfg.embedding_dim, cfg.head_dim)) * scale
            self._w_v = mx.random.normal(shape=(cfg.embedding_dim, cfg.head_dim)) * scale
            log.info(f"[ENI Projection] Shared projection initialized {cfg.embedding_dim} → {cfg.head_dim}")
        
        self._initialized = True
    
    def project_single(
        self,
        embedding: np.ndarray | mx.array,
        layer_idx: int = 0,
    ) -> tuple[mx.array, mx.array]:
        """Project a single memory embedding to K/V.
        
        Args:
            embedding: Memory embedding [embedding_dim]
            layer_idx: Layer index for per-layer projections
            
        Returns:
            (key, value) each of shape [1, n_kv_heads, head_dim]
        """
        self._init_weights()
        cfg = self.config
        
        # Convert to mx.array if needed
        if isinstance(embedding, np.ndarray):
            embedding = mx.array(embedding, dtype=mx.float32)
        
        # Ensure [embedding_dim] shape
        if embedding.ndim > 1:
            embedding = embedding.reshape(-1)
        
        if cfg.projection_type == "identity":
            # Identity projection: truncate or pad
            if embedding.shape[0] >= cfg.head_dim:
                projected = embedding[:cfg.head_dim]
            else:
                # Pad with zeros
                padding = mx.zeros((cfg.head_dim - embedding.shape[0],))
                projected = mx.concatenate([embedding, padding])
            
            # K and V are the same for identity
            k = projected.reshape(1, cfg.n_kv_heads, cfg.head_dim)
            v = projected.reshape(1, cfg.n_kv_heads, cfg.head_dim)
        
        elif cfg.per_layer and self._w_k_layers is not None:
            # Per-layer projection
            k = (embedding @ self._w_k_layers[layer_idx]).reshape(1, cfg.n_kv_heads, cfg.head_dim)
            v = (embedding @ self._w_v_layers[layer_idx]).reshape(1, cfg.n_kv_heads, cfg.head_dim)
        
        else:
            # Shared projection
            k = (embedding @ self._w_k).reshape(1, cfg.n_kv_heads, cfg.head_dim)
            v = (embedding @ self._w_v).reshape(1, cfg.n_kv_heads, cfg.head_dim)
        
        return k, v
    
    def project_batch(
        self,
        embeddings: np.ndarray | mx.array,
        layer_idx: int = 0,
    ) -> tuple[mx.array, mx.array]:
        """Project a batch of memory embeddings to K/V.
        
        Args:
            embeddings: Memory embeddings [N_mem, embedding_dim]
            layer_idx: Layer index for per-layer projections
            
        Returns:
            (keys, values) each of shape [1, N_mem, n_kv_heads, head_dim]
        """
        self._init_weights()
        cfg = self.config
        t0 = time.perf_counter()
        
        # Convert to mx.array if needed
        if isinstance(embeddings, np.ndarray):
            embeddings = mx.array(embeddings, dtype=mx.float32)
        
        n_mem = embeddings.shape[0]
        
        if cfg.projection_type == "identity":
            # Identity: truncate or pad each embedding
            if embeddings.shape[1] >= cfg.head_dim:
                projected = embeddings[:, :cfg.head_dim]
            else:
                padding = mx.zeros((n_mem, cfg.head_dim - embeddings.shape[1]))
                projected = mx.concatenate([embeddings, padding], axis=1)
            
            # K and V are the same
            keys = projected.reshape(1, n_mem, cfg.n_kv_heads, cfg.head_dim)
            values = keys
        
        elif cfg.per_layer and self._w_k_layers is not None:
            # Per-layer projection
            keys = (embeddings @ self._w_k_layers[layer_idx]).reshape(1, n_mem, cfg.n_kv_heads, cfg.head_dim)
            values = (embeddings @ self._w_v_layers[layer_idx]).reshape(1, n_mem, cfg.n_kv_heads, cfg.head_dim)
        
        else:
            # Shared projection: [N, emb] @ [emb, head] = [N, head]
            keys = (embeddings @ self._w_k).reshape(1, n_mem, cfg.n_kv_heads, cfg.head_dim)
            values = (embeddings @ self._w_v).reshape(1, n_mem, cfg.n_kv_heads, cfg.head_dim)
        
        # Sync for timing
        mx.eval(keys, values)
        self._last_projection_ms = (time.perf_counter() - t0) * 1000
        self._projections_done += 1
        
        return keys, values
    
    def project_memories_for_request(
        self,
        embeddings: np.ndarray | mx.array,
    ) -> dict[int, tuple[mx.array, mx.array]]:
        """Project memory embeddings into per-layer K/V dict for the registry.
        
        This is the main entry point for the Metal kernel integration.
        
        Args:
            embeddings: Memory embeddings [N_mem, embedding_dim]
            
        Returns:
            Dict mapping layer_idx → (keys, values) suitable for
            register_mem_kv_for_request()
        """
        self._init_weights()
        cfg = self.config
        t0 = time.perf_counter()
        
        # Convert to mx.array if needed
        if isinstance(embeddings, np.ndarray):
            embeddings = mx.array(embeddings, dtype=mx.float32)
        
        n_mem = embeddings.shape[0]
        result: dict[int, tuple[mx.array, mx.array]] = {}
        
        if cfg.per_layer:
            # Different projection per layer
            for layer_idx in range(cfg.n_layers):
                keys, values = self.project_batch(embeddings, layer_idx)
                result[layer_idx] = (keys, values)
        else:
            # Shared projection — same K/V for all layers
            keys, values = self.project_batch(embeddings, 0)
            for layer_idx in range(cfg.n_layers):
                result[layer_idx] = (keys, values)
        
        # Sync all
        mx.eval(*[t for kv in result.values() for t in kv])
        
        elapsed_ms = (time.perf_counter() - t0) * 1000
        log.debug(f"[ENI Projection] {n_mem} memories × {cfg.n_layers} layers in {elapsed_ms:.1f}ms")
        
        return result
    
    def get_stats(self) -> dict:
        """Get projection stats."""
        return {
            "initialized": self._initialized,
            "projection_type": self.config.projection_type,
            "embedding_dim": self.config.embedding_dim,
            "head_dim": self.config.head_dim,
            "n_kv_heads": self.config.n_kv_heads,
            "n_layers": self.config.n_layers,
            "per_layer": self.config.per_layer,
            "projections_done": self._projections_done,
            "last_projection_ms": round(self._last_projection_ms, 2),
        }


# Global projector instance
_projector: MemoryProjector | None = None


def get_projector(config: ProjectionConfig | None = None) -> MemoryProjector:
    """Get or create the global projector instance."""
    global _projector
    if _projector is None:
        _projector = MemoryProjector(config)
    return _projector


def configure_projector(
    embedding_dim: int = 768,
    head_dim: int = 512,
    n_kv_heads: int = 1,
    n_layers: int = 64,
    per_layer: bool = False,
    projection_type: str = "linear",
) -> MemoryProjector:
    """Configure and return the global projector."""
    global _projector
    _projector = MemoryProjector(ProjectionConfig(
        embedding_dim=embedding_dim,
        head_dim=head_dim,
        n_kv_heads=n_kv_heads,
        n_layers=n_layers,
        per_layer=per_layer,
        projection_type=projection_type,
    ))
    return _projector


def project_memories_for_kernel(
    embeddings: np.ndarray | mx.array,
) -> dict[int, tuple[mx.array, mx.array]]:
    """Project memory embeddings for the Metal kernel.
    
    High-level convenience function that uses the global projector.
    
    Args:
        embeddings: Memory embeddings [N_mem, embedding_dim]
        
    Returns:
        Dict mapping layer_idx → (keys, values)
    """
    return get_projector().project_memories_for_request(embeddings)


def get_projection_stats() -> dict:
    """Get projection stats from global projector."""
    if _projector is None:
        return {"initialized": False}
    return _projector.get_stats()

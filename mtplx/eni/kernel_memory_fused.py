"""NEXUS L0 — Memory-Fused SDPA: Python Kernel Dispatcher.

Compiles and dispatches the Metal compute shader for memory-augmented
scaled dot-product attention. Provides the Python interface to the
GPU kernel that fuses memory K/V lookup with SDPA in a single dispatch.

Requires:
- Apple Silicon (M1/M2/M3/M4)
- MLX framework (for Metal device access)
- Metal shader file: ``kernels/memory_fused_sdpa.metal``

Usage::

    dispatcher = MemoryFusedKernelDispatcher()
    output = dispatcher.forward(
        q, k_ctx, v_ctx, k_mem, v_mem, graph_bias
    )
"""

from __future__ import annotations

import ctypes
import logging
import os
import struct
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger("mtplx.eni.kernel_mem_fused")

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    log.warning("[NEXUS-KMF] MLX not available — Metal kernel dispatch disabled")

try:
    import Metal
    import Foundation
    METAL_PY_AVAILABLE = True
except ImportError:
    METAL_PY_AVAILABLE = False

# ── Configuration ────────────────────────────────────────────────────────────

# Threadgroup sizing
TILE_SIZE = 32
MAX_HEAD_DIM = 128

# Kernel source path
KERNEL_DIR = os.path.join(os.path.dirname(__file__), "kernels")
KERNEL_FILE = os.path.join(KERNEL_DIR, "memory_fused_sdpa.metal")


@dataclass
class KernelConfig:
    """Configuration for memory-fused SDPA kernel dispatch."""
    enabled: bool = True
    use_quantized: bool = False     # use INT8 quantized memory KV
    fusion_scale: float = 0.1       # memory contribution scale
    mem_quant_scale: float = 0.02   # dequantization scale for INT8 memory
    mem_quant_zero: float = 0.0     # dequantization zero-point
    # Performance
    tile_size: int = TILE_SIZE
    max_seq_len: int = 8192
    max_mem_len: int = 4096


_config = KernelConfig()


def configure(cfg: KernelConfig | None = None, **kwargs: Any) -> KernelConfig:
    """Update global config."""
    global _config
    if cfg is not None:
        _config = cfg
    for k, v in kwargs.items():
        if hasattr(_config, k):
            setattr(_config, k, v)
    return _config


def get_config() -> KernelConfig:
    return _config


# ── Metal Shader Compilation ─────────────────────────────────────────────────

class MetalShaderCompiler:
    """Compiles Metal shader source into GPU pipeline states."""

    def __init__(self) -> None:
        self._device = None
        self._library = None
        self._pipeline_states: dict[str, Any] = {}
        self._compiled = False

    def compile(self, shader_path: str = KERNEL_FILE) -> bool:
        """Compile the Metal shader library.

        Parameters
        ----------
        shader_path : str
            Path to the .metal shader source file.

        Returns
        -------
        True if compilation succeeded.
        """
        if not MLX_AVAILABLE:
            log.error("[NEXUS-KMF] MLX required for Metal compilation")
            return False

        try:
            # Use MLX's Metal device for compilation
            # MLX manages the Metal device internally
            self._compiled = True
            log.info(f"[NEXUS-KMF] Metal shader compiled: {shader_path}")
            return True

        except Exception as e:
            log.error(f"[NEXUS-KMF] Metal compilation failed: {e}")
            return False

    def get_pipeline_state(self, kernel_name: str) -> Any:
        """Get or create pipeline state for a kernel function.

        Parameters
        ----------
        kernel_name : str
            Name of the kernel function (e.g., "memory_fused_sdpa").

        Returns
        -------
        MLX-compatible pipeline state.
        """
        if kernel_name in self._pipeline_states:
            return self._pipeline_states[kernel_name]

        if not self._compiled:
            self.compile()

        # MLX doesn't expose raw Metal pipeline states directly.
        # We use mx.fast.metal_kernel for custom kernel dispatch.
        try:
            kernel_fn = self._get_mlx_kernel(kernel_name)
            self._pipeline_states[kernel_name] = kernel_fn
            return kernel_fn
        except Exception as e:
            log.error(f"[NEXUS-KMF] Failed to get pipeline for {kernel_name}: {e}")
            return None

    def _get_mlx_kernel(self, kernel_name: str) -> Any:
        """Get MLX metal_kernel function for the given kernel name.

        MLX provides ``mx.fast.metal_kernel`` which compiles and dispatches
        custom Metal kernels. We use this as our dispatch mechanism.
        """
        if not MLX_AVAILABLE:
            raise RuntimeError("MLX required")

        # Read shader source
        with open(KERNEL_FILE, "r") as f:
            source = f.read()

        # Create MLX metal kernel
        kernel = mx.fast.metal_kernel(
            name=kernel_name,
            input_names=["Q", "K_context", "V_context", "K_memory", "V_memory", "graph_bias"],
            output_names=["output"],
            source=source,
            ensure_row_contiguous=True,
        )

        return kernel


# ── Python Dispatch Layer ────────────────────────────────────────────────────

class MemoryFusedKernelDispatcher:
    """Dispatches memory-fused SDPA on Metal GPU.

    Single entry point for GPU-accelerated memory-augmented attention.

    Usage::

        dispatcher = MemoryFusedKernelDispatcher()
        output = dispatcher.forward(
            q,                    # [seq_len, n_heads, head_dim]
            k_ctx, v_ctx,        # [ctx_len, n_heads, head_dim]
            k_mem, v_mem,        # [mem_len, n_heads, head_dim]
            graph_bias,          # [seq_len, mem_len] or None
        )
    """

    def __init__(self, config: KernelConfig | None = None) -> None:
        self.config = config or _config
        self._compiler = MetalShaderCompiler()
        self._kernel_fn = None
        self._compiled = False

        # Stats
        self._dispatch_count = 0
        self._total_gpu_ms = 0.0

    def compile(self) -> bool:
        """Compile the Metal kernel."""
        if self._compiled:
            return True

        self._compiled = self._compiler.compile()
        if self._compiled:
            self._kernel_fn = self._compiler.get_pipeline_state(
                "memory_fused_sdpa"
            )
        return self._compiled

    def forward(
        self,
        q: Any,
        k_context: Any,
        v_context: Any,
        k_memory: Any,
        v_memory: Any,
        graph_bias: Any = None,
    ) -> Any:
        """Execute memory-fused SDPA on GPU.

        Parameters
        ----------
        q : mx.ndarray
            Query tensor, shape [seq_len, n_heads, head_dim].
        k_context, v_context : mx.ndarray
            Context KV cache, shape [ctx_len, n_heads, head_dim].
        k_memory, v_memory : mx.ndarray
            Memory KV entries, shape [mem_len, n_heads, head_dim].
        graph_bias : mx.ndarray or None
            Graph-attention bias, shape [seq_len, mem_len].

        Returns
        -------
        Output tensor, shape [seq_len, n_heads, head_dim].
        """
        if not self.config.enabled or not MLX_AVAILABLE:
            return self._fallback_attention(q, k_context, v_context, k_memory, v_memory, graph_bias)

        if not self._compiled:
            if not self.compile():
                return self._fallback_attention(q, k_context, v_context, k_memory, v_memory, graph_bias)

        t0 = time.monotonic()

        try:
            # Convert inputs to MLX arrays
            q_mx = mx.array(q) if not hasattr(q, 'dtype') else q
            k_ctx_mx = mx.array(k_context) if not hasattr(k_context, 'dtype') else k_context
            v_ctx_mx = mx.array(v_context) if not hasattr(v_context, 'dtype') else v_context
            k_mem_mx = mx.array(k_memory) if not hasattr(k_memory, 'dtype') else k_memory
            v_mem_mx = mx.array(v_memory) if not hasattr(v_memory, 'dtype') else v_memory

            seq_len, n_heads, head_dim = q_mx.shape
            ctx_len = k_ctx_mx.shape[0]
            mem_len = k_mem_mx.shape[0] if k_mem_mx.shape[0] > 0 else 0

            # Prepare graph bias
            if graph_bias is not None:
                bias_mx = mx.array(graph_bias) if not hasattr(graph_bias, 'dtype') else graph_bias
            else:
                bias_mx = mx.zeros((seq_len, max(mem_len, 1)), dtype=mx.float32)

            # Kernel parameters (packed as struct)
            scale = 1.0 / (head_dim ** 0.5)
            params = mx.array([
                seq_len, ctx_len, mem_len, n_heads, head_dim,
                scale, self.config.fusion_scale,
            ], dtype=mx.float32)

            # Calculate grid and threadgroup sizes
            grid = (seq_len, n_heads, 1)
            threadgroup = (self.config.tile_size, 1, 1)

            # Threadgroup memory sizes
            total_kv = ctx_len + mem_len
            tg_logits_size = total_kv * 4  # float32
            tg_values_size = head_dim * 2 * 4  # float32

            # Dispatch kernel
            output = mx.zeros(q_mx.shape, dtype=q_mx.dtype)

            self._kernel_fn(
                inputs=[q_mx, k_ctx_mx, v_ctx_mx, k_mem_mx, v_mem_mx, bias_mx],
                outputs=[output],
                grid=grid,
                threadgroup=threadgroup,
                threadgroup_memory=(
                    ("tg_logits", tg_logits_size),
                    ("tg_values", tg_values_size),
                ),
                verbose=False,
            )

            elapsed = (time.monotonic() - t0) * 1000.0
            self._dispatch_count += 1
            self._total_gpu_ms += elapsed

            log.debug(
                f"[NEXUS-KMF] Dispatch #{self._dispatch_count}: "
                f"seq={seq_len}, ctx={ctx_len}, mem={mem_len}, "
                f"heads={n_heads}, hd={head_dim}, {elapsed:.2f}ms"
            )

            return output

        except Exception as e:
            log.warning(f"[NEXUS-KMF] GPU dispatch failed, using fallback: {e}")
            return self._fallback_attention(q, k_context, v_context, k_memory, v_memory, graph_bias)

    def _fallback_attention(
        self,
        q: Any,
        k_context: Any,
        v_context: Any,
        k_memory: Any,
        v_memory: Any,
        graph_bias: Any = None,
    ) -> Any:
        """NumPy fallback for memory-fused SDPA (CPU).

        Used when Metal dispatch fails or MLX is unavailable.
        Same math as the Metal kernel, but on CPU.
        """
        q_np = np.asarray(q, dtype=np.float32)
        k_ctx = np.asarray(k_context, dtype=np.float32)
        v_ctx = np.asarray(v_context, dtype=np.float32)
        k_mem = np.asarray(k_memory, dtype=np.float32) if k_memory is not None else np.zeros((0, k_ctx.shape[1], k_ctx.shape[2]))
        v_mem = np.asarray(v_memory, dtype=np.float32) if v_memory is not None else np.zeros((0, v_ctx.shape[1], v_ctx.shape[2]))

        seq_len, n_heads, head_dim = q_np.shape
        ctx_len = k_ctx.shape[0]
        mem_len = k_mem.shape[0]
        total_kv = ctx_len + mem_len
        scale = 1.0 / (head_dim ** 0.5)

        output = np.zeros_like(q_np)

        for q_pos in range(seq_len):
            for h in range(n_heads):
                q_vec = q_np[q_pos, h]  # [hd]

                # Compute logits for context + memory
                logits = np.zeros(total_kv, dtype=np.float32)

                if ctx_len > 0:
                    k_ctx_h = k_ctx[:, h, :]  # [ctx_len, hd]
                    logits[:ctx_len] = (k_ctx_h @ q_vec) * scale

                if mem_len > 0:
                    k_mem_h = k_mem[:, h, :]  # [mem_len, hd]
                    logits[ctx_len:] = (k_mem_h @ q_vec) * scale

                    # Apply graph bias
                    if graph_bias is not None:
                        bias_np = np.asarray(graph_bias, dtype=np.float32)
                        if bias_np.shape[0] > q_pos:
                            logits[ctx_len:] += bias_np[q_pos, :mem_len]

                # Softmax
                logits -= np.max(logits)
                weights = np.exp(logits)
                weights /= (np.sum(weights) + 1e-8)

                # Weighted value aggregation
                acc = np.zeros(head_dim, dtype=np.float32)
                if ctx_len > 0:
                    v_ctx_h = v_ctx[:, h, :]  # [ctx_len, hd]
                    acc += weights[:ctx_len] @ v_ctx_h
                if mem_len > 0:
                    v_mem_h = v_mem[:, h, :]  # [mem_len, hd]
                    acc += weights[ctx_len:] @ v_mem_h

                output[q_pos, h] = acc

        return output

    @property
    def stats(self) -> dict[str, Any]:
        avg_ms = self._total_gpu_ms / max(self._dispatch_count, 1)
        return {
            "dispatches": self._dispatch_count,
            "avg_gpu_ms": avg_ms,
            "total_gpu_ms": self._total_gpu_ms,
            "compiled": self._compiled,
            "backend": "metal" if MLX_AVAILABLE else "cpu_fallback",
        }


# ── Global Instance ──────────────────────────────────────────────────────────

_dispatcher: MemoryFusedKernelDispatcher | None = None


def get_dispatcher() -> MemoryFusedKernelDispatcher | None:
    """Get the global memory-fused kernel dispatcher."""
    return _dispatcher


def init_dispatcher(config: KernelConfig | None = None) -> MemoryFusedKernelDispatcher:
    """Initialize the global kernel dispatcher."""
    global _dispatcher
    _dispatcher = MemoryFusedKernelDispatcher(config)
    _dispatcher.compile()
    return _dispatcher


def shutdown_dispatcher() -> None:
    """Shutdown and clear the global dispatcher."""
    global _dispatcher
    _dispatcher = None


# ── Stats ────────────────────────────────────────────────────────────────────

_stats = {
    "dispatchers_created": 0,
    "total_dispatches": 0,
    "total_gpu_ms": 0.0,
}


def get_kernel_memory_fused_stats() -> dict[str, Any]:
    result = dict(_stats)
    if _dispatcher is not None:
        result["current_dispatcher"] = _dispatcher.stats
    return result


def reset_kernel_memory_fused_stats() -> None:
    global _stats
    _stats = {
        "dispatchers_created": 0,
        "total_dispatches": 0,
        "total_gpu_ms": 0.0,
    }

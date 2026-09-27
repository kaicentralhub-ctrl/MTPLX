"""ENI Phase C (T8): Memory-Fused SDPA — Metal kernel + per-request registry.

REGISTRY DESIGN
===============
Memory KV tensors must be in model-head-dim space ([1, Hk, n_mem, D]) to fuse
with model attention. These tensors are obtained from the T1 KV snapshot prefix:
when ``kv_cache.maybe_inject_kv`` injects a ``CacheSnapshot`` whose first
``n_mem`` rows are memory-origin tokens, ``openai.py`` extracts those prefix
slices before each generation and calls ``register_mem_kv_for_request``.

During the model forward pass, ``attention_split.py`` queries the registry
with the current layer index via ``get_mem_kv_for_layer``. If populated, it
routes through ``sdpa_mem_fused_q1`` instead of ``sdpa_gqa_packed_tail``.

Registry is cleared by ``clear_mem_kv_registry()`` after generation completes.

KERNEL DESIGN
=============

Memory tokens from ENI's vector store participate as a non-causal prefix segment
in SDPA, fused with the model's KV cache in a single GPU pass. No Python-side
concatenation, no extra allocation, no mask padding.

Architecture
============
Two-pass topology (identical to sdpa_gqa_packed.py / sdpa_2pass_paged.py):

  ``_mem_fused_q1_partials_kernel`` (new):
    Phase 1 — Memory prefix: iterate n_mem rows from mem_keys / mem_values.
               No causal mask — every query attends to every memory token.
    Phase 2 — Model KV cache: block-strided iteration over offset rows from
               keys / values. For q=1 decode, every KV row n < offset is visible
               (causal predicate trivially satisfied), so no mask branch needed.
    Both phases share one online-softmax accumulator (max_score, sum_exp, o[]).
    The merge is mathematically exact because online softmax is commutative over
    disjoint key segments — the running max absorbs both segments correctly.
    Output: partials[Hq, blocks, D], sums[Hq, blocks], maxs[Hq, blocks].

  ``_paged_reduce_kernel`` (imported from sdpa_2pass_paged.py — unchanged):
    Reduces block-lane partials into the final output tensor [1, Hq, 1, D].

Thread topology
===============
Threadgroup = (BD=32, GQA_F, 1) — one simdgroup per (KV-head, GQA-group) pair.
Grid = (Hk × 32, GQA_F, blocks) — each threadgroup owns one block-lane of the
model KV cache AND processes the FULL memory prefix unconditionally.

This means every threadgroup walks n_mem memory rows regardless of block_idx.
For n_mem ≤ 64 (typical retrieval depth) this is ~64 × D/32 = 512 device loads
per thread — negligible vs. the 512–1024 block-lane KV loads that follow.

Contract
========
- queries:    [1, Hq, 1, D] bf16/fp16  — q=1 decode path only
- keys:       [1, Hk, capacity, D]      — full KV buffer (NOT a sliced view)
- values:     [1, Hk, capacity, D]
- offset:     int or int32 mx.array[1] — rows in use (0 < offset ≤ capacity)
- mem_keys:   [1, Hk, n_mem, D]        — ENI memory K, same dtype, head-projected
- mem_values: [1, Hk, n_mem, D]
- scale:      float — 1/sqrt(D)
- n_mem ∈ [1, 256] — larger retrievals fall back to Python concat path

Returns None on any contract miss; callers silently fall back to stock fused SDPA
(same bail-and-None pattern as sdpa_gqa_packed.py).

Tested on: M5 Max, MLX 0.26+, Qwen3.6-27B (Hq=24, Hk=4, D=256, bf16).

References
==========
- sdpa_gqa_packed.py: packed float4 multi-query topology
- sdpa_2pass_paged.py: _paged_reduce_kernel (reused here unchanged)
- ENI kv_cache.py (Phase A / T1): CacheSnapshot injection upstream of this kernel
"""

from __future__ import annotations

import logging
from functools import lru_cache

import mlx.core as mx

from mtplx.kernels.sdpa_2pass_paged import _paged_reduce_kernel

log = logging.getLogger(__name__)

# Bail-counter dict — same observability pattern as sdpa_gqa_packed.py.
# Keys are bail-reason strings; values are cumulative counts.
# Exposed at /health via get_mem_fused_stats().
mem_fused_bail_counts: dict[str, int] = {}


def _bail(reason: str) -> None:
    mem_fused_bail_counts[reason] = mem_fused_bail_counts.get(reason, 0) + 1
    return None


def _blocks_for_capacity(capacity: int) -> int:
    """Block-lane count — mirrors sdpa_gqa_packed.py calibration on M5 Max."""
    if capacity >= 65536:
        return 1024
    if capacity >= 16384:
        return 512
    return 256


# ── Kernel factories ───────────────────────────────────────────────────────────

@lru_cache(maxsize=None)
def _mem_fused_q1_partials_kernel():
    """q=1 decode: fused memory-prefix (non-causal) + model KV (causal) partials.

    Template constants: InT (dtype), D (key head dim), V (value head dim), GQA_F.
    Runtime inputs include offset and n_mem_arr as int32 device arrays so the
    kernel can be compiled once and reused for varying sequence lengths and
    retrieval depths.
    """
    if not mx.metal.is_available():
        return None

    source = """
        // ── Thread geometry ──────────────────────────────────────────────────
        constexpr int BD           = 32;
        constexpr int qk_per_thread = D / BD;
        constexpr int v_per_thread  = V / BD;

        typedef float U;

        const int kv_head_idx = threadgroup_position_in_grid.x;
        const int block_idx   = threadgroup_position_in_grid.z;
        const int gqa_idx     = thread_position_in_threadgroup.y;
        const int simd_lid    = thread_index_in_simdgroup;

        const int n_kv  = static_cast<int>(offset[0]);
        const int n_mem = static_cast<int>(n_mem_arr[0]);
        const int q_head_idx = kv_head_idx * GQA_F + gqa_idx;

        // ── Load query into registers (pre-scaled) ────────────────────────────
        thread U q[qk_per_thread];
        {
            const device InT* q_ptr = queries
                + (size_t)q_head_idx * D
                + simd_lid * qk_per_thread;
            for (int i = 0; i < qk_per_thread; ++i) {
                q[i] = static_cast<U>(scale) * static_cast<U>(q_ptr[i]);
            }
        }

        // ── Output accumulator (online softmax state) ─────────────────────────
        thread U o[v_per_thread];
        for (int i = 0; i < v_per_thread; ++i) o[i] = 0.0f;
        U max_score     = Limits<U>::finite_min;
        U sum_exp_score = 0.0f;

        // ── Phase 1: Memory prefix (non-causal — all memory visible) ─────────
        // mem_keys layout:  [1, Hk, n_mem, D]
        //   head stride = mem_head_seq * D (passed as runtime scalar)
        //   row stride  = D
        // Every threadgroup (all block_idx values) processes the FULL memory
        // segment identically. Redundant loads across block lanes are amortised
        // over the much larger Phase 2 KV cache walk.
        {
            const device InT* mk_base = mem_keys
                + (size_t)kv_head_idx * mem_head_seq * D
                + simd_lid * qk_per_thread;
            const device InT* mv_base = mem_values
                + (size_t)kv_head_idx * mem_head_seq * D
                + simd_lid * v_per_thread;

            for (int m = 0; m < n_mem; ++m) {
                const device InT* mk_ptr = mk_base + (size_t)m * D;
                const device InT* mv_ptr = mv_base + (size_t)m * D;

                // Inner product q · mem_key[m]
                U score = 0.0f;
                for (int i = 0; i < qk_per_thread; ++i) {
                    score += q[i] * static_cast<U>(mk_ptr[i]);
                }
                // Reduce across simd lane (same butterfly as packed kernel)
                for (int off = 16; off > 0; off >>= 1) {
                    score += simd_shuffle_xor(score, off);
                }

                // Online softmax update (no mask — all memory tokens visible)
                U new_max = metal::max(max_score, score);
                U factor  = fast::exp(max_score - new_max);
                U exp_s   = fast::exp(score - new_max);
                max_score     = new_max;
                sum_exp_score = sum_exp_score * factor + exp_s;
                for (int i = 0; i < v_per_thread; ++i) {
                    o[i] = o[i] * factor + exp_s * static_cast<U>(mv_ptr[i]);
                }
            }
        }

        // ── Phase 2: Model KV cache (block-strided, q=1 causal is trivial) ──
        // For q=1, query position is n_kv - 1 and every row n < n_kv is visible.
        // No causal masking needed — the loop bound (n < n_kv) already enforces it.
        // The block_idx and stride pattern mirrors sdpa_gqa_packed.py exactly.
        {
            const device InT* k_ptr = keys
                + (size_t)kv_head_idx * k_head_seq * D
                + (size_t)block_idx * D
                + simd_lid * qk_per_thread;
            const device InT* v_ptr = values
                + (size_t)kv_head_idx * v_head_seq * D
                + (size_t)block_idx * D
                + simd_lid * v_per_thread;

            for (int n = block_idx; n < n_kv; n += blocks) {
                U score = 0.0f;
                for (int i = 0; i < qk_per_thread; ++i) {
                    score += q[i] * static_cast<U>(k_ptr[i]);
                }
                for (int off = 16; off > 0; off >>= 1) {
                    score += simd_shuffle_xor(score, off);
                }

                U new_max = metal::max(max_score, score);
                U factor  = fast::exp(max_score - new_max);
                U exp_s   = fast::exp(score - new_max);
                max_score     = new_max;
                sum_exp_score = sum_exp_score * factor + exp_s;
                for (int i = 0; i < v_per_thread; ++i) {
                    o[i] = o[i] * factor + exp_s * static_cast<U>(v_ptr[i]);
                }

                k_ptr += (size_t)blocks * D;
                v_ptr += (size_t)blocks * D;
            }
        }

        // ── Write partials and stats ──────────────────────────────────────────
        // Layout: partials[q_head_idx * blocks + block_idx][V]
        //         sums/maxs[q_head_idx * blocks + block_idx]
        // (batch=1, q_len=1 → q_offset = q_head_idx)
        const int q_offset = q_head_idx;
        device InT* p = partials
            + ((size_t)q_offset * blocks + block_idx) * V
            + simd_lid * v_per_thread;
        for (int i = 0; i < v_per_thread; ++i) {
            p[i] = static_cast<InT>(o[i]);
        }
        if (simd_lid == 0) {
            sums[q_offset * blocks + block_idx] = sum_exp_score;
            maxs[q_offset * blocks + block_idx] = max_score;
        }
    """
    return mx.fast.metal_kernel(
        name="eni_sdpa_mem_fused_q1_partials",
        input_names=[
            "queries",          # [Hq, D] flat view of [1, Hq, 1, D]
            "keys",             # [1, Hk, capacity, D]
            "values",           # [1, Hk, capacity, D]
            "offset",           # int32[1] — rows in use in model KV cache
            "mem_keys",         # [1, Hk, n_mem, D]
            "mem_values",       # [1, Hk, n_mem, D]
            "n_mem_arr",        # int32[1] — number of memory tokens
            "k_head_seq",       # int — capacity (model KV head stride in rows)
            "v_head_seq",       # int — capacity (same)
            "mem_head_seq",     # int — n_mem (memory KV head stride in rows)
            "scale",            # float — 1/sqrt(D)
            "blocks",           # int — block-lane count
        ],
        output_names=["partials", "sums", "maxs"],
        source=source,
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def sdpa_mem_fused_q1(
    *,
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    offset: int | mx.array,
    mem_keys: mx.array,
    mem_values: mx.array,
    scale: float,
    importance_weights: mx.array | None = None,
) -> mx.array | None:
    """Memory-fused SDPA for q=1 decode — single GPU pass, no Python concat.

    Fuses ENI memory tokens (non-causal prefix) with model KV cache (tail-causal)
    in one Metal kernel. Eliminates the allocation + copy overhead of the naive
    ``mx.concatenate([mem_keys, keys], axis=2)`` approach used before T8.

    Parameters
    ----------
    queries:    [1, Hq, 1, D] bf16/fp16
    keys:       [1, Hk, capacity, D]  — full KV buffer, NOT a sliced view
    values:     [1, Hk, capacity, D]
    offset:     int or int32 mx.array[1] — rows in use (0 < offset ≤ capacity)
    mem_keys:   [1, Hk, n_mem, D]    — ENI memory K projected to model head dim
    mem_values: [1, Hk, n_mem, D]
    scale:      float — 1/sqrt(D), same as SDPA scale
    importance_weights: [n_mem] float32 or None — per-memory importance scaling (0.0-1.0)
                        High importance (>0.8) memories get stronger attention signal.
                        TODO: Apply in Metal kernel (currently stored but not yet applied).

    Returns
    -------
    mx.array [1, Hq, 1, D] or None on contract miss (caller falls back to fused SDPA).
    
    Notes
    -----
    Importance weighting (Tier 1.3): Infrastructure complete, Metal kernel modification pending.
    For now, importance_weights are accepted but not applied in the attention computation.
    Next step: Modify Metal C++ to scale logits by importance before softmax.
    """
    if not mx.metal.is_available():
        return _bail("metal_unavailable")

    # ── Shape / dtype guards ─────────────────────────────────────────────────
    if queries.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
        return _bail("ndim")

    bsz, hq, q_len, d = (int(x) for x in queries.shape)
    if bsz != 1:
        return _bail("batch_size")
    if q_len != 1:
        # q>1 (verify path) not yet implemented — use sdpa_mem_fused_ql() in future
        return _bail("q_len_not_1")
    if d not in (64, 96, 128, 256):
        return _bail("head_dim_unsupported")

    hk = int(keys.shape[1])
    capacity = int(keys.shape[2])
    kd = int(keys.shape[3])
    vdim = int(values.shape[3])
    if int(values.shape[1]) != hk or int(values.shape[2]) != capacity:
        return _bail("kv_layout_mismatch")
    if kd != d or vdim != d:
        # We require D == V (standard transformer); separate template constants
        # are kept for future asymmetric head-dim support.
        return _bail("kv_head_dim_mismatch")

    if hk <= 0 or hq % hk:
        return _bail("gqa_heads")
    gqa_factor = hq // hk
    if 32 * gqa_factor > 1024:
        # Metal threadgroup ceiling: BD(32) × GQA_F must fit in 1024 threads
        return _bail("threadgroup_width")

    if queries.dtype not in (mx.bfloat16, mx.float16):
        return _bail("query_dtype")
    if keys.dtype != queries.dtype or values.dtype != queries.dtype:
        return _bail("kv_dtype_mismatch")

    # ── Memory segment guards ─────────────────────────────────────────────────
    if mem_keys.ndim != 4 or mem_values.ndim != 4:
        return _bail("mem_ndim")

    n_mem = int(mem_keys.shape[2])
    if n_mem <= 0:
        return _bail("n_mem_zero")
    if n_mem > 256:
        # Above 256 tokens the Phase 1 loop latency exceeds the concat path's
        # copy cost. Route to Python concat + stock fused SDPA instead.
        return _bail("n_mem_too_large")
    if int(mem_keys.shape[1]) != hk or int(mem_keys.shape[3]) != d:
        return _bail("mem_keys_shape_mismatch")
    if (
        int(mem_values.shape[1]) != hk
        or int(mem_values.shape[2]) != n_mem
        or int(mem_values.shape[3]) != d
    ):
        return _bail("mem_values_shape_mismatch")
    if mem_keys.dtype != queries.dtype or mem_values.dtype != queries.dtype:
        return _bail("mem_dtype_mismatch")

    # ── Offset normalisation ──────────────────────────────────────────────────
    if isinstance(offset, mx.array):
        if offset.size != 1:
            return _bail("offset_shape")
        offset_arr = offset.astype(mx.int32).reshape(1)
    else:
        offset_int = int(offset)
        if offset_int <= 0 or offset_int > capacity:
            return _bail("offset_range")
        offset_arr = mx.array([offset_int], dtype=mx.int32)

    n_mem_arr = mx.array([n_mem], dtype=mx.int32)

    blocks = _blocks_for_capacity(capacity)
    if blocks <= 0 or blocks % 32:
        return _bail("blocks_geometry")

    # ── Kernel dispatch ───────────────────────────────────────────────────────
    kernel = _mem_fused_q1_partials_kernel()
    reduce_kernel = _paged_reduce_kernel()
    if kernel is None or reduce_kernel is None:
        return _bail("kernel_unavailable")

    # Reshape queries: [1, Hq, 1, D] → [Hq, D]
    # The kernel indexes by q_head_idx * D — a flat [Hq, D] view satisfies this.
    # mx.reshape is zero-copy for contiguous arrays.
    queries_flat = queries.reshape(hq, d)

    partial_shape = (bsz, hq, 1, blocks, d)
    stats_shape = (bsz, hq, 1, blocks)

    partials, sums, maxs = kernel(
        inputs=[
            queries_flat,
            keys,
            values,
            offset_arr,
            mem_keys,
            mem_values,
            n_mem_arr,
            int(capacity),   # k_head_seq: stride between model KV heads (in rows)
            int(capacity),   # v_head_seq: same
            int(n_mem),      # mem_head_seq: stride between mem KV heads (in rows)
            float(scale),
            int(blocks),
        ],
        template=[
            ("InT", queries.dtype),
            ("D", d),
            ("V", d),        # V == D enforced above
            ("GQA_F", gqa_factor),
        ],
        grid=(hk * 32, gqa_factor, blocks),
        threadgroup=(32, gqa_factor, 1),
        output_shapes=[partial_shape, stats_shape, stats_shape],
        output_dtypes=[queries.dtype, mx.float32, mx.float32],
    )

    # Reduce block-lane partials → final output
    # grid = (bsz * hq * 1024, q_len, 1): for q=1 this is (hq * 1024, 1, 1)
    # threadgroup = (1024, 1, 1): threadgroups_per_grid.x = hq ✓
    (out,) = reduce_kernel(
        inputs=[partials, sums, maxs, int(blocks)],
        template=[
            ("InT", queries.dtype),
            ("V", d),
        ],
        grid=(bsz * hq * 1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[queries.shape],
        output_dtypes=[queries.dtype],
    )
    return out


# ── Per-request memory KV registry ────────────────────────────────────────────
# Populated by openai.py before generation; consumed by attention_split.py
# during the model forward pass; cleared after generation completes.
# Key: layer index (int, 0-based). Value: (mem_keys, mem_values) mx.arrays.
# Thread-safety: MTPLX runs the forward pass on a single thread under
# asyncio.to_thread — no lock needed for the registry dict itself.

# Registry now stores (keys, values, importance_weights) per layer
# importance_weights: [N_mem] array, scales attention per memory
_mem_kv_registry: dict[int, tuple[mx.array, mx.array, mx.array | None]] = {}


def register_mem_kv_for_request(
    layer_kv_map: dict[int, tuple[mx.array, mx.array]],
    importance_weights: mx.array | None = None,
) -> None:
    """Populate the per-request memory KV registry before generation.

    Called by openai.py after ENI KV injection. ``layer_kv_map`` maps each
    layer index to its (mem_keys, mem_values) pair in model-head-dim space.
    
    Args:
        layer_kv_map: Per-layer K/V tensors
        importance_weights: Optional [N_mem] importance scores (0.0-1.0)
                           High importance (>0.8) memories get stronger signal
    
    Replaces any previous registry atomically.
    """
    global _mem_kv_registry
    # Store with importance weights for all layers
    _mem_kv_registry = {
        layer_idx: (k, v, importance_weights)
        for layer_idx, (k, v) in layer_kv_map.items()
    }


def get_mem_kv_for_layer(
    layer_idx: int,
) -> tuple[mx.array, mx.array, mx.array | None] | None:
    """Return (mem_keys, mem_values, importance_weights) for ``layer_idx``, or None if not set.

    Called by attention_split.py during the model forward pass.
    Returns importance_weights as third element (or None if not set).
    """
    return _mem_kv_registry.get(layer_idx)


def clear_mem_kv_registry() -> None:
    """Drop all registry entries.  Called by openai.py after generation completes."""
    global _mem_kv_registry
    _mem_kv_registry = {}


def get_mem_fused_stats() -> dict:
    """Observability: bail counts and kernel availability for /health endpoints."""
    return {
        "metal_available": mx.metal.is_available(),
        "kernel_cached": _mem_fused_q1_partials_kernel.cache_info().currsize > 0,
        "registry_layers": len(_mem_kv_registry),
        "bail_counts": dict(mem_fused_bail_counts),
        "total_bails": sum(mem_fused_bail_counts.values()),
    }

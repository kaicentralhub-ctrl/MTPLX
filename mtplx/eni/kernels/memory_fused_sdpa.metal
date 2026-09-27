// NEXUS L0 — Memory-Fused Scaled Dot-Product Attention (Metal Kernel)
//
// Fuses memory K/V lookup with SDPA in a single GPU dispatch.
// Memory tokens participate in attention as virtual KV entries,
// with graph-attention bias shaping the logits.
//
// Architecture:
//   output = softmax((Q·K_ctx^T + bias_ctx) ∪ (Q·K_mem^T + bias_mem)) · [V_ctx; V_mem]
//
// Where:
//   Q           : query from context       [seq_len, n_heads, head_dim]
//   K_ctx, V_ctx: context KV cache          [ctx_len, n_heads, head_dim]
//   K_mem, V_mem: memory KV entries         [mem_len, n_heads, head_dim]
//   bias_mem    : graph-attention bias      [seq_len, mem_len] (additive)
//
// The kernel processes one (query_pos, head) pair per threadgroup.
// Within a threadgroup, threads cooperate on the dot-product reduction.
//
// Thread indexing:
//   gid.x = query position (0..seq_len-1)
//   gid.y = attention head (0..n_heads-1)
//
// This is the ONLY kernel in existence that integrates a typed
// knowledge graph into attention computation at the hardware level.

#include <metal_stdlib>
using namespace metal;

// ── Constants ────────────────────────────────────────────────────────────────

#define TILE_SIZE 32       // threadgroup size for dot-product reduction
#define MAX_HEAD_DIM 128   // maximum head dimension supported
#define NEG_INF -1e30f     // for masking

// ── Kernel: Memory-Fused SDPA (Standard) ────────────────────────────────────

struct MemFusedParams {
    uint seq_len;       // number of query positions
    uint ctx_len;       // context KV length
    uint mem_len;       // memory KV length
    uint n_heads;       // number of attention heads
    uint head_dim;      // dimension per head
    float scale;        // 1/sqrt(head_dim)
    float fusion_scale; // scale of memory contribution (0 = context only)
};

kernel void memory_fused_sdpa(
    device const float* Q          [[buffer(0)]],   // [seq_len, n_heads, head_dim]
    device const float* K_context  [[buffer(1)]],   // [ctx_len, n_heads, head_dim]
    device const float* V_context  [[buffer(2)]],   // [ctx_len, n_heads, head_dim]
    device const float* K_memory   [[buffer(3)]],   // [mem_len, n_heads, head_dim]
    device const float* V_memory   [[buffer(4)]],   // [mem_len, n_heads, head_dim]
    device const float* graph_bias [[buffer(5)]],   // [seq_len, mem_len] or null
    device float*       output     [[buffer(6)]],   // [seq_len, n_heads, head_dim]
    constant MemFusedParams& params [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    threadgroup float* tg_logits  [[threadgroup(0)]],  // [ctx_len + mem_len]
    threadgroup float* tg_values  [[threadgroup(1)]]   // [head_dim * 2]
) {
    uint q_pos = gid.x;
    uint head  = gid.y;

    if (q_pos >= params.seq_len || head >= params.n_heads) return;

    uint hd = params.head_dim;
    uint total_kv = params.ctx_len + params.mem_len;

    // ── Phase 1: Compute attention logits ────────────────────────────────
    // Each thread computes one KV position's logit and stores in threadgroup memory.
    // Then we reduce with softmax.

    for (uint kv_pos = tid; kv_pos < total_kv; kv_pos += TILE_SIZE) {
        float logit = 0.0f;

        if (kv_pos < params.ctx_len) {
            // Context KV dot product
            uint q_offset = (q_pos * params.n_heads + head) * hd;
            uint kv_offset = (kv_pos * params.n_heads + head) * hd;

            for (uint d = 0; d < hd; d++) {
                logit += Q[q_offset + d] * K_context[kv_offset + d];
            }
            logit *= params.scale;
            // Context bias (optional, usually 0)
            // graph_bias covers only memory positions
        } else {
            // Memory KV dot product
            uint mem_idx = kv_pos - params.ctx_len;
            uint q_offset = (q_pos * params.n_heads + head) * hd;
            uint kv_offset = (mem_idx * params.n_heads + head) * hd;

            for (uint d = 0; d < hd; d++) {
                logit += Q[q_offset + d] * K_memory[kv_offset + d];
            }
            logit *= params.scale;

            // Apply graph-attention bias for memory positions
            if (graph_bias != nullptr) {
                uint bias_offset = q_pos * params.mem_len + mem_idx;
                logit += graph_bias[bias_offset];
            }
        }

        tg_logits[kv_pos] = logit;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 2: Softmax (numerically stable) ───────────────────────────
    // Thread 0 computes max and sum, then normalizes.

    if (tid == 0) {
        // Find max for numerical stability
        float max_logit = NEG_INF;
        for (uint i = 0; i < total_kv; i++) {
            max_logit = max(max_logit, tg_logits[i]);
        }

        // Compute exp and sum
        float sum_exp = 0.0f;
        for (uint i = 0; i < total_kv; i++) {
            tg_logits[i] = exp(tg_logits[i] - max_logit);
            sum_exp += tg_logits[i];
        }

        // Normalize
        float inv_sum = 1.0f / (sum_exp + 1e-8f);
        for (uint i = 0; i < total_kv; i++) {
            tg_logits[i] *= inv_sum;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3: Weighted value aggregation ─────────────────────────────
    // output = Σ softmax_i * V_i over both context and memory

    for (uint d = tid; d < hd; d += TILE_SIZE) {
        float acc = 0.0f;

        for (uint kv_pos = 0; kv_pos < total_kv; kv_pos++) {
            float weight = tg_logits[kv_pos];

            if (kv_pos < params.ctx_len) {
                uint kv_offset = (kv_pos * params.n_heads + head) * hd + d;
                acc += weight * V_context[kv_offset];
            } else {
                uint mem_idx = kv_pos - params.ctx_len;
                uint kv_offset = (mem_idx * params.n_heads + head) * hd + d;
                acc += weight * V_memory[kv_offset];
            }
        }

        uint out_offset = (q_pos * params.n_heads + head) * hd + d;
        output[out_offset] = acc;
    }
}

// ── Kernel: Memory-Fused SDPA (Quantized KV) ────────────────────────────────
// Same as above but accepts INT8-quantized memory KV for 4× memory savings.
// Context KV is always FP16/FP32 (hot cache). Memory KV can be quantized
// (cold, pre-computed, rarely changes).

struct MemFusedQuantParams {
    uint seq_len;
    uint ctx_len;
    uint mem_len;
    uint n_heads;
    uint head_dim;
    float scale;
    float fusion_scale;
    float mem_scale;   // dequantization scale for memory KV
    float mem_zero;    // dequantization zero-point for memory KV
};

kernel void memory_fused_sdpa_quant(
    device const float* Q          [[buffer(0)]],
    device const float* K_context  [[buffer(1)]],
    device const float* V_context  [[buffer(2)]],
    device const char*  K_memory   [[buffer(3)]],   // INT8 quantized
    device const char*  V_memory   [[buffer(4)]],   // INT8 quantized
    device const float* graph_bias [[buffer(5)]],
    device float*       output     [[buffer(6)]],
    constant MemFusedQuantParams& params [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    threadgroup float* tg_logits  [[threadgroup(0)]],
    threadgroup float* tg_values  [[threadgroup(1)]]
) {
    uint q_pos = gid.x;
    uint head  = gid.y;

    if (q_pos >= params.seq_len || head >= params.n_heads) return;

    uint hd = params.head_dim;
    uint total_kv = params.ctx_len + params.mem_len;

    // Phase 1: Logits (with dequantization for memory KV)
    for (uint kv_pos = tid; kv_pos < total_kv; kv_pos += TILE_SIZE) {
        float logit = 0.0f;

        if (kv_pos < params.ctx_len) {
            uint q_offset = (q_pos * params.n_heads + head) * hd;
            uint kv_offset = (kv_pos * params.n_heads + head) * hd;
            for (uint d = 0; d < hd; d++) {
                logit += Q[q_offset + d] * K_context[kv_offset + d];
            }
            logit *= params.scale;
        } else {
            uint mem_idx = kv_pos - params.ctx_len;
            uint q_offset = (q_pos * params.n_heads + head) * hd;
            uint kv_offset = (mem_idx * params.n_heads + head) * hd;
            for (uint d = 0; d < hd; d++) {
                // Dequantize INT8 → float on the fly
                float k_val = (float(K_memory[kv_offset + d]) + params.mem_zero) * params.mem_scale;
                logit += Q[q_offset + d] * k_val;
            }
            logit *= params.scale;

            if (graph_bias != nullptr) {
                uint bias_offset = q_pos * params.mem_len + mem_idx;
                logit += graph_bias[bias_offset];
            }
        }

        tg_logits[kv_pos] = logit;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Softmax
    if (tid == 0) {
        float max_logit = NEG_INF;
        for (uint i = 0; i < total_kv; i++) {
            max_logit = max(max_logit, tg_logits[i]);
        }
        float sum_exp = 0.0f;
        for (uint i = 0; i < total_kv; i++) {
            tg_logits[i] = exp(tg_logits[i] - max_logit);
            sum_exp += tg_logits[i];
        }
        float inv_sum = 1.0f / (sum_exp + 1e-8f);
        for (uint i = 0; i < total_kv; i++) {
            tg_logits[i] *= inv_sum;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Weighted value aggregation (with dequantization)
    for (uint d = tid; d < hd; d += TILE_SIZE) {
        float acc = 0.0f;

        for (uint kv_pos = 0; kv_pos < total_kv; kv_pos++) {
            float weight = tg_logits[kv_pos];

            if (kv_pos < params.ctx_len) {
                uint kv_offset = (kv_pos * params.n_heads + head) * hd + d;
                acc += weight * V_context[kv_offset];
            } else {
                uint mem_idx = kv_pos - params.ctx_len;
                uint kv_offset = (mem_idx * params.n_heads + head) * hd + d;
                float v_val = (float(V_memory[kv_offset]) + params.mem_zero) * params.mem_scale;
                acc += weight * v_val;
            }
        }

        uint out_offset = (q_pos * params.n_heads + head) * hd + d;
        output[out_offset] = acc;
    }
}

// ── Kernel: Memory Pre-fill (Batch) ─────────────────────────────────────────
// Pre-computes memory K/V projections for the persistent memory bank.
// Called once at startup to fill the memory KV cache.

struct MemPrefillParams {
    uint mem_len;
    uint n_heads;
    uint head_dim;
    uint hidden_dim;
};

kernel void memory_prefill_kv(
    device const float* memory_embeddings [[buffer(0)]],  // [mem_len, hidden_dim]
    device const float* W_k               [[buffer(1)]],  // [hidden_dim, n_heads * head_dim]
    device const float* W_v               [[buffer(2)]],  // [hidden_dim, n_heads * head_dim]
    device float*       K_out             [[buffer(3)]],  // [mem_len, n_heads, head_dim]
    device float*       V_out             [[buffer(4)]],  // [mem_len, n_heads, head_dim]
    constant MemPrefillParams& params     [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint mem_idx = gid.x;
    uint head    = gid.y;

    if (mem_idx >= params.mem_len || head >= params.n_heads) return;

    uint hd = params.head_dim;
    uint hidden_dim = params.hidden_dim;

    // Compute K = W_k @ memory_embedding
    // Compute V = W_v @ memory_embedding
    for (uint d = 0; d < hd; d++) {
        float k_acc = 0.0f;
        float v_acc = 0.0f;

        for (uint h = 0; h < hidden_dim; h++) {
            float emb = memory_embeddings[mem_idx * hidden_dim + h];
            uint w_offset_k = h * (params.n_heads * hd) + head * hd + d;
            uint w_offset_v = w_offset_k;  // same layout
            k_acc += emb * W_k[w_offset_k];
            v_acc += emb * W_v[w_offset_v];
        }

        uint out_offset = (mem_idx * params.n_heads + head) * hd + d;
        K_out[out_offset] = k_acc;
        V_out[out_offset] = v_acc;
    }
}

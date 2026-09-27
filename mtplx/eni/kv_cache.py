"""
ENI Phase A: KV Cache Pre-Computation (Tier 1 — FusionRAG/TurboRAG technique)
==============================================================================

Architecture
------------
First request with a given ENI injection hash:
  normal MTPLX prefill → we capture the resulting CacheSnapshot at the
  prompt boundary → serialize MLX arrays → store in Postgres alongside
  the injection_hash.

All subsequent requests with the SAME injection hash:
  load snapshot from Postgres → reconstruct MLX arrays → inject a
  synthetic SessionBankEntry into session_bank._entries with the correct
  token_ids prefix → MTPLX sees a bank HIT → skips prefill entirely.

Result: ~9× TTFT reduction on memory-heavy requests after the first cold hit.

Position alignment note
-----------------------
ENI memories are injected as the *system prompt* — positions [0 .. N_mem).
This means their absolute RoPE positions are fixed and predictable, so
pre-computed KV is position-correct without LazyAttention or any
re-encoding. The user turn that follows always starts at position N_mem,
exactly where the live prefill would have left it.

Postgres schema (runs once at import)
--------------------------------------
  CREATE TABLE IF NOT EXISTS eni_kv_cache (
      injection_hash   TEXT PRIMARY KEY,
      token_ids        BYTEA NOT NULL,      -- pickled tuple[int,...]
      snapshot_bytes   BYTEA NOT NULL,      -- serialized CacheSnapshot
      logits_bytes     BYTEA,              -- serialized logits array
      model_path       TEXT NOT NULL,
      token_count      INT NOT NULL,
      nbytes           BIGINT NOT NULL,
      created_at       TIMESTAMPTZ DEFAULT now(),
      hits             INT DEFAULT 0
  );

Usage
-----
From openai.py (patched in Phase A):

  # Before restore_or_prefill_prompt_state():
  from mtplx.eni.kv_cache import maybe_inject_kv
  injected = maybe_inject_kv(session_bank, token_ids, eni_hash, rt)

  # After restore_or_prefill_prompt_state() on a miss:
  from mtplx.eni.kv_cache import capture_kv_async
  if not injected:
      capture_kv_async(cache, logits, token_ids, eni_hash, rt)
"""

from __future__ import annotations

import hashlib
import io
import logging
import pickle
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

log = logging.getLogger("mtplx.eni.kv_cache")

# ---------------------------------------------------------------------------
# Lazy imports — only pulled when KV cache is actually active
# ---------------------------------------------------------------------------
_mlx: Any = None
_np: Any = None


def _get_mlx():
    global _mlx
    if _mlx is None:
        try:
            import mlx.core as mx
            _mlx = mx
        except ImportError:
            log.warning("[ENI-KV] mlx.core not available — KV pre-computation disabled")
    return _mlx


def _get_np():
    global _np
    if _np is None:
        try:
            import numpy as np
            _np = np
        except ImportError:
            log.warning("[ENI-KV] numpy not available — KV pre-computation disabled")
    return _np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class KVCacheConfig:
    enabled: bool = True
    max_token_count: int = 4096       # skip capture for very long injections
    max_snapshot_mb: float = 512.0    # don't persist snapshots > 512 MB
    async_capture: bool = True        # fire captures on background thread
    dsn: str = ""                     # postgres DSN (inherits from ENIConfig if empty)
    # In-process LRU for loaded snapshots (avoids Postgres round-trip on hits)
    ram_cache_size: int = 16          # number of snapshots to hold in RAM


_config: KVCacheConfig = KVCacheConfig()


def configure(cfg: KVCacheConfig) -> None:
    global _config
    _config = cfg


# ---------------------------------------------------------------------------
# In-process RAM cache
# ---------------------------------------------------------------------------
_ram_cache: dict[str, tuple[Any, Any, tuple[int, ...]]] = {}  # hash → (snapshot, logits, token_ids)
_ram_cache_order: list[str] = []
_ram_lock = threading.Lock()


def _ram_put(injection_hash: str, snapshot: Any, logits: Any, token_ids: tuple[int, ...]) -> None:
    with _ram_lock:
        if injection_hash in _ram_cache:
            return  # already cached
        _ram_cache[injection_hash] = (snapshot, logits, token_ids)
        _ram_cache_order.append(injection_hash)
        # Evict LRU if over limit
        while len(_ram_cache_order) > _config.ram_cache_size:
            old = _ram_cache_order.pop(0)
            _ram_cache.pop(old, None)


def _ram_get(injection_hash: str) -> tuple[Any, Any, tuple[int, ...]] | None:
    with _ram_lock:
        entry = _ram_cache.get(injection_hash)
        if entry is None:
            return None
        # Move to MRU end
        try:
            _ram_cache_order.remove(injection_hash)
        except ValueError:
            pass
        _ram_cache_order.append(injection_hash)
        return entry


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------
def _mlx_to_numpy(value: Any) -> Any:
    """Recursively convert MLX arrays to numpy. Returns JSON-able structure."""
    mx = _get_mlx()
    np = _get_np()
    if mx is None or np is None:
        return None
    if isinstance(value, mx.array):
        # eval() forces computation before conversion — avoids lazy graph issues
        mx.eval(value)
        return np.array(value)
    if isinstance(value, tuple):
        return tuple(_mlx_to_numpy(v) for v in value)
    if isinstance(value, list):
        return [_mlx_to_numpy(v) for v in value]
    if isinstance(value, dict):
        return {k: _mlx_to_numpy(v) for k, v in value.items()}
    return value  # None, int, float, str — pass through


def _numpy_to_mlx(value: Any) -> Any:
    """Recursively convert numpy arrays back to MLX."""
    mx = _get_mlx()
    np = _get_np()
    if mx is None or np is None:
        return None
    if isinstance(value, np.ndarray):
        return mx.array(value)
    if isinstance(value, tuple):
        return tuple(_numpy_to_mlx(v) for v in value)
    if isinstance(value, list):
        return [_numpy_to_mlx(v) for v in value]
    if isinstance(value, dict):
        return {k: _numpy_to_mlx(v) for k, v in value.items()}
    return value


def serialize_snapshot(snapshot: Any) -> bytes:
    """Serialize a CacheSnapshot to bytes via numpy.
    
    CacheSnapshot is frozen(states=tuple[state|None], meta_states=tuple[meta|None]).
    Each state is a (keys, values) tuple of MLX arrays for one transformer layer.
    Qwen3.6-27B (pure transformer): 28 layers, meta_states all None.
    """
    np = _get_np()
    if np is None:
        raise RuntimeError("numpy required for snapshot serialization")
    
    payload = {
        "states": _mlx_to_numpy(snapshot.states),
        "meta_states": _mlx_to_numpy(snapshot.meta_states),
    }
    buf = io.BytesIO()
    # pickle handles None, tuples, nested numpy arrays cleanly
    pickle.dump(payload, buf, protocol=pickle.HIGHEST_PROTOCOL)
    return buf.getvalue()


def deserialize_snapshot(data: bytes) -> Any:
    """Deserialize bytes back to a CacheSnapshot (with MLX arrays)."""
    from mtplx.cache_state import CacheSnapshot
    
    payload = pickle.loads(data)
    states = _numpy_to_mlx(payload["states"])
    meta_states = _numpy_to_mlx(payload["meta_states"])
    return CacheSnapshot(
        states=states if isinstance(states, tuple) else tuple(states or []),
        meta_states=meta_states if isinstance(meta_states, tuple) else tuple(meta_states or []),
    )


def serialize_logits(logits: Any) -> bytes | None:
    """Serialize the final-prefix logits array (used for MTP draft seeding)."""
    if logits is None:
        return None
    np = _get_np()
    if np is None:
        return None
    try:
        buf = io.BytesIO()
        pickle.dump(_mlx_to_numpy(logits), buf, protocol=pickle.HIGHEST_PROTOCOL)
        return buf.getvalue()
    except Exception as e:
        log.debug(f"[ENI-KV] logits serialize failed (non-fatal): {e}")
        return None


def deserialize_logits(data: bytes | None) -> Any:
    if data is None:
        return None
    try:
        np_logits = pickle.loads(data)
        return _numpy_to_mlx(np_logits)
    except Exception as e:
        log.debug(f"[ENI-KV] logits deserialize failed (non-fatal): {e}")
        return None


def estimate_snapshot_bytes(snapshot: Any) -> int:
    """Estimate serialized size without full serialization."""
    mx = _get_mlx()
    if mx is None:
        return 0
    
    def _count_bytes(value: Any) -> int:
        if isinstance(value, mx.array):
            return value.nbytes
        if isinstance(value, (tuple, list)):
            return sum(_count_bytes(v) for v in value)
        if isinstance(value, dict):
            return sum(_count_bytes(v) for v in value.values())
        return 0
    
    return _count_bytes(snapshot.states) + _count_bytes(snapshot.meta_states)


# ---------------------------------------------------------------------------
# Postgres persistence
# ---------------------------------------------------------------------------
_db_conn_lock = threading.Lock()
_db_conn: Any = None
_schema_initialized = False


def _get_dsn() -> str:
    dsn = _config.dsn
    if not dsn:
        # Fall back to ENI config
        try:
            from mtplx.eni.config import get_config
            c = get_config()
            dsn = c.db_dsn
        except Exception:
            pass
    return dsn


def _get_conn():
    global _db_conn, _schema_initialized
    dsn = _get_dsn()
    if not dsn:
        return None
    
    try:
        import psycopg
    except ImportError:
        log.debug("[ENI-KV] psycopg3 not available — no Postgres persistence")
        return None
    
    with _db_conn_lock:
        # Check if existing connection is alive
        if _db_conn is not None:
            try:
                _db_conn.execute("SELECT 1")
                return _db_conn
            except Exception:
                try:
                    _db_conn.close()
                except Exception:
                    pass
                _db_conn = None
        
        try:
            _db_conn = psycopg.connect(dsn, autocommit=True)
            if not _schema_initialized:
                _init_schema(_db_conn)
                _schema_initialized = True
            return _db_conn
        except Exception as e:
            log.warning(f"[ENI-KV] Postgres connect failed: {e}")
            _db_conn = None
            return None


def _init_schema(conn: Any) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS eni_kv_cache (
            injection_hash   TEXT PRIMARY KEY,
            token_ids        BYTEA NOT NULL,
            snapshot_bytes   BYTEA NOT NULL,
            logits_bytes     BYTEA,
            model_path       TEXT NOT NULL,
            token_count      INT NOT NULL,
            nbytes           BIGINT NOT NULL,
            created_at       TIMESTAMPTZ DEFAULT now(),
            hits             INT DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS eni_kv_cache_model_idx
        ON eni_kv_cache (model_path, token_count)
    """)
    log.info("[ENI-KV] Schema initialized (eni_kv_cache)")


def store_kv(
    injection_hash: str,
    token_ids: tuple[int, ...],
    snapshot: Any,
    logits: Any,
    model_path: str,
) -> bool:
    """Persist a CacheSnapshot to Postgres. Returns True on success."""
    if not _config.enabled:
        return False
    
    # Size guard
    nbytes = estimate_snapshot_bytes(snapshot)
    max_bytes = int(_config.max_snapshot_mb * 1024 * 1024)
    if nbytes > max_bytes:
        log.info(
            f"[ENI-KV] Snapshot too large ({nbytes / 2**20:.1f} MB > "
            f"{_config.max_snapshot_mb:.0f} MB) — skipping persist for {injection_hash[:8]}"
        )
        return False
    
    try:
        t0 = time.perf_counter()
        snapshot_bytes = serialize_snapshot(snapshot)
        logits_bytes = serialize_logits(logits)
        token_ids_bytes = pickle.dumps(token_ids, protocol=pickle.HIGHEST_PROTOCOL)
        serialize_ms = (time.perf_counter() - t0) * 1000
        
        conn = _get_conn()
        if conn is None:
            log.debug(f"[ENI-KV] No DB conn — RAM-only cache for {injection_hash[:8]}")
            _ram_put(injection_hash, snapshot, logits, token_ids)
            return True
        
        conn.execute(
            """
            INSERT INTO eni_kv_cache
                (injection_hash, token_ids, snapshot_bytes, logits_bytes,
                 model_path, token_count, nbytes)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (injection_hash) DO NOTHING
            """,
            (
                injection_hash,
                token_ids_bytes,
                snapshot_bytes,
                logits_bytes,
                model_path,
                len(token_ids),
                nbytes,
            ),
        )
        _ram_put(injection_hash, snapshot, logits, token_ids)
        log.info(
            f"[ENI-KV] Stored snapshot: hash={injection_hash[:8]} "
            f"tokens={len(token_ids)} size={nbytes/2**20:.1f}MB "
            f"serialize={serialize_ms:.0f}ms"
        )
        return True
    except Exception as e:
        log.warning(f"[ENI-KV] store_kv failed: {e}")
        return False


def load_kv(
    injection_hash: str,
    model_path: str,
) -> tuple[Any, Any, tuple[int, ...]] | None:
    """Load (snapshot, logits, token_ids) for an injection_hash. Returns None on miss."""
    if not _config.enabled:
        return None
    
    # RAM cache first (no deserialization cost)
    cached = _ram_get(injection_hash)
    if cached is not None:
        log.debug(f"[ENI-KV] RAM hit: {injection_hash[:8]}")
        return cached
    
    conn = _get_conn()
    if conn is None:
        return None
    
    try:
        row = conn.execute(
            """
            UPDATE eni_kv_cache
            SET hits = hits + 1
            WHERE injection_hash = %s AND model_path = %s
            RETURNING snapshot_bytes, logits_bytes, token_ids
            """,
            (injection_hash, model_path),
        ).fetchone()
        
        if row is None:
            return None
        
        t0 = time.perf_counter()
        snapshot = deserialize_snapshot(bytes(row[0]))
        logits = deserialize_logits(bytes(row[1]) if row[1] else None)
        token_ids = pickle.loads(bytes(row[2]))
        deser_ms = (time.perf_counter() - t0) * 1000
        
        _ram_put(injection_hash, snapshot, logits, token_ids)
        log.info(
            f"[ENI-KV] DB hit: hash={injection_hash[:8]} "
            f"tokens={len(token_ids)} deserialize={deser_ms:.0f}ms"
        )
        return snapshot, logits, token_ids
    except Exception as e:
        log.warning(f"[ENI-KV] load_kv failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Bank injection — synthetic SessionBankEntry injection
# ---------------------------------------------------------------------------
def _compute_token_hash(token_ids: tuple[int, ...]) -> str:
    """Compute a token hash matching MTPLX's internal convention (SHA-256 hex)."""
    raw = b"".join(t.to_bytes(4, "little") for t in token_ids)
    return hashlib.sha256(raw).hexdigest()


def inject_into_bank(
    session_bank: Any,
    snapshot: Any,
    logits: Any,
    token_ids: tuple[int, ...],
    runtime: Any,
) -> bool:
    """Inject a pre-computed CacheSnapshot into session_bank._entries.
    
    Creates a synthetic SessionBankEntry with the pre-computed state, keyed by
    token_ids. When MTPLX calls session_bank.get(prompt_token_ids) and the prompt
    starts with these token_ids, it gets a bank HIT and skips prefill entirely.
    
    Returns True if injection succeeded.
    """
    if session_bank is None:
        return False
    
    try:
        from mtplx.session_bank import SessionBankEntry
        from mtplx.cache_state import CacheSnapshot
        
        nbytes = estimate_snapshot_bytes(snapshot)
        token_hash = _compute_token_hash(token_ids)
        
        # Build synthetic entry — mirrors what put() would produce
        entry = SessionBankEntry(
            token_ids=token_ids,
            token_hash=token_hash,
            model_path=str(getattr(runtime, "model_path", "")),
            mtp_enabled=bool(getattr(runtime, "mtp_enabled", False)),
            hidden_variant=None,
            cache_snapshot=snapshot,
            logits=logits,
            hidden=None,           # no hidden for prefix-only entry
            cache_ref=None,
            live_ref_only=False,
            nbytes=nbytes,
            session_id=None,       # shared across sessions (memory content is stable)
            lazy_kv=False,
            has_recurrent=False,
        )
        
        # Direct injection into the entries dict
        # Thread safety: session_bank._entries is a plain dict; MTPLX's own
        # put() is also non-atomic on _entries, so we match the existing
        # concurrency contract. A lock around this would be ideal but would
        # require patching SessionBank itself.
        session_bank._entries[token_ids] = entry
        log.info(
            f"[ENI-KV] Bank injection: tokens={len(token_ids)} "
            f"size={nbytes/2**20:.1f}MB hash={token_hash[:8]}"
        )
        return True
    except Exception as e:
        log.warning(f"[ENI-KV] Bank injection failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Public API — called from openai.py patches
# ---------------------------------------------------------------------------
def maybe_inject_kv(
    session_bank: Any,
    prompt_token_ids: list[int] | tuple[int, ...],
    injection_hash: str,
    runtime: Any,
) -> bool:
    """Try to pre-populate session_bank with a cached KV snapshot.
    
    Called BEFORE restore_or_prefill_prompt_state(). If we have a cached
    snapshot for this injection_hash, we inject it so MTPLX sees a bank hit.
    
    Returns True if injection succeeded (prefill will be skipped by MTPLX).
    Returns False if cache miss (normal prefill proceeds, capture fires after).
    """
    if not _config.enabled or injection_hash in ("", "empty"):
        return False
    
    model_path = str(getattr(runtime, "model_path", ""))
    cached = load_kv(injection_hash, model_path)
    if cached is None:
        log.debug(f"[ENI-KV] Cache miss: {injection_hash[:8]}")
        return False
    
    snapshot, logits, cached_token_ids = cached
    
    # Verify the cached token_ids are a prefix of the current prompt
    # (sanity check — the injection hash guarantees this, but be defensive)
    n = len(cached_token_ids)
    if list(prompt_token_ids[:n]) != list(cached_token_ids):
        log.warning(
            f"[ENI-KV] Token mismatch for {injection_hash[:8]} — "
            "injection hash collision or model changed. Evicting."
        )
        _evict(injection_hash)
        return False
    
    return inject_into_bank(session_bank, snapshot, logits, cached_token_ids, runtime)


def capture_kv_async(
    cache: list[Any],
    logits: Any,
    token_ids: list[int] | tuple[int, ...],
    injection_hash: str,
    runtime: Any,
) -> None:
    """Capture and store the KV snapshot for the current injection hash.
    
    Called AFTER restore_or_prefill_prompt_state() on a cache MISS.
    The cache/logits at the prompt boundary are snapshotted and persisted
    so future requests with the same injection_hash get a bank hit.
    
    Runs on a background thread to not block generation start.
    """
    if not _config.enabled or injection_hash in ("", "empty"):
        return
    if len(token_ids) > _config.max_token_count:
        log.debug(
            f"[ENI-KV] Token count {len(token_ids)} > max {_config.max_token_count} "
            "— skipping capture"
        )
        return
    
    # Already cached?
    if _ram_get(injection_hash) is not None:
        return
    
    model_path = str(getattr(runtime, "model_path", ""))
    
    def _do_capture():
        try:
            from mtplx.cache_state import snapshot_cache
            
            t0 = time.perf_counter()
            snapshot = snapshot_cache(cache)
            token_ids_tuple = tuple(int(t) for t in token_ids)
            
            # Eval all arrays now, while the cache is still valid
            mx = _get_mlx()
            if mx is not None:
                # Force evaluation of all lazy MLX expressions in the snapshot
                def _collect_arrays(v):
                    mx = _get_mlx()
                    if isinstance(v, mx.array):
                        return [v]
                    if isinstance(v, (tuple, list)):
                        result = []
                        for x in v:
                            result.extend(_collect_arrays(x))
                        return result
                    if isinstance(v, dict):
                        result = []
                        for x in v.values():
                            result.extend(_collect_arrays(x))
                        return result
                    return []
                
                all_arrays = (
                    _collect_arrays(snapshot.states) +
                    _collect_arrays(snapshot.meta_states)
                )
                if all_arrays:
                    mx.eval(*all_arrays)
            
            snapshot_ms = (time.perf_counter() - t0) * 1000
            store_kv(injection_hash, token_ids_tuple, snapshot, logits, model_path)
            log.info(f"[ENI-KV] Capture complete: {injection_hash[:8]} snapshot_ms={snapshot_ms:.0f}")
        except Exception as e:
            log.warning(f"[ENI-KV] Async capture failed: {e}")
    
    if _config.async_capture:
        t = threading.Thread(target=_do_capture, daemon=True, name=f"eni-kv-capture-{injection_hash[:8]}")
        t.start()
    else:
        _do_capture()


def _evict(injection_hash: str) -> None:
    """Remove a stale/mismatched entry from RAM and Postgres."""
    with _ram_lock:
        _ram_cache.pop(injection_hash, None)
        try:
            _ram_cache_order.remove(injection_hash)
        except ValueError:
            pass
    
    conn = _get_conn()
    if conn is not None:
        try:
            conn.execute(
                "DELETE FROM eni_kv_cache WHERE injection_hash = %s",
                (injection_hash,),
            )
        except Exception as e:
            log.debug(f"[ENI-KV] Evict from DB failed: {e}")


# ---------------------------------------------------------------------------
# Stats / introspection
# ---------------------------------------------------------------------------
def get_kv_stats() -> dict:
    """Return diagnostic stats for observability."""
    conn = _get_conn()
    db_stats: dict = {}
    if conn is not None:
        try:
            row = conn.execute(
                "SELECT COUNT(*), SUM(nbytes), SUM(hits) FROM eni_kv_cache"
            ).fetchone()
            if row:
                db_stats = {
                    "entries": row[0] or 0,
                    "total_mb": round((row[1] or 0) / 2**20, 1),
                    "total_hits": row[2] or 0,
                }
        except Exception:
            pass
    
    with _ram_lock:
        ram_entries = len(_ram_cache)
    
    return {
        "enabled": _config.enabled,
        "ram_entries": ram_entries,
        "ram_cache_size": _config.ram_cache_size,
        **db_stats,
    }

"""
ENI Phase B (T7): Cross-Request Persistent Adaptive Draft Temperature
======================================================================

Problem
-------
MTPLX's built-in ``adaptive_dtemp`` controller is per-request: it seeds
from scratch on each request, wasting N=seed_rounds (typically 3-5 decode
rounds) before having any signal about whether to boost draft temperature for
the current content register.

For ENI-augmented requests the injection hash uniquely identifies the memory
content set. Across requests with the SAME injection hash, the content
register (formulaic vs. duplicated vs. natural text) is essentially identical.
We know the historical accept rate from prior requests — ENI T7 persists it.

Solution
--------
Cross-request EMA of pos-1 draft acceptance rate, keyed by injection_hash:

* After each generation, record the observed accept_rate + base draft temp
  into Postgres (async, non-blocking).
* Before draft sampler resolution on the NEXT request with the same hash,
  load the historical EMA and recommend a starting draft temperature —
  skipping the seed phase entirely.
* MTPLX's own adaptive_dtemp still runs per-request; ENI T7 pre-seeds its
  effective starting temperature, compounding gains.

Mechanism (mirroring adaptive_dtemp.py's calibrated bands, §08-25 receipts)
-----------------------------------------------------------------------------
* ema in [0.45, 0.78] → BOOST: recommend draft_temp = 0.85 (sharpening pays
  when the MTP head is right-but-diffuse: base accept ~.71-.76)
* ema >= 0.80 → HOLD: recommend base_temp (confident matching, sharpening
  hurts: base accept ~.80)
* ema < 0.45 → HOLD: no receipts, unmeasured territory
* cross-request EMA alpha=0.3 (slower than intra-request, more stable)

Hook points in openai.py
-------------------------
* INSERT 1 — inside ``_resolve_draft_sampler_for_request()``, after
  request_explicit check, before pinned/curve branches:
    ```python
    if _ENI_AS_AVAILABLE and _ENI_AVAILABLE and _eni_enabled():
        rec = _eni_recommend_draft_temp(...)
        if rec is not None:
            resolved = replace(base or target_sampler, temperature=rec)
            source, policy = "eni_content_aware", "eni_content_aware"
    ```
* INSERT 2 — after ``generated = run_generation_for_response()`` in the
  serial path:
    ```python
    if _ENI_AS_AVAILABLE and _ENI_AVAILABLE and _eni_enabled():
        _eni_record_accept_rate_async(generated, _eni_hash(), draft_sampler)
    ```

Postgres schema
---------------
    CREATE TABLE IF NOT EXISTS eni_adaptive_sampling (
        injection_hash     TEXT PRIMARY KEY,
        ema_accept_rate    FLOAT NOT NULL,
        n_observations     INT NOT NULL DEFAULT 1,
        recommended_temp   FLOAT NOT NULL,
        base_temp_observed FLOAT NOT NULL,
        updated_at         TIMESTAMPTZ DEFAULT now()
    );
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("mtplx.eni.adaptive_sampling")

# ---------------------------------------------------------------------------
# Thresholds (calibrated from adaptive_dtemp §08-25 receipts)
# ---------------------------------------------------------------------------
BOOST_TEMP = 0.85          # draft temperature when sharpening is beneficial
EMA_RAISE_THRESHOLD = 0.78 # boost when ema <= this (right-but-diffuse register)
EMA_FLOOR = 0.45           # don't touch below (unmeasured territory)
EMA_DROP_THRESHOLD = 0.80  # hold when ema >= this (confident matching register)
CROSS_REQUEST_ALPHA = 0.3  # EMA smoothing factor for cross-request updates
MIN_OBSERVATIONS = 2       # don't recommend until we have at least N data points


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class AdaptiveSamplingConfig:
    enabled: bool = True
    ram_cache_size: int = 256       # number of hashes in RAM LRU
    min_observations: int = MIN_OBSERVATIONS
    boost_temp: float = BOOST_TEMP
    raise_threshold: float = EMA_RAISE_THRESHOLD
    floor: float = EMA_FLOOR
    drop_threshold: float = EMA_DROP_THRESHOLD
    alpha: float = CROSS_REQUEST_ALPHA
    dsn: str = ""                   # inherits from ENIConfig if empty


_config: AdaptiveSamplingConfig = AdaptiveSamplingConfig()


def configure(cfg: AdaptiveSamplingConfig) -> None:
    global _config
    _config = cfg


# ---------------------------------------------------------------------------
# RAM LRU cache
# ---------------------------------------------------------------------------
@dataclass
class _CachedEntry:
    ema_accept_rate: float
    n_observations: int
    recommended_temp: float
    base_temp_observed: float


_ram_cache: dict[str, _CachedEntry] = {}
_ram_cache_order: list[str] = []
_ram_lock = threading.Lock()


def _ram_put(injection_hash: str, entry: _CachedEntry) -> None:
    with _ram_lock:
        _ram_cache[injection_hash] = entry
        try:
            _ram_cache_order.remove(injection_hash)
        except ValueError:
            pass
        _ram_cache_order.append(injection_hash)
        while len(_ram_cache_order) > _config.ram_cache_size:
            old = _ram_cache_order.pop(0)
            _ram_cache.pop(old, None)


def _ram_get(injection_hash: str) -> _CachedEntry | None:
    with _ram_lock:
        entry = _ram_cache.get(injection_hash)
        if entry is not None:
            try:
                _ram_cache_order.remove(injection_hash)
            except ValueError:
                pass
            _ram_cache_order.append(injection_hash)
        return entry


# ---------------------------------------------------------------------------
# Postgres persistence
# ---------------------------------------------------------------------------
_db_conn_lock = threading.Lock()
_db_conn: Any = None
_schema_initialized = False


def _get_dsn() -> str:
    dsn = _config.dsn
    if not dsn:
        try:
            from mtplx.eni.config import get_config
            dsn = get_config().db_dsn
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
        return None
    with _db_conn_lock:
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
            log.warning(f"[ENI-AS] Postgres connect failed: {e}")
            _db_conn = None
            return None


def _init_schema(conn: Any) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS eni_adaptive_sampling (
            injection_hash     TEXT PRIMARY KEY,
            ema_accept_rate    FLOAT NOT NULL,
            n_observations     INT NOT NULL DEFAULT 1,
            recommended_temp   FLOAT NOT NULL,
            base_temp_observed FLOAT NOT NULL,
            updated_at         TIMESTAMPTZ DEFAULT now()
        )
    """)
    log.info("[ENI-AS] Schema initialized (eni_adaptive_sampling)")


# ---------------------------------------------------------------------------
# EMA temperature recommendation logic
# ---------------------------------------------------------------------------
def _compute_recommended_temp(ema: float, base_temp: float) -> float:
    """Map a historical accept EMA to a recommended draft temperature.

    Mirrors adaptive_dtemp.py's threshold logic but as a stateless function
    over the cross-request EMA rather than intra-request obs.

    Returns:
        boost_temp when ema is in the benefiting band [floor, raise_threshold]
        base_temp otherwise (floor = hold, drop_threshold = hold too)
    """
    if ema < _config.floor or ema >= _config.drop_threshold:
        # Too low (unmeasured) or too high (confident) — don't sharpen
        return base_temp
    if ema <= _config.raise_threshold:
        # Right-but-diffuse register — sharpening helps
        return _config.boost_temp
    return base_temp


def _update_ema(current_ema: float, new_obs: float) -> float:
    alpha = _config.alpha
    return (1 - alpha) * current_ema + alpha * new_obs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_recommended_draft_temp(
    injection_hash: str,
    base_temp: float,
) -> float | None:
    """Return a recommended draft temperature from cross-request history.

    Returns None if:
    - disabled / empty hash
    - fewer than min_observations recorded
    - recommendation equals base_temp (no change needed)
    - base_temp is greedy (0) — never un-greedy a greedy draft

    Returns a float otherwise.
    """
    if not _config.enabled or not injection_hash or injection_hash == "empty":
        return None
    if base_temp <= 0:
        # Never touch greedy draft chains — matches adaptive_dtemp contract
        return None

    # RAM first
    entry = _ram_get(injection_hash)
    if entry is None:
        conn = _get_conn()
        if conn is None:
            return None
        try:
            row = conn.execute(
                """
                SELECT ema_accept_rate, n_observations, recommended_temp, base_temp_observed
                FROM eni_adaptive_sampling
                WHERE injection_hash = %s
                """,
                (injection_hash,),
            ).fetchone()
            if row is None:
                return None
            entry = _CachedEntry(
                ema_accept_rate=float(row[0]),
                n_observations=int(row[1]),
                recommended_temp=float(row[2]),
                base_temp_observed=float(row[3]),
            )
            _ram_put(injection_hash, entry)
        except Exception as e:
            log.debug(f"[ENI-AS] load failed: {e}")
            return None

    if entry.n_observations < _config.min_observations:
        log.debug(
            f"[ENI-AS] Insufficient observations ({entry.n_observations}) for {injection_hash[:8]}"
        )
        return None

    # Recompute recommendation with current base_temp
    # (base_temp may differ if operator restarted with different --draft-temperature)
    recommended = _compute_recommended_temp(entry.ema_accept_rate, base_temp)

    if abs(recommended - base_temp) < 0.001:
        # No meaningful change
        return None

    log.info(
        f"[ENI-AS] Content-aware draft temp: hash={injection_hash[:8]} "
        f"ema={entry.ema_accept_rate:.3f} n={entry.n_observations} "
        f"base={base_temp:.2f} → recommended={recommended:.2f}"
    )
    return recommended


def record_accept_rate_async(
    injection_hash: str,
    accept_rate: float,
    base_temp: float,
) -> None:
    """Record observed accept rate and update cross-request EMA.

    Called after generation completes. Runs on a background thread.
    """
    if not _config.enabled or not injection_hash or injection_hash == "empty":
        return
    if base_temp <= 0:
        return  # greedy — not meaningful to track
    if accept_rate < 0 or accept_rate > 1:
        return  # sanity guard

    def _do_record():
        try:
            conn = _get_conn()
            existing = _ram_get(injection_hash)

            if existing is not None:
                new_ema = _update_ema(existing.ema_accept_rate, accept_rate)
                new_n = existing.n_observations + 1
            else:
                # Try Postgres
                if conn is not None:
                    row = conn.execute(
                        "SELECT ema_accept_rate, n_observations FROM eni_adaptive_sampling WHERE injection_hash = %s",
                        (injection_hash,),
                    ).fetchone()
                    if row:
                        new_ema = _update_ema(float(row[0]), accept_rate)
                        new_n = int(row[1]) + 1
                    else:
                        new_ema = float(accept_rate)
                        new_n = 1
                else:
                    new_ema = float(accept_rate)
                    new_n = 1

            new_rec = _compute_recommended_temp(new_ema, base_temp)

            # Update RAM
            updated = _CachedEntry(
                ema_accept_rate=new_ema,
                n_observations=new_n,
                recommended_temp=new_rec,
                base_temp_observed=base_temp,
            )
            _ram_put(injection_hash, updated)

            # Persist to Postgres
            if conn is not None:
                conn.execute(
                    """
                    INSERT INTO eni_adaptive_sampling
                        (injection_hash, ema_accept_rate, n_observations,
                         recommended_temp, base_temp_observed, updated_at)
                    VALUES (%s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (injection_hash) DO UPDATE SET
                        ema_accept_rate    = EXCLUDED.ema_accept_rate,
                        n_observations     = EXCLUDED.n_observations,
                        recommended_temp   = EXCLUDED.recommended_temp,
                        base_temp_observed = EXCLUDED.base_temp_observed,
                        updated_at         = NOW()
                    """,
                    (injection_hash, new_ema, new_n, new_rec, base_temp),
                )
            log.info(
                f"[ENI-AS] Recorded: hash={injection_hash[:8]} "
                f"accept={accept_rate:.3f} ema={new_ema:.3f} "
                f"n={new_n} rec_temp={new_rec:.2f}"
            )
        except Exception as e:
            log.warning(f"[ENI-AS] record failed: {e}")

    t = threading.Thread(
        target=_do_record,
        daemon=True,
        name=f"eni-as-record-{injection_hash[:8]}",
    )
    t.start()


def extract_accept_rate_from_generated(generated: dict[str, Any]) -> float | None:
    """Pull accept_rate out of a generation result dict.

    Handles both nested stats dict and flat structure.
    """
    if not isinstance(generated, dict):
        return None
    try:
        stats = generated.get("stats") or generated
        ar = stats.get("accept_rate")
        if ar is None:
            return None
        return float(ar)
    except (TypeError, ValueError):
        return None


def get_sampling_stats() -> dict:
    """Diagnostic stats for observability."""
    conn = _get_conn()
    db_stats: dict = {}
    if conn is not None:
        try:
            row = conn.execute(
                "SELECT COUNT(*), AVG(ema_accept_rate), AVG(recommended_temp) FROM eni_adaptive_sampling"
            ).fetchone()
            if row:
                db_stats = {
                    "entries": row[0] or 0,
                    "avg_ema_accept_rate": round(float(row[1] or 0), 3),
                    "avg_recommended_temp": round(float(row[2] or 0), 3),
                }
        except Exception:
            pass
    with _ram_lock:
        ram_entries = len(_ram_cache)
    return {
        "enabled": _config.enabled,
        "ram_entries": ram_entries,
        **db_stats,
    }


# Canonical alias used by __init__.py and openai.py imports
get_adaptive_sampling_stats = get_sampling_stats

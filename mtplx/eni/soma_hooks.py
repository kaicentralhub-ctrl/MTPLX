"""ENI SOMA Hooks - Integration between MTPLX and SOMA adaptation.

Detects learning signals during inference and routes them to SomaWorker:
- Corrections: When Kai re-phrases after ENI's response
- Session end: When conversation ends (flush buffer, run adaptation)
- Re-prompts: Potential retrieval misses

Architecture:
    MTPLX inference → soma_hooks → SomaWorker (IPC) → SOMA adapt()
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable

from .config import get_config


log = logging.getLogger("mtplx.eni.soma_hooks")


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SomaHooksConfig:
    """Configuration for SOMA signal detection and routing."""
    enabled: bool = True
    
    # IPC settings
    socket_path: str = "/tmp/soma_worker.sock"
    socket_timeout: float = 5.0
    
    # Signal detection
    correction_min_similarity: float = 0.3  # Word overlap threshold for re-prompt detection
    session_end_flush: bool = True
    
    # Buffering
    emit_async: bool = True  # Don't block inference on SOMA
    max_pending_signals: int = 100


_config: SomaHooksConfig | None = None


def get_soma_config() -> SomaHooksConfig:
    """Get SOMA hooks configuration."""
    global _config
    if _config is None:
        _config = SomaHooksConfig()
        # Try to load from ENV
        if os.environ.get("SOMA_HOOKS_ENABLED", "1") == "0":
            _config.enabled = False
        if os.environ.get("SOMA_SOCKET_PATH"):
            _config.socket_path = os.environ["SOMA_SOCKET_PATH"]
    return _config


# ─────────────────────────────────────────────────────────────────────────────
# Signal Types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SomaSignal:
    """A signal to send to SomaWorker."""
    signal_type: str  # CORRECTION | RETRIEVAL_MISS | EXPLICIT | STYLE_SIGNAL
    prompt: str
    target: str
    weight: float = 1.0
    session_id: str | None = None
    memory_context: list[str] | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Session Tracking
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SessionState:
    """Tracks conversation state for signal detection."""
    session_id: str
    messages: list[dict[str, str]]  # role, content pairs
    last_assistant_response: str | None = None
    last_user_prompt: str | None = None
    memory_ids_used: list[str] | None = None
    signal_count: int = 0
    started_at: float = 0.0


_sessions: dict[str, SessionState] = {}
_lock = threading.Lock()


def get_or_create_session(session_id: str) -> SessionState:
    """Get or create session state."""
    with _lock:
        if session_id not in _sessions:
            _sessions[session_id] = SessionState(
                session_id=session_id,
                messages=[],
                started_at=time.time(),
            )
        return _sessions[session_id]


def cleanup_old_sessions(max_age_hours: float = 24.0) -> int:
    """Remove sessions older than max_age_hours."""
    cutoff = time.time() - (max_age_hours * 3600)
    removed = 0
    with _lock:
        to_remove = [
            sid for sid, state in _sessions.items()
            if state.started_at < cutoff
        ]
        for sid in to_remove:
            del _sessions[sid]
            removed += 1
    return removed


# ─────────────────────────────────────────────────────────────────────────────
# IPC Client
# ─────────────────────────────────────────────────────────────────────────────

class SomaIPCClient:
    """Unix socket client for communicating with SomaWorker."""
    
    def __init__(self, socket_path: str, timeout: float = 5.0) -> None:
        self.socket_path = socket_path
        self.timeout = timeout
        self._connected = False
    
    def send_event(self, event_type: str, data: dict[str, Any]) -> dict[str, Any] | None:
        """Send event to SomaWorker and get response."""
        if not Path(self.socket_path).exists():
            log.debug(f"[SOMA IPC] Socket not available: {self.socket_path}")
            return None
        
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect(self.socket_path)
            
            # Send event
            event = {"type": event_type, "data": data}
            sock.sendall(json.dumps(event).encode() + b"\n")
            
            # Get response
            response_data = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response_data += chunk
                if b"\n" in chunk:
                    break
            
            sock.close()
            
            if response_data:
                return json.loads(response_data.decode().strip())
            return None
        
        except socket.timeout:
            log.warning("[SOMA IPC] Request timed out")
            return None
        except Exception as e:
            log.debug(f"[SOMA IPC] Error: {e}")
            return None
    
    def emit_signal(self, signal: SomaSignal) -> bool:
        """Emit a SOMA signal."""
        response = self.send_event("SOMA_SIGNAL", {
            "signal_type": signal.signal_type,
            "prompt": signal.prompt,
            "target": signal.target,
            "weight": signal.weight,
            "session_id": signal.session_id,
            "memory_context": signal.memory_context or [],
        })
        return response is not None and response.get("status") == "buffered"
    
    def session_end(self, session_id: str, signal_count: int) -> dict[str, Any] | None:
        """Notify SomaWorker of session end."""
        return self.send_event("SESSION_END", {
            "session_id": session_id,
            "signal_count": signal_count,
        })
    
    def get_status(self) -> dict[str, Any] | None:
        """Query SomaWorker status."""
        return self.send_event("SOMA_STATUS", {})


_ipc_client: SomaIPCClient | None = None


def get_ipc_client() -> SomaIPCClient:
    """Get or create IPC client."""
    global _ipc_client
    if _ipc_client is None:
        config = get_soma_config()
        _ipc_client = SomaIPCClient(config.socket_path, config.socket_timeout)
    return _ipc_client


# ─────────────────────────────────────────────────────────────────────────────
# Signal Detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_correction(
    current_prompt: str,
    last_prompt: str | None,
    last_response: str | None,
) -> tuple[bool, float]:
    """
    Detect if current_prompt is a correction of last_response.
    
    Returns (is_correction, confidence)
    """
    if not last_prompt or not last_response:
        return False, 0.0
    
    config = get_soma_config()
    
    # Simple heuristic: word overlap between prompts
    current_words = set(current_prompt.lower().split())
    last_words = set(last_prompt.lower().split())
    
    if not last_words:
        return False, 0.0
    
    overlap = len(current_words & last_words) / len(last_words)
    
    # High overlap suggests it's a correction/clarification
    is_correction = overlap >= config.correction_min_similarity
    
    # Look for explicit correction markers
    correction_markers = [
        "no,", "actually", "i meant", "not that", "wrong",
        "correction:", "let me clarify", "to clarify"
    ]
    for marker in correction_markers:
        if marker in current_prompt.lower():
            is_correction = True
            overlap = max(overlap, 0.8)
            break
    
    return is_correction, overlap


def detect_retrieval_miss(
    current_prompt: str,
    last_response: str | None,
    memory_ids_used: list[str] | None,
) -> bool:
    """
    Detect if current_prompt indicates the previous retrieval was wrong.
    
    Indicators:
    - User says "no" or "that's not what I asked"
    - Topic is similar but response was clearly off
    """
    if not last_response:
        return False
    
    miss_markers = [
        "that's not", "not what i", "wrong memory", "different",
        "i was asking about", "no, i meant"
    ]
    
    lower_prompt = current_prompt.lower()
    return any(marker in lower_prompt for marker in miss_markers)


# ─────────────────────────────────────────────────────────────────────────────
# MTPLX Hooks
# ─────────────────────────────────────────────────────────────────────────────

def on_user_message(
    session_id: str,
    user_message: str,
    memory_ids: list[str] | None = None,
) -> None:
    """Called when a user message is received, before generation."""
    config = get_soma_config()
    if not config.enabled:
        return
    
    session = get_or_create_session(session_id)
    
    # Check for correction
    is_correction, confidence = detect_correction(
        user_message,
        session.last_user_prompt,
        session.last_assistant_response,
    )
    
    if is_correction and session.last_assistant_response:
        # Emit CORRECTION signal
        signal = SomaSignal(
            signal_type="CORRECTION",
            prompt=session.last_user_prompt or user_message,
            target=user_message,  # The correction IS the new target
            weight=confidence,
            session_id=session_id,
            memory_context=session.memory_ids_used,
        )
        
        if config.emit_async:
            threading.Thread(
                target=lambda: get_ipc_client().emit_signal(signal),
                daemon=True,
            ).start()
        else:
            get_ipc_client().emit_signal(signal)
        
        session.signal_count += 1
        log.info(f"[SOMA] Emitted CORRECTION signal (confidence={confidence:.2f})")
    
    # Check for retrieval miss
    if detect_retrieval_miss(user_message, session.last_assistant_response, session.memory_ids_used):
        signal = SomaSignal(
            signal_type="RETRIEVAL_MISS",
            prompt=session.last_user_prompt or user_message,
            target=user_message,
            weight=0.7,
            session_id=session_id,
            memory_context=session.memory_ids_used,
        )
        
        if config.emit_async:
            threading.Thread(
                target=lambda: get_ipc_client().emit_signal(signal),
                daemon=True,
            ).start()
        else:
            get_ipc_client().emit_signal(signal)
        
        session.signal_count += 1
        log.info("[SOMA] Emitted RETRIEVAL_MISS signal")
    
    # Update session state
    session.last_user_prompt = user_message
    session.memory_ids_used = memory_ids
    session.messages.append({"role": "user", "content": user_message})


def on_assistant_response(
    session_id: str,
    assistant_response: str,
) -> None:
    """Called after assistant generates a response."""
    config = get_soma_config()
    if not config.enabled:
        return
    
    session = get_or_create_session(session_id)
    session.last_assistant_response = assistant_response
    session.messages.append({"role": "assistant", "content": assistant_response})


def on_session_end(session_id: str) -> dict[str, Any] | None:
    """Called when a session ends. Triggers buffer flush and adaptation."""
    config = get_soma_config()
    if not config.enabled or not config.session_end_flush:
        return None
    
    session = _sessions.get(session_id)
    if not session:
        return None
    
    log.info(f"[SOMA] Session end: {session_id}, signals={session.signal_count}")
    
    # Notify SomaWorker
    result = get_ipc_client().session_end(session_id, session.signal_count)
    
    # Clean up session
    with _lock:
        if session_id in _sessions:
            del _sessions[session_id]
    
    return result


def on_explicit_feedback(
    session_id: str,
    prompt: str,
    ideal_response: str,
    weight: float = 1.0,
) -> bool:
    """Called when Kai provides explicit training feedback."""
    config = get_soma_config()
    if not config.enabled:
        return False
    
    signal = SomaSignal(
        signal_type="EXPLICIT",
        prompt=prompt,
        target=ideal_response,
        weight=weight,
        session_id=session_id,
    )
    
    result = get_ipc_client().emit_signal(signal)
    
    if result:
        session = get_or_create_session(session_id)
        session.signal_count += 1
        log.info("[SOMA] Emitted EXPLICIT signal")
    
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Install Hooks
# ─────────────────────────────────────────────────────────────────────────────

def install_soma_hooks() -> None:
    """Install SOMA hooks into MTPLX server."""
    config = get_soma_config()
    if not config.enabled:
        log.debug("[SOMA] Hooks disabled")
        return
    
    log.info("[SOMA] Installing SOMA hooks")
    
    # The hooks are called manually from openai.py or via middleware
    # This function is a registration point for when we want to auto-patch


def get_hook_stats() -> dict[str, Any]:
    """Get SOMA hook statistics."""
    config = get_soma_config()
    
    total_signals = sum(s.signal_count for s in _sessions.values())
    
    return {
        "enabled": config.enabled,
        "active_sessions": len(_sessions),
        "total_signals": total_signals,
        "socket_path": config.socket_path,
        "worker_status": get_ipc_client().get_status(),
    }


__all__ = [
    "on_user_message",
    "on_assistant_response",
    "on_session_end",
    "on_explicit_feedback",
    "install_soma_hooks",
    "get_hook_stats",
    "SomaSignal",
    "SomaHooksConfig",
    "get_soma_config",
]

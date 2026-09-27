"""ENI Tier 3: Multi-Modal Memory Extension.

Extends the Metal kernel pipeline to handle multiple embedding modalities:

    3.1 Visual Memory — CLIP/SigLIP image embeddings → K/V
    3.2 Code Memory Bank — Code-aware embeddings with AST metadata
    3.3 Audio/Voice Memory — Whisper transcription + embeddings → K/V

Architecture:
    Raw content (image/code/audio)
        ↓ Modality-specific embedding adapter
    Native embedding (512/768/1280-dim)
        ↓ Alignment projector (native → shared 768-dim)
    Shared embedding (stored in pgvector)
        ↓ Per-modality kernel projector (768 → 512 head_dim)
    K/V tensors registered per-layer
        ↓ Metal kernel fused attention

All three modalities are independently toggleable and compose with
Tier 2 (quantization, layer routing, hierarchical clustering).

Dependencies:
    Always available: text + code (via Ollama nomic-embed-text)
    Optional: open_clip (visual), mlx-whisper or whisper (audio)
"""

from __future__ import annotations

import hashlib
import io
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

# Optional: CLIP for visual embeddings
try:
    import open_clip
    import torch
    from PIL import Image
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

# Optional: MLX Whisper for audio
try:
    import mlx_whisper
    WHISPER_MLX_AVAILABLE = True
except ImportError:
    WHISPER_MLX_AVAILABLE = False

# Fallback: OpenAI Whisper
try:
    import whisper as openai_whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

log = logging.getLogger("mtplx.eni.multimodal")


# ═══════════════════════════════════════════════════════════════════════════════
# MODALITY DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

class Modality(str, Enum):
    """Memory modality types."""
    TEXT = "text"
    VISUAL = "visual"
    CODE = "code"
    AUDIO = "audio"


@dataclass
class ModalityConfig:
    """Configuration for a single modality."""
    modality: Modality
    enabled: bool = True
    
    # Embedding model
    model_name: str = ""
    embedding_dim: int = 768
    endpoint: str | None = None  # None = local inference
    
    # Alignment to shared space
    alignment_dim: int = 768     # Shared vector space dimension
    
    # Layer routing override (None = use default from memory_capacity)
    layer_range: tuple[int, int] | None = None


# Default configs per modality
MODALITY_DEFAULTS: dict[Modality, ModalityConfig] = {
    Modality.TEXT: ModalityConfig(
        modality=Modality.TEXT,
        model_name="nomic-embed-text",
        embedding_dim=768,
        endpoint="http://192.168.0.68:11434/api/embeddings",
    ),
    Modality.VISUAL: ModalityConfig(
        modality=Modality.VISUAL,
        model_name="ViT-B-32::laion2b_s34b_b79k",  # open_clip model
        embedding_dim=512,
        endpoint=None,  # Local inference via open_clip
        layer_range=(0, 48),  # Visual → early + middle layers
    ),
    Modality.CODE: ModalityConfig(
        modality=Modality.CODE,
        model_name="nomic-embed-text",  # Handles code well
        embedding_dim=768,
        endpoint="http://192.168.0.68:11434/api/embeddings",
        layer_range=(0, 40),  # Code → early + middle layers
    ),
    Modality.AUDIO: ModalityConfig(
        modality=Modality.AUDIO,
        model_name="whisper-large-v3",
        embedding_dim=768,   # Transcribe → text embed → 768
        endpoint="http://192.168.0.68:11434/api/embeddings",  # For text embedding
        layer_range=(16, 64),  # Audio/conversational → middle + late
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDING ADAPTERS
# ═══════════════════════════════════════════════════════════════════════════════

class BaseEmbeddingAdapter(ABC):
    """Abstract base for modality-specific embedding adapters."""
    
    def __init__(self, config: ModalityConfig):
        self.config = config
        self._initialized = False
        self._embed_count = 0
        self._total_ms = 0.0
    
    @abstractmethod
    def embed(self, content: bytes | str, **kwargs: Any) -> np.ndarray | None:
        """Embed content into a vector.
        
        Args:
            content: Raw content (text string, image bytes, audio bytes)
            
        Returns:
            Embedding vector [embedding_dim] or None on failure
        """
        ...
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this adapter's dependencies are available."""
        ...
    
    def get_stats(self) -> dict:
        return {
            "modality": self.config.modality.value,
            "model": self.config.model_name,
            "available": self.is_available(),
            "embed_count": self._embed_count,
            "avg_ms": round(self._total_ms / max(1, self._embed_count), 2),
        }


class TextEmbeddingAdapter(BaseEmbeddingAdapter):
    """Text embeddings via Ollama nomic-embed-text (768-dim)."""
    
    def __init__(self, config: ModalityConfig | None = None):
        super().__init__(config or MODALITY_DEFAULTS[Modality.TEXT])
        self._http: httpx.Client | None = None
        self._cache: dict[str, np.ndarray] = {}
    
    def is_available(self) -> bool:
        return HTTPX_AVAILABLE and self.config.endpoint is not None
    
    def _get_http(self) -> httpx.Client | None:
        if self._http is None and HTTPX_AVAILABLE:
            self._http = httpx.Client(timeout=30.0)
        return self._http
    
    def embed(self, content: bytes | str, **kwargs: Any) -> np.ndarray | None:
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")
        
        # Cache check
        cache_key = hashlib.md5(content.encode()[:2000]).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        http = self._get_http()
        if http is None:
            return None
        
        t0 = time.perf_counter()
        try:
            resp = http.post(
                self.config.endpoint,
                json={"model": self.config.model_name, "prompt": content[:8000]},
            )
            resp.raise_for_status()
            embedding = np.array(resp.json()["embedding"], dtype=np.float32)
            
            elapsed = (time.perf_counter() - t0) * 1000
            self._embed_count += 1
            self._total_ms += elapsed
            
            # LRU-ish cache
            if len(self._cache) > 500:
                self._cache.clear()
            self._cache[cache_key] = embedding
            
            return embedding
        except Exception as e:
            log.debug(f"[ENI Multimodal] Text embed failed: {e}")
            return None


class VisualEmbeddingAdapter(BaseEmbeddingAdapter):
    """Visual embeddings via CLIP/SigLIP (512-dim for ViT-B/32).
    
    Requires: pip install open_clip_torch Pillow
    
    Supports:
        - Raw image bytes (PNG, JPEG, WebP)
        - PIL Image objects
        - File paths
    """
    
    def __init__(self, config: ModalityConfig | None = None):
        super().__init__(config or MODALITY_DEFAULTS[Modality.VISUAL])
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._device = "cpu"  # MLX models use CPU-side numpy, GPU via MLX later
    
    def is_available(self) -> bool:
        return CLIP_AVAILABLE
    
    def _init_model(self) -> bool:
        if self._initialized:
            return True
        if not CLIP_AVAILABLE:
            log.warning("[ENI Multimodal] open_clip not installed: pip install open_clip_torch Pillow")
            return False
        
        try:
            # Parse model spec: "ViT-B-32::laion2b_s34b_b79k"
            parts = self.config.model_name.split("::")
            model_name = parts[0]
            pretrained = parts[1] if len(parts) > 1 else "laion2b_s34b_b79k"
            
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=self._device
            )
            self._model.eval()
            self._tokenizer = open_clip.get_tokenizer(model_name)
            self._initialized = True
            log.info(f"[ENI Multimodal] CLIP model loaded: {model_name} ({pretrained})")
            return True
        except Exception as e:
            log.warning(f"[ENI Multimodal] CLIP init failed: {e}")
            return False
    
    def embed(self, content: bytes | str, **kwargs: Any) -> np.ndarray | None:
        if not self._init_model():
            return None
        
        t0 = time.perf_counter()
        
        try:
            # Handle different input types
            if isinstance(content, (str, Path)):
                # File path
                image = Image.open(content).convert("RGB")
            elif isinstance(content, bytes):
                # Raw bytes
                image = Image.open(io.BytesIO(content)).convert("RGB")
            else:
                log.warning(f"[ENI Multimodal] Unsupported visual input type: {type(content)}")
                return None
            
            # Preprocess and embed
            image_tensor = self._preprocess(image).unsqueeze(0).to(self._device)
            
            with torch.no_grad():
                image_features = self._model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            embedding = image_features.squeeze().cpu().numpy().astype(np.float32)
            
            elapsed = (time.perf_counter() - t0) * 1000
            self._embed_count += 1
            self._total_ms += elapsed
            
            log.debug(f"[ENI Multimodal] Visual embed: {embedding.shape} in {elapsed:.1f}ms")
            return embedding
            
        except Exception as e:
            log.warning(f"[ENI Multimodal] Visual embed failed: {e}")
            return None
    
    def embed_text_query(self, text: str) -> np.ndarray | None:
        """Embed a text query into CLIP space for cross-modal retrieval.
        
        This enables "find images similar to this text description".
        """
        if not self._init_model():
            return None
        
        try:
            tokens = self._tokenizer([text]).to(self._device)
            with torch.no_grad():
                text_features = self._model.encode_text(tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return text_features.squeeze().cpu().numpy().astype(np.float32)
        except Exception as e:
            log.warning(f"[ENI Multimodal] CLIP text embed failed: {e}")
            return None


class CodeEmbeddingAdapter(BaseEmbeddingAdapter):
    """Code embeddings with AST-aware preprocessing.
    
    Uses nomic-embed-text via Ollama (handles code well) with
    code-specific preprocessing:
        - Language detection tag
        - Import/dependency extraction
        - Docstring emphasis
        - Function signature extraction
    """
    
    def __init__(self, config: ModalityConfig | None = None):
        super().__init__(config or MODALITY_DEFAULTS[Modality.CODE])
        self._text_adapter = TextEmbeddingAdapter(self.config)
    
    def is_available(self) -> bool:
        return self._text_adapter.is_available()
    
    def _preprocess_code(self, code: str, **kwargs: Any) -> str:
        """Code-aware preprocessing for better embedding quality.
        
        Adds structured metadata that the embedding model can use
        to produce more meaningful vectors.
        """
        language = kwargs.get("language", self._detect_language(code))
        filename = kwargs.get("filename", "")
        
        parts = []
        
        # Language tag
        if language:
            parts.append(f"[{language}]")
        
        # Filename context
        if filename:
            parts.append(f"File: {filename}")
        
        # Extract imports (dependency signal)
        imports = self._extract_imports(code, language)
        if imports:
            parts.append(f"Imports: {', '.join(imports[:10])}")
        
        # Extract function/class signatures (structural signal)
        signatures = self._extract_signatures(code, language)
        if signatures:
            parts.append("Signatures: " + "; ".join(signatures[:5]))
        
        # Extract docstrings (semantic signal)
        docstrings = self._extract_docstrings(code)
        if docstrings:
            parts.append("Docs: " + " ".join(docstrings[:3]))
        
        # The code itself (truncated for embedding)
        parts.append(code[:4000])
        
        return "\n".join(parts)
    
    def _detect_language(self, code: str) -> str:
        """Simple heuristic language detection."""
        if "def " in code and ("import " in code or "from " in code):
            return "python"
        if "function " in code or "const " in code or "=>" in code:
            return "javascript"
        if "fn " in code and "let " in code:
            return "rust"
        if "#include" in code:
            return "c++"
        if "func " in code and "package " in code:
            return "go"
        if "public class" in code or "private " in code:
            return "java"
        return "unknown"
    
    def _extract_imports(self, code: str, language: str) -> list[str]:
        """Extract import statements."""
        imports = []
        for line in code.split("\n")[:50]:
            line = line.strip()
            if language == "python":
                if line.startswith("import ") or line.startswith("from "):
                    imports.append(line.split()[1].split(".")[0])
            elif language in ("javascript", "typescript"):
                if "require(" in line or "import " in line:
                    imports.append(line)
        return imports
    
    def _extract_signatures(self, code: str, language: str) -> list[str]:
        """Extract function/class signatures."""
        sigs = []
        for line in code.split("\n"):
            line = line.strip()
            if language == "python":
                if line.startswith("def ") or line.startswith("class "):
                    sigs.append(line.rstrip(":"))
            elif language in ("javascript", "typescript"):
                if line.startswith("function ") or line.startswith("class "):
                    sigs.append(line.split("{")[0].strip())
        return sigs
    
    def _extract_docstrings(self, code: str) -> list[str]:
        """Extract docstrings/comments."""
        docs = []
        in_docstring = False
        current_doc = []
        
        for line in code.split("\n"):
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                if in_docstring:
                    current_doc.append(stripped.rstrip('"""').rstrip("'''"))
                    docs.append(" ".join(current_doc))
                    current_doc = []
                    in_docstring = False
                else:
                    in_docstring = True
                    content = stripped[3:]
                    if content.endswith('"""') or content.endswith("'''"):
                        # Single-line docstring
                        docs.append(content[:-3])
                    else:
                        current_doc.append(content)
            elif in_docstring:
                current_doc.append(stripped)
        
        return docs[:5]
    
    def embed(self, content: bytes | str, **kwargs: Any) -> np.ndarray | None:
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="replace")
        
        # Preprocess with code-aware metadata
        preprocessed = self._preprocess_code(content, **kwargs)
        
        t0 = time.perf_counter()
        result = self._text_adapter.embed(preprocessed)
        elapsed = (time.perf_counter() - t0) * 1000
        
        if result is not None:
            self._embed_count += 1
            self._total_ms += elapsed
        
        return result


class AudioEmbeddingAdapter(BaseEmbeddingAdapter):
    """Audio embeddings via Whisper transcription + text embedding.
    
    Pipeline:
        Audio bytes → Whisper (MLX or OpenAI) → transcript text
        Transcript → nomic-embed-text → 768-dim embedding
    
    Also stores the transcript in memory content for full-text search.
    
    Requires: pip install mlx-whisper  (preferred on Apple Silicon)
              OR: pip install openai-whisper
    """
    
    def __init__(self, config: ModalityConfig | None = None):
        super().__init__(config or MODALITY_DEFAULTS[Modality.AUDIO])
        self._text_adapter = TextEmbeddingAdapter(
            ModalityConfig(
                modality=Modality.TEXT,
                model_name="nomic-embed-text",
                embedding_dim=768,
                endpoint=self.config.endpoint,
            )
        )
        self._whisper_model = None
    
    def is_available(self) -> bool:
        return (WHISPER_MLX_AVAILABLE or WHISPER_AVAILABLE) and self._text_adapter.is_available()
    
    def _init_whisper(self) -> bool:
        if self._initialized:
            return True
        
        if WHISPER_MLX_AVAILABLE:
            # MLX Whisper — native Apple Silicon, fastest
            self._whisper_backend = "mlx"
            self._initialized = True
            log.info("[ENI Multimodal] Whisper backend: mlx-whisper")
            return True
        
        if WHISPER_AVAILABLE:
            # OpenAI Whisper — CPU/torch fallback
            try:
                self._whisper_model = openai_whisper.load_model("base")
                self._whisper_backend = "openai"
                self._initialized = True
                log.info("[ENI Multimodal] Whisper backend: openai-whisper (base)")
                return True
            except Exception as e:
                log.warning(f"[ENI Multimodal] Whisper init failed: {e}")
        
        log.warning("[ENI Multimodal] No whisper backend: pip install mlx-whisper")
        return False
    
    def transcribe(self, audio: bytes | str) -> str | None:
        """Transcribe audio to text.
        
        Args:
            audio: Audio bytes or file path
            
        Returns:
            Transcript text or None on failure
        """
        if not self._init_whisper():
            return None
        
        try:
            # Write bytes to temp file if needed
            if isinstance(audio, bytes):
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(audio)
                    audio_path = f.name
            else:
                audio_path = str(audio)
            
            if self._whisper_backend == "mlx":
                result = mlx_whisper.transcribe(
                    audio_path,
                    path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
                )
                return result.get("text", "")
            
            elif self._whisper_backend == "openai":
                result = self._whisper_model.transcribe(audio_path)
                return result.get("text", "")
            
            return None
            
        except Exception as e:
            log.warning(f"[ENI Multimodal] Transcription failed: {e}")
            return None
    
    def embed(self, content: bytes | str, **kwargs: Any) -> np.ndarray | None:
        """Embed audio by transcribing first, then embedding the text.
        
        Returns text embedding of the transcript.
        Also stores transcript in kwargs['_transcript'] for caller to use.
        """
        t0 = time.perf_counter()
        
        # Step 1: Transcribe
        transcript = kwargs.get("transcript")  # Pre-transcribed text
        if transcript is None:
            transcript = self.transcribe(content)
        
        if not transcript:
            log.debug("[ENI Multimodal] No transcript from audio")
            return None
        
        # Store transcript for caller
        kwargs["_transcript"] = transcript
        
        # Step 2: Embed the transcript text
        embedding = self._text_adapter.embed(transcript)
        
        elapsed = (time.perf_counter() - t0) * 1000
        if embedding is not None:
            self._embed_count += 1
            self._total_ms += elapsed
            log.debug(
                f"[ENI Multimodal] Audio embed: transcribed "
                f"{len(transcript)} chars in {elapsed:.0f}ms"
            )
        
        return embedding


# ═══════════════════════════════════════════════════════════════════════════════
# ALIGNMENT PROJECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class AlignmentProjector:
    """Projects modality-specific embeddings to shared 768-dim space.
    
    Embeddings from different models live in incompatible vector spaces.
    This projector learns a linear map from each modality's native space
    to a shared 768-dim space for unified pgvector storage and retrieval.
    
    For modalities already at 768-dim (text, code), this is identity.
    For visual (CLIP 512-dim), this is a learned 512→768 expansion.
    For audio with direct encoder output, this would be 1280→768 compression.
    """
    
    def __init__(self):
        self._projections: dict[Modality, np.ndarray | None] = {}
        self._initialized: set[Modality] = set()
    
    def _init_projection(self, modality: Modality) -> None:
        if modality in self._initialized:
            return
        
        config = MODALITY_DEFAULTS.get(modality)
        if config is None:
            return
        
        native_dim = config.embedding_dim
        shared_dim = config.alignment_dim
        
        if native_dim == shared_dim:
            # Identity — no projection needed
            self._projections[modality] = None
        else:
            # Xavier init linear projection
            scale = (2.0 / (native_dim + shared_dim)) ** 0.5
            self._projections[modality] = (
                np.random.randn(native_dim, shared_dim).astype(np.float32) * scale
            )
            log.info(
                f"[ENI Multimodal] Alignment: {modality.value} "
                f"{native_dim} → {shared_dim}"
            )
        
        self._initialized.add(modality)
    
    def align(
        self,
        embedding: np.ndarray,
        modality: Modality,
    ) -> np.ndarray:
        """Project embedding to shared 768-dim space.
        
        Args:
            embedding: Native modality embedding [native_dim]
            modality: Source modality
            
        Returns:
            Aligned embedding [768]
        """
        self._init_projection(modality)
        
        proj = self._projections.get(modality)
        if proj is None:
            # Identity — already in shared space
            return embedding
        
        # Linear projection: [native_dim] @ [native_dim, shared_dim] = [shared_dim]
        aligned = embedding @ proj
        
        # L2 normalize for cosine similarity compatibility
        norm = np.linalg.norm(aligned)
        if norm > 0:
            aligned = aligned / norm
        
        return aligned.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# PER-MODALITY KERNEL PROJECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class MultiModalKernelProjector:
    """Per-modality K/V projection for the Metal kernel.
    
    Different modalities need different projection weights because their
    embedding spaces encode fundamentally different semantic structures.
    An image embedding captures spatial/visual features; a code embedding
    captures structural/syntactic features. The model needs different
    W_k/W_v matrices to map these into useful attention keys/values.
    
    This extends MemoryProjector with modality-aware weight selection.
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        head_dim: int = 512,
        n_kv_heads: int = 1,
        n_layers: int = 64,
    ):
        self.embedding_dim = embedding_dim
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads
        self.n_layers = n_layers
        
        # Per-modality shared projection weights
        # {modality: (W_k, W_v)} where W_k/W_v are [embedding_dim, head_dim]
        self._weights: dict[Modality, tuple[Any, Any]] = {}
        self._initialized: set[Modality] = set()
    
    def _init_modality(self, modality: Modality) -> None:
        if modality in self._initialized:
            return
        
        if not MLX_AVAILABLE:
            return
        
        scale = (2.0 / (self.embedding_dim + self.head_dim)) ** 0.5
        w_k = mx.random.normal(shape=(self.embedding_dim, self.head_dim)) * scale
        w_v = mx.random.normal(shape=(self.embedding_dim, self.head_dim)) * scale
        
        self._weights[modality] = (w_k, w_v)
        self._initialized.add(modality)
        log.info(
            f"[ENI Multimodal] Kernel projector for {modality.value}: "
            f"{self.embedding_dim} → {self.head_dim}"
        )
    
    def project(
        self,
        embeddings: np.ndarray,
        modality: Modality,
    ) -> dict[int, tuple[Any, Any]]:
        """Project modality-specific embeddings to per-layer K/V.
        
        Args:
            embeddings: [N_mem, embedding_dim] in shared space
            modality: Source modality (selects W_k/W_v weights)
            
        Returns:
            Dict mapping layer_idx → (keys, values) for registry
        """
        if not MLX_AVAILABLE:
            return {}
        
        self._init_modality(modality)
        
        w_k, w_v = self._weights[modality]
        emb_mx = mx.array(embeddings, dtype=mx.float32)
        
        n_mem = emb_mx.shape[0]
        
        # Shared projection across all layers for this modality
        keys = (emb_mx @ w_k).reshape(1, n_mem, self.n_kv_heads, self.head_dim)
        values = (emb_mx @ w_v).reshape(1, n_mem, self.n_kv_heads, self.head_dim)
        mx.eval(keys, values)
        
        # Apply modality-specific layer routing
        config = MODALITY_DEFAULTS.get(modality)
        if config and config.layer_range:
            lo, hi = config.layer_range
            return {i: (keys, values) for i in range(lo, min(hi, self.n_layers))}
        
        # Default: all layers
        return {i: (keys, values) for i in range(self.n_layers)}
    
    def get_stats(self) -> dict:
        return {
            "initialized_modalities": [m.value for m in self._initialized],
            "embedding_dim": self.embedding_dim,
            "head_dim": self.head_dim,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA MIGRATION
# ═══════════════════════════════════════════════════════════════════════════════

MIGRATION_SQL = """
-- Add modality column if not exists
DO $$ 
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'memories' AND column_name = 'modality'
    ) THEN
        ALTER TABLE memories ADD COLUMN modality VARCHAR(16) DEFAULT 'text';
        CREATE INDEX idx_memories_modality ON memories(modality);
        
        -- Backfill: all existing memories are text
        UPDATE memories SET modality = 'text' WHERE modality IS NULL;
    END IF;
END $$;

-- Add source_file column for visual/audio original paths
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'memories' AND column_name = 'source_file'
    ) THEN
        ALTER TABLE memories ADD COLUMN source_file TEXT;
    END IF;
END $$;

-- Add transcript column for audio memories
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'memories' AND column_name = 'transcript'
    ) THEN
        ALTER TABLE memories ADD COLUMN transcript TEXT;
    END IF;
END $$;
"""


def migrate_multimodal_schema(client: Any) -> bool:
    """Run schema migration for multi-modal support.
    
    Adds:
        - modality column (text/visual/code/audio)
        - source_file column (path to original file)
        - transcript column (audio transcriptions)
    
    Args:
        client: ENIMemoryClient with _pool
        
    Returns:
        True if migration succeeded
    """
    if not hasattr(client, "_pool") or client._pool is None:
        return False
    
    try:
        with client._pool.connection() as conn:
            conn.execute(MIGRATION_SQL)
            conn.commit()
        log.info("[ENI Multimodal] Schema migration complete")
        return True
    except Exception as e:
        log.warning(f"[ENI Multimodal] Schema migration failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-MODAL INGESTION
# ═══════════════════════════════════════════════════════════════════════════════

class MultiModalIngester:
    """Ingest content of any modality into the memory graph.
    
    Pipeline per modality:
        1. Modality-specific embedding adapter generates native embedding
        2. Alignment projector maps to shared 768-dim space
        3. Store in PostgreSQL with modality tag + aligned embedding
    """
    
    def __init__(self):
        self._adapters: dict[Modality, BaseEmbeddingAdapter] = {
            Modality.TEXT: TextEmbeddingAdapter(),
            Modality.VISUAL: VisualEmbeddingAdapter(),
            Modality.CODE: CodeEmbeddingAdapter(),
            Modality.AUDIO: AudioEmbeddingAdapter(),
        }
        self._aligner = AlignmentProjector()
        self._ingest_count = 0
    
    def get_adapter(self, modality: Modality) -> BaseEmbeddingAdapter:
        return self._adapters[modality]
    
    def embed_and_align(
        self,
        content: bytes | str,
        modality: Modality,
        **kwargs: Any,
    ) -> np.ndarray | None:
        """Embed content and align to shared space.
        
        Returns:
            768-dim aligned embedding or None
        """
        adapter = self._adapters.get(modality)
        if adapter is None or not adapter.is_available():
            log.warning(f"[ENI Multimodal] {modality.value} adapter not available")
            return None
        
        # Get native embedding
        native_emb = adapter.embed(content, **kwargs)
        if native_emb is None:
            return None
        
        # Align to shared space
        aligned = self._aligner.align(native_emb, modality)
        return aligned
    
    def ingest(
        self,
        client: Any,
        content: bytes | str,
        modality: Modality,
        title: str,
        category: str = "note",
        importance: float = 0.6,
        tags: list[str] | None = None,
        source_file: str | None = None,
        **kwargs: Any,
    ) -> str | None:
        """Ingest content into the memory graph with modality-aware embedding.
        
        Args:
            client: ENIMemoryClient
            content: Raw content (text, image bytes, code, audio bytes)
            modality: Content modality
            title: Memory title
            category: Memory category (fact, note, code, etc.)
            importance: 0.0-1.0
            tags: Optional tags
            source_file: Original file path (for images/audio)
            
        Returns:
            Memory ID or None on failure
        """
        if not hasattr(client, "_pool") or client._pool is None:
            return None
        
        # Step 1: Embed and align
        aligned_embedding = self.embed_and_align(content, modality, **kwargs)
        if aligned_embedding is None:
            return None
        
        # Step 2: Prepare content text
        if modality == Modality.VISUAL:
            # For images, content is a description/caption
            content_text = kwargs.get("caption", f"[Image: {title}]")
            if isinstance(content_text, bytes):
                content_text = f"[Image: {title}]"
        elif modality == Modality.AUDIO:
            # For audio, content is the transcript
            content_text = kwargs.get("_transcript", f"[Audio: {title}]")
        elif isinstance(content, bytes):
            content_text = content.decode("utf-8", errors="replace")
        else:
            content_text = content
        
        # Step 3: Store in PostgreSQL
        embedding_str = "[" + ",".join(str(x) for x in aligned_embedding) + "]"
        
        import uuid
        memory_id = f"{category[:3]}-{title[:30].lower().replace(' ', '-')}-{uuid.uuid4().hex[:8]}"
        
        tags_str = ",".join(tags) if tags else ""
        
        sql = """
            INSERT INTO memories (id, title, content, category, importance, 
                                  tags, embedding, modality, source_file, transcript,
                                  created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s, %s, %s, 
                    NOW(), NOW())
            ON CONFLICT (id) DO UPDATE SET
                content = EXCLUDED.content,
                embedding = EXCLUDED.embedding,
                modality = EXCLUDED.modality,
                updated_at = NOW()
            RETURNING id
        """
        
        transcript = kwargs.get("_transcript")
        
        try:
            with client._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (
                        memory_id, title, content_text, category,
                        importance, tags_str, embedding_str,
                        modality.value, source_file, transcript,
                    ))
                    result = cur.fetchone()
                conn.commit()
            
            self._ingest_count += 1
            log.info(
                f"[ENI Multimodal] Ingested {modality.value} memory: "
                f"{memory_id} ({len(content_text)} chars)"
            )
            return result[0] if result else memory_id
            
        except Exception as e:
            log.warning(f"[ENI Multimodal] Ingest failed: {e}")
            return None
    
    def ingest_image(
        self,
        client: Any,
        image: bytes | str,
        title: str,
        caption: str = "",
        **kwargs: Any,
    ) -> str | None:
        """Convenience: ingest an image memory."""
        source_file = str(image) if isinstance(image, (str, Path)) else None
        return self.ingest(
            client, image, Modality.VISUAL, title,
            category="note", caption=caption, source_file=source_file, **kwargs,
        )
    
    def ingest_code(
        self,
        client: Any,
        code: str,
        title: str,
        language: str = "",
        filename: str = "",
        **kwargs: Any,
    ) -> str | None:
        """Convenience: ingest a code memory."""
        return self.ingest(
            client, code, Modality.CODE, title,
            category="code", language=language, filename=filename,
            source_file=filename or None, **kwargs,
        )
    
    def ingest_audio(
        self,
        client: Any,
        audio: bytes | str,
        title: str,
        **kwargs: Any,
    ) -> str | None:
        """Convenience: ingest an audio memory."""
        source_file = str(audio) if isinstance(audio, (str, Path)) else None
        return self.ingest(
            client, audio, Modality.AUDIO, title,
            category="note", source_file=source_file, **kwargs,
        )
    
    def get_stats(self) -> dict:
        return {
            "ingest_count": self._ingest_count,
            "adapters": {
                m.value: adapter.get_stats()
                for m, adapter in self._adapters.items()
            },
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-MODAL RETRIEVAL
# ═══════════════════════════════════════════════════════════════════════════════

class MultiModalRetriever:
    """Retrieve memories across modalities for the Metal kernel.
    
    Extends the flat/hierarchical retrieval from kernel_integration.py
    with modality-aware filtering and per-modality kernel projection.
    """
    
    def __init__(self):
        self._kernel_projector = MultiModalKernelProjector()
        self._retrieval_count = 0
    
    def retrieve_with_modalities(
        self,
        client: Any,
        query: str,
        modalities: list[Modality] | None = None,
        k: int = 10,
        min_similarity: float = 0.3,
    ) -> tuple[list[Any], np.ndarray | None, list[Modality]]:
        """Retrieve memories with modality information.
        
        Args:
            client: ENIMemoryClient
            query: Query text
            modalities: Filter to specific modalities (None = all)
            k: Number of memories
            min_similarity: Minimum cosine similarity
            
        Returns:
            (memories, embeddings, modality_tags) — modality_tags[i] is
            the modality of memories[i]
        """
        if not hasattr(client, "_pool") or client._pool is None:
            return [], None, []
        
        # Get query embedding
        query_embedding = client._embed_text(query)
        if query_embedding is None:
            return [], None, []
        
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
        
        # Build modality filter
        modality_filter = ""
        params: list[Any] = [embedding_str, embedding_str]
        
        if modalities:
            modality_values = [m.value for m in modalities]
            modality_filter = "AND modality = ANY(%s)"
            params.append(modality_values)
        
        params.append(k * 2)
        
        sql = f"""
            SELECT
                id, title, content, category, importance, modality,
                1 - (embedding <=> %s::vector) AS similarity,
                embedding::text AS embedding_text
            FROM memories
            WHERE embedding IS NOT NULL
                {modality_filter}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        
        try:
            from psycopg.rows import dict_row
            from .memory_client import Memory
            
            with client._pool.connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(sql, params)
                    rows = cur.fetchall()
            
            memories = []
            embeddings_list = []
            modality_tags = []
            
            for row in rows:
                sim = float(row["similarity"])
                if sim >= min_similarity:
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
                    
                    # Parse modality
                    mod_str = row.get("modality", "text") or "text"
                    try:
                        modality_tags.append(Modality(mod_str))
                    except ValueError:
                        modality_tags.append(Modality.TEXT)
                
                if len(memories) >= k:
                    break
            
            embeddings = np.array(embeddings_list, dtype=np.float32) if embeddings_list else None
            self._retrieval_count += 1
            
            return memories, embeddings, modality_tags
            
        except Exception as e:
            log.warning(f"[ENI Multimodal] Multi-modal retrieval failed: {e}")
            return [], None, []
    
    def project_for_kernel(
        self,
        embeddings: np.ndarray,
        modality_tags: list[Modality],
    ) -> dict[int, tuple[Any, Any]]:
        """Project mixed-modality embeddings to per-layer K/V.
        
        Groups memories by modality and applies modality-specific
        projection weights, then merges the K/V tensors.
        
        Args:
            embeddings: [N_mem, 768] aligned embeddings
            modality_tags: Modality for each memory
            
        Returns:
            Dict mapping layer_idx → (merged_keys, merged_values)
        """
        if not MLX_AVAILABLE:
            return {}
        
        # Group by modality
        modality_groups: dict[Modality, list[int]] = {}
        for i, mod in enumerate(modality_tags):
            modality_groups.setdefault(mod, []).append(i)
        
        # Project each group and merge
        all_layer_kvs: dict[int, list[tuple[Any, Any]]] = {}
        
        for modality, indices in modality_groups.items():
            group_embeddings = embeddings[indices]
            layer_kv_map = self._kernel_projector.project(group_embeddings, modality)
            
            for layer_idx, (k, v) in layer_kv_map.items():
                all_layer_kvs.setdefault(layer_idx, []).append((k, v))
        
        # Merge K/V tensors across modalities per layer
        merged: dict[int, tuple[Any, Any]] = {}
        for layer_idx, kv_list in all_layer_kvs.items():
            if len(kv_list) == 1:
                merged[layer_idx] = kv_list[0]
            else:
                # Concatenate along memory dimension (dim 1)
                all_keys = mx.concatenate([kv[0] for kv in kv_list], axis=1)
                all_values = mx.concatenate([kv[1] for kv in kv_list], axis=1)
                merged[layer_idx] = (all_keys, all_values)
        
        return merged
    
    def get_stats(self) -> dict:
        return {
            "retrieval_count": self._retrieval_count,
            "kernel_projector": self._kernel_projector.get_stats(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCES + CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MultiModalConfig:
    """Master config for Tier 3 multi-modal extension."""
    enabled: bool = True
    visual_enabled: bool = True
    code_enabled: bool = True
    audio_enabled: bool = True


_multimodal_config: MultiModalConfig = MultiModalConfig()
_ingester: MultiModalIngester | None = None
_retriever: MultiModalRetriever | None = None


def get_multimodal_config() -> MultiModalConfig:
    return _multimodal_config


def configure_multimodal(
    enabled: bool = True,
    visual: bool = True,
    code: bool = True,
    audio: bool = True,
) -> MultiModalConfig:
    global _multimodal_config
    _multimodal_config = MultiModalConfig(
        enabled=enabled,
        visual_enabled=visual,
        code_enabled=code,
        audio_enabled=audio,
    )
    return _multimodal_config


def get_ingester() -> MultiModalIngester:
    global _ingester
    if _ingester is None:
        _ingester = MultiModalIngester()
    return _ingester


def get_retriever() -> MultiModalRetriever:
    global _retriever
    if _retriever is None:
        _retriever = MultiModalRetriever()
    return _retriever


def get_multimodal_stats() -> dict:
    """Get stats for all multi-modal systems."""
    return {
        "config": {
            "enabled": _multimodal_config.enabled,
            "visual": _multimodal_config.visual_enabled,
            "code": _multimodal_config.code_enabled,
            "audio": _multimodal_config.audio_enabled,
        },
        "adapters": {
            Modality.TEXT.value: TextEmbeddingAdapter().is_available(),
            Modality.VISUAL.value: CLIP_AVAILABLE,
            Modality.CODE.value: TextEmbeddingAdapter().is_available(),
            Modality.AUDIO.value: WHISPER_MLX_AVAILABLE or WHISPER_AVAILABLE,
        },
        "ingester": get_ingester().get_stats() if _ingester else None,
        "retriever": get_retriever().get_stats() if _retriever else None,
    }

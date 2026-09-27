"""ENI Memory Client - PostgreSQL connection to memory graph.

Provides:
- Connection pooling to CT268 PostgreSQL
- HNSW vector search for semantic retrieval
- Full-text search fallback
- LRU caching for hot memories
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

try:
    import psycopg
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool
    PSYCOPG_AVAILABLE = True
except ImportError:
    PSYCOPG_AVAILABLE = False

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from .config import get_config, ENIConfig


log = logging.getLogger("mtplx.eni.memory")


@dataclass
class Memory:
    """A memory from the graph."""
    id: str
    title: str
    content: str
    category: str
    importance: float
    similarity: float = 0.0  # Filled by retrieval


class ENIMemoryClient:
    """Client for ENI memory graph in PostgreSQL."""
    
    def __init__(self, config: ENIConfig | None = None):
        self.config = config or get_config()
        self._pool: ConnectionPool | None = None
        self._http: httpx.Client | None = None
        self._embedding_cache: dict[str, np.ndarray] = {}
        self._connected = False
        
        if not PSYCOPG_AVAILABLE:
            log.warning("[ENI] psycopg3 not available - memory client disabled")
            return
        
        try:
            self._pool = ConnectionPool(
                self.config.pg_dsn,
                min_size=1,
                max_size=4,
                timeout=10.0,
            )
            self._connected = True
            log.info(f"[ENI] Memory client connected to {self.config.pg_dsn.split('@')[1]}")
        except Exception as e:
            log.warning(f"[ENI] Failed to connect to PostgreSQL: {e}")
    
    def is_connected(self) -> bool:
        """Check if connected to database."""
        return self._connected and self._pool is not None
    
    def close(self) -> None:
        """Close connections."""
        if self._pool:
            self._pool.close()
            self._pool = None
        if self._http:
            self._http.close()
            self._http = None
        self._connected = False
    
    def _get_http(self) -> httpx.Client:
        """Get or create HTTP client for embeddings."""
        if self._http is None and HTTPX_AVAILABLE:
            self._http = httpx.Client(timeout=30.0)
        return self._http
    
    def _embed_text(self, text: str) -> np.ndarray | None:
        """Get embedding for text using Ollama."""
        # Check cache
        cache_key = hashlib.md5(text.encode()[:1000]).hexdigest()
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        http = self._get_http()
        if http is None:
            return None
        
        try:
            resp = http.post(
                self.config.embedding_endpoint,
                json={"model": self.config.embedding_model, "prompt": text[:8000]}
            )
            resp.raise_for_status()
            embedding = np.array(resp.json()["embedding"], dtype=np.float32)
            
            # Cache (LRU-ish: clear if too big)
            if len(self._embedding_cache) > 1000:
                self._embedding_cache.clear()
            self._embedding_cache[cache_key] = embedding
            
            return embedding
        except Exception as e:
            log.debug(f"[ENI] Embedding failed: {e}")
            return None
    
    def retrieve_semantic(
        self,
        query: str,
        k: int | None = None,
        min_similarity: float | None = None,
    ) -> list[Memory]:
        """Retrieve memories using semantic (HNSW) search."""
        if not self.is_connected():
            return []
        
        k = k or self.config.memory_k
        min_similarity = min_similarity or self.config.memory_min_similarity
        
        # Get query embedding
        query_embedding = self._embed_text(query)
        if query_embedding is None:
            log.debug("[ENI] No embedding, falling back to FTS")
            return self.retrieve_fts(query, k) if self.config.fallback_to_fts else []
        
        # Format for PostgreSQL pgvector
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
        
        sql = """
            SELECT 
                id, title, content, category, importance,
                1 - (embedding <=> %s::vector) AS similarity
            FROM memories
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        
        try:
            with self._pool.connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(sql, (embedding_str, embedding_str, k * 2))
                    rows = cur.fetchall()
            
            memories = []
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
                if len(memories) >= k:
                    break
            
            return memories
        except Exception as e:
            log.warning(f"[ENI] Semantic search failed: {e}")
            return self.retrieve_fts(query, k) if self.config.fallback_to_fts else []
    
    def retrieve_fts(self, query: str, k: int | None = None) -> list[Memory]:
        """Retrieve memories using full-text search (fallback)."""
        if not self.is_connected():
            return []
        
        k = k or self.config.memory_k
        
        # Build FTS query - simple word-based
        words = [w.strip() for w in query.split()[:10] if len(w) > 2]
        if not words:
            return []
        
        fts_query = " | ".join(words)
        
        sql = """
            SELECT 
                id, title, content, category, importance,
                ts_rank(fts_vector, websearch_to_tsquery('english', %s)) AS rank
            FROM memories
            WHERE fts_vector @@ websearch_to_tsquery('english', %s)
            ORDER BY importance DESC, rank DESC
            LIMIT %s
        """
        
        try:
            with self._pool.connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(sql, (fts_query, fts_query, k))
                    rows = cur.fetchall()
            
            return [
                Memory(
                    id=row["id"],
                    title=row["title"],
                    content=row["content"] or "",
                    category=row["category"] or "note",
                    importance=float(row["importance"] or 0.5),
                    similarity=float(row["rank"]) if row["rank"] else 0.0,
                )
                for row in rows
            ]
        except Exception as e:
            log.warning(f"[ENI] FTS search failed: {e}")
            return []
    
    def retrieve(
        self,
        query: str,
        k: int | None = None,
        min_similarity: float | None = None,
    ) -> list[Memory]:
        """Retrieve memories using best available method."""
        if self.config.use_semantic_index:
            return self.retrieve_semantic(query, k, min_similarity)
        return self.retrieve_fts(query, k)
    
    def get_memory(self, memory_id: str) -> Memory | None:
        """Get a specific memory by ID."""
        if not self.is_connected():
            return None
        
        sql = "SELECT id, title, content, category, importance FROM memories WHERE id = %s"
        
        try:
            with self._pool.connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(sql, (memory_id,))
                    row = cur.fetchone()
            
            if row:
                return Memory(
                    id=row["id"],
                    title=row["title"],
                    content=row["content"] or "",
                    category=row["category"] or "note",
                    importance=float(row["importance"] or 0.5),
                )
            return None
        except Exception as e:
            log.warning(f"[ENI] Get memory failed: {e}")
            return None
    
    def retrieve_with_embeddings(
        self,
        query: str,
        k: int | None = None,
        min_similarity: float | None = None,
    ) -> tuple[list[Memory], np.ndarray | None]:
        """Retrieve memories with their embeddings for Metal kernel projection.
        
        Returns:
            Tuple of (memories, embeddings) where embeddings is [N, 768] array
            or None if retrieval failed.
        """
        if not self.is_connected():
            return [], None
        
        k = k or self.config.memory_k
        min_similarity = min_similarity or self.config.memory_min_similarity
        
        # Get query embedding
        query_embedding = self._embed_text(query)
        if query_embedding is None:
            log.debug("[ENI] No embedding for retrieve_with_embeddings")
            return [], None
        
        # Format for PostgreSQL pgvector
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
        
        # Query with embeddings returned
        sql = """
            SELECT
                id, title, content, category, importance,
                1 - (embedding <=> %s::vector) AS similarity,
                embedding::text AS embedding_text
            FROM memories
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        
        try:
            with self._pool.connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(sql, (embedding_str, embedding_str, k * 2))
                    rows = cur.fetchall()
            
            memories = []
            embeddings_list = []
            
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
                    
                    # Parse embedding from PostgreSQL text format: [0.1,0.2,...]
                    emb_text = row["embedding_text"]
                    if emb_text:
                        emb_values = [float(x) for x in emb_text.strip("[]").split(",")]
                        embeddings_list.append(emb_values)
                    
                if len(memories) >= k:
                    break
            
            if not memories:
                return [], None
            
            # Stack embeddings into numpy array [N, embedding_dim]
            embeddings = np.array(embeddings_list, dtype=np.float32)
            
            log.debug(f"[ENI] Retrieved {len(memories)} memories with embeddings shape {embeddings.shape}")
            return memories, embeddings
            
        except Exception as e:
            log.warning(f"[ENI] retrieve_with_embeddings failed: {e}")
            return [], None
    
    def stats(self) -> dict[str, Any]:
        """Get memory graph statistics."""
        if not self.is_connected():
            return {"connected": False}
        
        try:
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM memories")
                    total = cur.fetchone()[0]
                    
                    cur.execute("SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL")
                    embedded = cur.fetchone()[0]
            
            return {
                "connected": True,
                "total_memories": total,
                "embedded_memories": embedded,
                "embedding_cache_size": len(self._embedding_cache),
            }
        except Exception as e:
            return {"connected": False, "error": str(e)}


# Global client instance
_client: ENIMemoryClient | None = None


def get_client() -> ENIMemoryClient:
    """Get or create the global memory client."""
    global _client
    if _client is None:
        _client = ENIMemoryClient()
    return _client


def close_client() -> None:
    """Close the global memory client."""
    global _client
    if _client:
        _client.close()
        _client = None

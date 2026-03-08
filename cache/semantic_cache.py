import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    query: str
    result: str
    embedding: np.ndarray
    cluster_probs: np.ndarray
    dominant_cluster: int
    timestamp: float = field(default_factory=time.time)


class SemanticCache:
    """
    Built from scratch — no Redis, no caching library.

    Stores entries in a plain dict. The cluster index maps each cluster_id
    to a list of (query, embedding) pairs stored there. On lookup we only
    compare against entries in the same dominant cluster — reduces scan
    from O(N) to O(N/K). Falls back to full scan if the cluster is empty.

    The similarity threshold is the main thing to tune. At 0.85 it only
    catches clear paraphrases. At 0.75 it starts catching topically related
    queries. Below 0.70 it over-caches — queries get hits they shouldn't.
    0.75 felt like the right balance for this dataset after testing.
    """

    def __init__(self, similarity_threshold: float = 0.75):
        self.similarity_threshold = similarity_threshold
        self._entries: dict = {}
        self._cluster_index: dict = defaultdict(list)
        self._hit_count = 0
        self._miss_count = 0
        self._lock = threading.RLock()

    def lookup(self, query: str, query_embedding: np.ndarray, cluster_probs: np.ndarray) -> Optional[tuple]:
        with self._lock:
            # exact match first
            if query in self._entries:
                self._hit_count += 1
                logger.info(f"EXACT HIT: '{query}'")
                return self._entries[query], 1.0

            # search top 2 clusters to handle boundary docs
            top2 = sorted(enumerate(cluster_probs.tolist()), key=lambda x: -x[1])[:2]
            search_clusters = [c for c, p in top2 if p > 0.1]

            candidates = []
            for cid in search_clusters:
                candidates.extend(self._cluster_index.get(cid, []))

            # fallback to full scan if cluster is empty (small cache)
            if not candidates:
                candidates = [(q, e.embedding) for q, e in self._entries.items()]

            best_entry, best_score = None, 0.0
            for cached_query, cached_embedding in candidates:
                score = self._cosine_similarity(query_embedding, cached_embedding)
                logger.info(f"  '{query}' vs '{cached_query}' => {score:.4f}")
                if score > best_score:
                    best_score = score
                    best_entry = self._entries.get(cached_query)

            if best_entry and best_score >= self.similarity_threshold:
                self._hit_count += 1
                logger.info(f"SEMANTIC HIT: '{query}' ~ '{best_entry.query}' score={best_score:.4f}")
                return best_entry, best_score

            self._miss_count += 1
            logger.info(f"MISS: '{query}' best={best_score:.4f} threshold={self.similarity_threshold}")
            return None

    def store(self, query: str, result: str, query_embedding: np.ndarray, cluster_probs: np.ndarray) -> CacheEntry:
        dominant_cluster = int(np.argmax(cluster_probs))
        entry = CacheEntry(
            query=query,
            result=result,
            embedding=query_embedding.copy(),
            cluster_probs=cluster_probs.copy(),
            dominant_cluster=dominant_cluster,
        )
        with self._lock:
            self._entries[query] = entry
            self._cluster_index[dominant_cluster].append((query, query_embedding.copy()))
        logger.info(f"STORED: '{query}' -> cluster {dominant_cluster}")
        return entry

    def flush(self) -> None:
        with self._lock:
            self._entries.clear()
            self._cluster_index.clear()
            self._hit_count = 0
            self._miss_count = 0
        logger.info("Cache flushed.")

    @property
    def stats(self) -> dict:
        with self._lock:
            total = self._hit_count + self._miss_count
            return {
                "total_entries": len(self._entries),
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "hit_rate": round(self._hit_count / total, 3) if total > 0 else 0.0,
                "similarity_threshold": self.similarity_threshold,
                "entries_per_cluster": {str(k): len(v) for k, v in self._cluster_index.items()},
            }

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))
"""
FastAPI service with semantic search + cache endpoints.
State (model, cache, DB) loaded once at startup via lifespan.
"""
import json
from pathlib import Path
import logging
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from cache.semantic_cache import SemanticCache
from clustering.fuzzy_cluster import get_query_cluster_probs
from embeddings.setup_vectordb import query_similar, EMBED_MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_model: SentenceTransformer = None
_cache: SemanticCache = None
SIMILARITY_THRESHOLD = 0.75


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _cache
    logger.info("Loading embedding model...")
    _model = SentenceTransformer(EMBED_MODEL)
    logger.info("Initialising semantic cache...")
    _cache = SemanticCache(similarity_threshold=SIMILARITY_THRESHOLD)
    logger.info("API ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Trademarkia Semantic Search",
    description="Semantic search over 20 Newsgroups with fuzzy clustering and semantic cache",
    version="1.0.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: Optional[str]
    similarity_score: Optional[float] 
    result: str
    dominant_cluster: int
    


class CacheStatsResponse(BaseModel):
    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float


def embed_query(query: str) -> np.ndarray:
    return _model.encode(query, show_progress_bar=False)


def compute_result(query_embedding: np.ndarray, query: str) -> str:
    results = query_similar(query_embedding.tolist(), n_results=5)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not docs:
        return "No results found."

    lines = [f"Top {len(docs)} results for: \"{query}\"\n"]
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
        similarity = 1 - dist
        snippet = doc[:200].replace("\n", " ")
        lines.append(
            f"{i}. [{meta.get('category', 'unknown')}] (similarity: {similarity:.3f})\n"
            f"   {snippet}..."
        )
    return "\n".join(lines)


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    query_embedding = embed_query(req.query)

    try:
        cluster_probs = get_query_cluster_probs(query_embedding)
        dominant_cluster = int(np.argmax(cluster_probs))
    except FileNotFoundError:
        cluster_probs = np.array([1.0])
        dominant_cluster = 0

    cache_result = _cache.lookup(req.query, query_embedding, cluster_probs)

    if cache_result is not None:
        entry, score = cache_result
        return QueryResponse(
            query=req.query,
            cache_hit=True,
            matched_query=entry.query,
            similarity_score=round(score, 4),
            result=entry.result,
            dominant_cluster=entry.dominant_cluster,
        )

    result = compute_result(query_embedding, req.query)
    _cache.store(req.query, result, query_embedding, cluster_probs)

    
    

    return QueryResponse(
        query=req.query,
        cache_hit=False,
        matched_query=None,
        similarity_score=None,
        result=result,
        dominant_cluster=dominant_cluster,
        
    )


@app.get("/cache/stats", response_model=CacheStatsResponse)
async def cache_stats():
    return CacheStatsResponse(**_cache.stats)


@app.delete("/cache")
async def flush_cache():
    _cache.flush()
    return {"message": "Cache flushed successfully"}


@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/cluster/stats")
async def cluster_stats():
    """
    Returns semantic cluster analysis:
    - How many docs per cluster
    - Top categories per cluster  
    - Average confidence
    - Boundary documents (most uncertain)
    """
    assignments_path = Path("clustering/doc_cluster_assignments.json")
    processed_path = Path("data/processed_corpus.json")

    if not assignments_path.exists():
        raise HTTPException(status_code=404, detail="Cluster assignments not found. Run pipeline.py first.")

    with open(assignments_path) as f:
        assignments = json.load(f)

    with open(processed_path) as f:
        docs = json.load(f)

    doc_map = {d["id"]: d for d in docs}
    k = len(next(iter(assignments.values())))

    cluster_summaries = []

    for cluster_id in range(k):
        # all docs where this is dominant cluster
        dominant_docs = [
            (doc_id, probs[cluster_id])
            for doc_id, probs in assignments.items()
            if np.argmax(probs) == cluster_id
        ]

        if not dominant_docs:
            continue

        # top categories
        from collections import Counter
        categories = Counter(
            doc_map[doc_id]["category"]
            for doc_id, _ in dominant_docs
            if doc_id in doc_map
        )
        top_categories = [cat for cat, _ in categories.most_common(3)]

        # average confidence
        avg_confidence = round(
            float(np.mean([p for _, p in dominant_docs])), 4
        )

        # most uncertain doc (highest entropy = boundary doc)
        entropies = []
        for doc_id, _ in dominant_docs:
            probs_array = np.array(assignments[doc_id])
            entropy = float(-np.sum(probs_array * np.log(probs_array + 1e-10)))
            entropies.append((doc_id, entropy))

        entropies.sort(key=lambda x: -x[1])
        boundary_doc_id = entropies[0][0] if entropies else None
        boundary_snippet = ""
        boundary_distribution = {}

        if boundary_doc_id and boundary_doc_id in doc_map:
            boundary_snippet = doc_map[boundary_doc_id]["text"][:150].replace("\n", " ")
            all_probs = assignments[boundary_doc_id]
            top5 = sorted(enumerate(all_probs), key=lambda x: -x[1])[:5]
            boundary_distribution = {
                f"cluster_{i}": round(p, 4) for i, p in top5
            }

        cluster_summaries.append({
            "cluster_id": cluster_id,
            "doc_count": len(dominant_docs),
            "top_categories": top_categories,
            "avg_confidence": avg_confidence,
            "boundary_doc": {
                "snippet": boundary_snippet,
                "soft_distribution": boundary_distribution
            }
        })

    # sort by doc count descending
    cluster_summaries.sort(key=lambda x: -x["doc_count"])

    return {
        "total_clusters": k,
        "total_docs": len(assignments),
        "clusters": cluster_summaries
    }
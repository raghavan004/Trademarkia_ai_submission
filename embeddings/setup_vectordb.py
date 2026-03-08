"""
Embeds the corpus and stores vectors in Qdrant (local, in-memory/persistent).

Switched from ChromaDB to Qdrant due to ChromaDB incompatibility with Python 3.14.
Qdrant supports Python 3.14 and offers the same cosine similarity search.
"""

import logging
from pathlib import Path
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

QDRANT_DIR = Path(__file__).parent.parent / "qdrant_db"
COLLECTION_NAME = "newsgroups"
EMBED_MODEL = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384
BATCH_SIZE = 64


def get_client():
    QDRANT_DIR.mkdir(exist_ok=True)
    return QdrantClient(path=str(QDRANT_DIR))


def embed_and_store(docs: list, force: bool = False) -> None:
    client = get_client()

    # Check if collection already exists
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing and not force:
        count = client.count(COLLECTION_NAME).count
        logger.info(f"Collection already has {count} docs. Skipping embedding.")
        return

    # Create collection
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    logger.info(f"Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    logger.info(f"Embedding {len(docs)} documents in batches of {BATCH_SIZE}...")
    for i in tqdm(range(0, len(docs), BATCH_SIZE), desc="Embedding"):
        batch = docs[i: i + BATCH_SIZE]
        texts = [d["text"][:2000] for d in batch]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        points = [
            PointStruct(
                id=int(d["id"]),
                vector=emb,
                payload={"category": d["category"], "label_id": d["label_id"], "text": text},
            )
            for d, emb, text in zip(batch, embeddings, texts)
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)

    logger.info(f"Stored {client.count(COLLECTION_NAME).count} documents in Qdrant")


def query_similar(query_embedding: list, n_results: int = 5) -> dict:
    client = get_client()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=n_results,
        with_payload=True,
    ).points

    docs, metas, distances = [], [], []
    for r in results:
        docs.append(r.payload.get("text", ""))
        metas.append({
            "category": r.payload.get("category", "unknown"),
            "label_id": r.payload.get("label_id", -1),
        })
        distances.append(1 - r.score)  # convert similarity to distance

    return {"documents": [docs], "metadatas": [metas], "distances": [distances]}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.preprocess import load_and_preprocess
    docs = load_and_preprocess()
    embed_and_store(docs)
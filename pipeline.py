"""
Run this ONCE before starting the API.
Downloads data → embeds → clusters → prints analysis.
"""

import sys
import logging
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data.preprocess import load_and_preprocess
from embeddings.setup_vectordb import embed_and_store, EMBED_MODEL
from clustering.fuzzy_cluster import fit_clusters, analyse_clusters
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("=== STEP 1: Preprocessing ===")
    docs = load_and_preprocess()
    logger.info(f"Corpus: {len(docs)} documents")

    logger.info("=== STEP 2: Embedding + Vector DB ===")
    embed_and_store(docs)

    logger.info("=== STEP 3: Fuzzy Clustering ===")
    model = SentenceTransformer(EMBED_MODEL)
    texts = [d["text"][:2000] for d in docs]
    logger.info("Encoding all documents for GMM clustering...")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)

    assignments = fit_clusters(np.array(embeddings), [d["id"] for d in docs])

    logger.info("=== STEP 4: Cluster Analysis ===")
    analyse_clusters(docs, assignments)

    logger.info("\n✅ Pipeline complete!")
    logger.info("Start the API with:")
    logger.info("   uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload")


if __name__ == "__main__":
    main()
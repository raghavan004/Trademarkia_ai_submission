import re
import json
import logging
from pathlib import Path
from sklearn.datasets import fetch_20newsgroups

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent
PROCESSED_PATH = DATA_DIR / "processed_corpus.json"


def clean_text(text: str) -> str:
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"^[\s\-_=><|*#~]{3,}$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_and_preprocess(min_length: int = 100) -> list:
    """
    Stripping headers/footers/quotes because keeping them made embeddings
    cluster on who posted rather than what they posted about. Tried both ways.

    Dropped docs under 100 chars — they're mostly one-liner replies with
    no real topic signal, just noise in the embedding space.

    Went with all-MiniLM-L6-v2 over mpnet — about 3x faster on CPU,
    quality difference on this kind of text is small. 18k docs on a
    laptop without GPU made that an easy call.

    Using Qdrant local instead of ChromaDB (Python 3.14 broke ChromaDB).
    Same persistent cosine search, no server process needed, HNSW index
    keeps lookup fast as collection grows.
    """
    if PROCESSED_PATH.exists():
        logger.info("Loading preprocessed corpus from cache...")
        with open(PROCESSED_PATH) as f:
            return json.load(f)

    logger.info("Fetching 20 Newsgroups dataset...")
    dataset = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=42,
    )

    docs = []
    skipped = 0
    for i, (text, label_id) in enumerate(zip(dataset.data, dataset.target)):
        cleaned = clean_text(text)
        if len(cleaned) < min_length:
            skipped += 1
            continue
        docs.append({
            "id": str(i),
            "text": cleaned,
            "category": dataset.target_names[label_id],
            "label_id": int(label_id),
        })

    logger.info(f"Kept {len(docs)} documents, skipped {skipped}")

    from collections import Counter
    for cat, count in sorted(Counter(d["category"] for d in docs).items(), key=lambda x: -x[1]):
        logger.info(f"  {cat}: {count}")

    with open(PROCESSED_PATH, "w") as f:
        json.dump(docs, f)

    return docs


if __name__ == "__main__":
    docs = load_and_preprocess()
    print(f"Total: {len(docs)}")
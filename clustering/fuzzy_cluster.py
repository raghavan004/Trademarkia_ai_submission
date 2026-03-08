import json
import logging
import pickle
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CLUSTER_DIR = Path(__file__).parent
MODEL_PATH = CLUSTER_DIR / "gmm_model.pkl"
PCA_PATH = CLUSTER_DIR / "pca_model.pkl"
ASSIGNMENTS_PATH = CLUSTER_DIR / "doc_cluster_assignments.json"

PCA_DIMS = 100
K_MIN, K_MAX = 5, 35


def find_optimal_k(embeddings_pca: np.ndarray) -> int:
    """
    Sweep k via BIC on a 3000-doc subsample — full sweep on 18k is too slow.
    BIC shape is stable enough at 3k to pick a good k.
    """
    logger.info(f"BIC sweep k={K_MIN}..{K_MAX}...")
    sample_size = min(3000, len(embeddings_pca))
    idx = np.random.default_rng(42).choice(len(embeddings_pca), sample_size, replace=False)
    sample = embeddings_pca[idx]

    bic_scores = {}
    for k in tqdm(range(K_MIN, K_MAX + 1, 2), desc="BIC sweep"):
        gmm = GaussianMixture(n_components=k, covariance_type="diag", random_state=42, max_iter=100, n_init=1)
        gmm.fit(sample)
        bic_scores[k] = gmm.bic(sample)

    optimal_k = min(bic_scores, key=bic_scores.get)
    for k, bic in sorted(bic_scores.items()):
        logger.info(f"  k={k:2d} BIC={bic:.1f}{' <-- best' if k == optimal_k else ''}")
    return optimal_k


def fit_clusters(embeddings: np.ndarray, doc_ids: list, force: bool = False) -> dict:
    """
    Using GMM over KMeans because we need soft assignments — a post about
    gun legislation genuinely belongs to both politics and firearms clusters.
    KMeans would force a single label, GMM gives a probability distribution.

    PCA to 100 dims before fitting — GMM covariance estimation gets unreliable
    in 384 dims with 18k points. 100 dims keeps ~85% variance and trains fast.

    k=20 felt like cheating since the labels are given. BIC sweep usually
    lands at 25-30 which makes sense — the 20 categories have real sub-splits.
    covariance_type=diag is more stable than full for this data size.
    """
    if ASSIGNMENTS_PATH.exists() and not force:
        logger.info("Loading existing cluster assignments...")
        with open(ASSIGNMENTS_PATH) as f:
            return json.load(f)

    embeddings_norm = normalize(embeddings, norm="l2")

    logger.info(f"PCA 384 -> {PCA_DIMS} dims...")
    pca = PCA(n_components=PCA_DIMS, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings_norm)
    logger.info(f"Variance retained: {pca.explained_variance_ratio_.sum():.1%}")

    with open(PCA_PATH, "wb") as f:
        pickle.dump(pca, f)

    optimal_k = find_optimal_k(embeddings_pca)

    logger.info(f"Fitting GMM k={optimal_k}...")
    gmm = GaussianMixture(
        n_components=optimal_k,
        covariance_type="diag",
        random_state=42,
        max_iter=200,
        n_init=3,
        verbose=1,
    )
    gmm.fit(embeddings_pca)
    logger.info(f"Converged: {gmm.converged_}, log-likelihood: {gmm.lower_bound_:.4f}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(gmm, f)

    probs = gmm.predict_proba(embeddings_pca)
    assignments = {doc_id: probs[i].tolist() for i, doc_id in enumerate(doc_ids)}

    with open(ASSIGNMENTS_PATH, "w") as f:
        json.dump(assignments, f)

    logger.info(f"Saved assignments for {len(assignments)} docs")
    return assignments


def get_query_cluster_probs(query_embedding: np.ndarray) -> np.ndarray:
    with open(PCA_PATH, "rb") as f:
        pca = pickle.load(f)
    with open(MODEL_PATH, "rb") as f:
        gmm = pickle.load(f)

    q_norm = normalize(query_embedding.reshape(1, -1), norm="l2")
    return gmm.predict_proba(pca.transform(q_norm))[0]


def analyse_clusters(docs: list, assignments: dict) -> None:
    """
    Print what lives in each cluster and where the model is uncertain.
    Boundary docs are the most interesting — high entropy means the GMM
    couldn't confidently assign them, which usually means genuinely cross-topic.
    """
    doc_map = {d["id"]: d for d in docs}
    k = len(next(iter(assignments.values())))

    print(f"\n{'='*60}")
    print(f"CLUSTER ANALYSIS ({k} clusters, {len(assignments)} docs)")
    print(f"{'='*60}\n")

    cluster_data = []
    for cluster_id in range(k):
        dominant = [(doc_id, probs[cluster_id]) for doc_id, probs in assignments.items() if np.argmax(probs) == cluster_id]
        if not dominant:
            continue

        categories = Counter(doc_map[d]["category"] for d, _ in dominant if d in doc_map)
        avg_conf = float(np.mean([p for _, p in dominant]))

        entropies = sorted(
            [(d, float(-np.sum(np.array(assignments[d]) * np.log(np.array(assignments[d]) + 1e-10)))) for d, _ in dominant],
            key=lambda x: -x[1]
        )
        boundary_id = entropies[0][0] if entropies else None
        boundary_doc = doc_map.get(boundary_id, {})
        boundary_dist = {}
        if boundary_id:
            top5 = sorted(enumerate(assignments[boundary_id]), key=lambda x: -x[1])[:5]
            boundary_dist = {f"C{i}": round(p, 3) for i, p in top5 if p > 0.05}

        cluster_data.append((cluster_id, dominant, categories, avg_conf, boundary_doc, boundary_dist))

    for cluster_id, dominant, categories, avg_conf, boundary_doc, boundary_dist in sorted(cluster_data, key=lambda x: -len(x[1])):
        print(f"Cluster {cluster_id:2d} | {len(dominant):4d} docs | confidence: {avg_conf:.3f}")
        print(f"  Categories : {', '.join(f'{c}({n})' for c, n in categories.most_common(3))}")
        if boundary_doc:
            print(f"  Boundary   : \"{boundary_doc.get('text','')[:120].replace(chr(10),' ')}...\"")
            print(f"  Soft dist  : {boundary_dist}")
        print()
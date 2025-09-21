import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_docs(jsonl_path: Path) -> List[str]:
    texts: List[str] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            title = obj.get("title") or ""
            text = obj.get("text") or ""
            texts.append((title + "\n" + text).strip())
    return texts


def load_bm25(index_dir: Path):
    with (index_dir / "bm25.pkl").open("rb") as f:
        bm25 = pickle.load(f)
    with (index_dir / "doc_ids.pkl").open("rb") as f:
        doc_ids = pickle.load(f)
    return bm25, doc_ids


def load_vector(index_dir: Path) -> Tuple[str, object, List[str]]:
    doc_ids = (index_dir / "doc_ids.txt").read_text(encoding="utf-8").splitlines()
    try:
        import faiss  # type: ignore
        idx_path = index_dir / "index.faiss"
        if idx_path.exists():
            index = faiss.read_index(str(idx_path))
            return "faiss", index, doc_ids
    except Exception:
        pass
    # sklearn fallback
    import pickle
    with (index_dir / "nn.pkl").open("rb") as f:
        nn = pickle.load(f)
    return "sklearn", nn, doc_ids


def tokenize(text: str) -> List[str]:
    return text.lower().split()


def rrf(rank_lists: List[List[Tuple[str, float]]], k: int, k_rrf: int = 60) -> List[Tuple[str, float]]:
    scores: Dict[str, float] = {}
    for ranked in rank_lists:
        for rank, (doc_id, _) in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k_rrf + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--docs",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "coral" / "docs.jsonl"),
    )
    parser.add_argument(
        "--bm25_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "index" / "coral_bm25"),
    )
    parser.add_argument(
        "--vec_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "index" / "coral_faiss"),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    args = parser.parse_args()

    docs = load_docs(Path(args.docs))
    # BM25
    bm25, bm25_ids = load_bm25(Path(args.bm25_dir))
    q_toks = tokenize(args.query)
    bm25_scores = bm25.get_scores(q_toks)
    bm25_ranked = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)[: args.k * 5]
    bm25_list: List[Tuple[str, float]] = [(bm25_ids[i], float(s)) for i, s in bm25_ranked]

    # Vector
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model)
    q = model.encode([args.query], convert_to_numpy=True, normalize_embeddings=True)[0]
    vec_type, vec_index, vec_ids = load_vector(Path(args.vec_dir))

    if vec_type == "faiss":
        scores, idx = vec_index.search(np.expand_dims(q, 0), args.k * 5)
        idx = idx[0].tolist()
        scores = scores[0].tolist()
    else:
        import pickle
        distances, indices = vec_index.kneighbors(np.expand_dims(q, 0), n_neighbors=args.k * 5)
        idx = indices[0].tolist()
        scores = (1 - distances[0]).tolist()
    vec_list: List[Tuple[str, float]] = [(vec_ids[i], float(s)) for i, s in zip(idx, scores)]

    # RRF fusion
    fused = rrf([bm25_list, vec_list], k=args.k)

    # Prepare output with snippets
    id_to_idx = {did: i for i, did in enumerate(bm25_ids)}
    results = []
    for did, s in fused:
        # prefer bm25 index mapping; if not found, fallback to vector's map
        idx = id_to_idx.get(did)
        if idx is None:
            try:
                idx = load_doc_ids(Path(args.vec_dir) / "doc_ids.txt").index(did)  # type: ignore
            except Exception:
                idx = 0
        results.append({"doc_id": did, "rrf": round(s, 6), "snippet": docs[idx][:300]})

    print(json.dumps({"ok": True, "query": args.query, "results": results}, ensure_ascii=False, indent=2))


def load_doc_ids(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


if __name__ == "__main__":
    main()



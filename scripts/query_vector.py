import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple


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


def load_doc_ids(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--index_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "index" / "coral_faiss"),
    )
    parser.add_argument(
        "--docs",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "coral" / "docs.jsonl"),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    args = parser.parse_args()

    docs = load_docs(Path(args.docs))
    doc_ids = load_doc_ids(Path(args.index_dir) / "doc_ids.txt")

    # Encode query
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)
    q = model.encode([args.query], convert_to_numpy=True, normalize_embeddings=True)[0]

    # Try FAISS else sklearn fallback
    try:
        import faiss  # type: ignore

        index = faiss.read_index(str(Path(args.index_dir) / "index.faiss"))
        scores, idx = index.search(np.expand_dims(q, 0), args.k)
        idx = idx[0].tolist()
        scores = scores[0].tolist()
    except Exception:
        # sklearn fallback
        import pickle
        from sklearn.metrics.pairwise import cosine_similarity

        with (Path(args.index_dir) / "nn.pkl").open("rb") as f:
            nn = pickle.load(f)
        # sklearn's NearestNeighbors doesn't expose similarity directly; use kneighbors to get indices
        distances, indices = nn.kneighbors(np.expand_dims(q, 0), n_neighbors=args.k)
        idx = indices[0].tolist()
        # approximate cosine similarity from distances
        scores = (1 - distances[0]).tolist()

    results = []
    for i, s in zip(idx, scores):
        results.append({"doc_id": doc_ids[i], "score": float(s), "snippet": docs[i][:300]})
    print(json.dumps({"ok": True, "query": args.query, "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()



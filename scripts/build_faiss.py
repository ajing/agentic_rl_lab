import argparse
import json
from pathlib import Path
from typing import List, Tuple


def load_docs(jsonl_path: Path) -> Tuple[List[str], List[str]]:
    doc_ids: List[str] = []
    texts: List[str] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            doc_ids.append(obj.get("id", str(len(doc_ids))))
            title = obj.get("title") or ""
            text = obj.get("text") or ""
            texts.append((title + "\n" + text).strip())
    return doc_ids, texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--docs",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "coral" / "docs.jsonl"),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformers model id",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "index" / "coral_faiss"),
    )
    args = parser.parse_args()

    docs_path = Path(args.docs)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    doc_ids, texts = load_docs(docs_path)

    # Encode with SentenceTransformers
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise SystemExit(
            "sentence-transformers not installed. Install via: pip install sentence-transformers"
        ) from e

    model = SentenceTransformer(args.model)
    embeddings = model.encode(texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)

    # Try FAISS first
    index_type = None
    try:
        import faiss  # type: ignore

        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)
        faiss.write_index(index, str(out_dir / "index.faiss"))
        with (out_dir / "doc_ids.txt").open("w", encoding="utf-8") as f:
            for did in doc_ids:
                f.write(did + "\n")
        index_type = "faiss"
    except Exception:
        # Fallback to sklearn NearestNeighbors
        try:
            import numpy as np
            from sklearn.neighbors import NearestNeighbors

            nn = NearestNeighbors(metric="cosine", algorithm="brute")
            nn.fit(embeddings)
            import pickle

            with (out_dir / "nn.pkl").open("wb") as f:
                pickle.dump(nn, f)
            with (out_dir / "doc_ids.txt").open("w", encoding="utf-8") as f:
                for did in doc_ids:
                    f.write(did + "\n")
            index_type = "sklearn"
        except Exception as e:
            raise SystemExit(f"Failed to build vector index: {e}")

    print(json.dumps({"ok": True, "index_type": index_type, "num_docs": len(doc_ids), "index_dir": str(out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()


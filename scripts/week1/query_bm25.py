import argparse
import json
import pickle
from pathlib import Path
from typing import List, Tuple


def load_index(index_dir: Path):
    with (index_dir / "bm25.pkl").open("rb") as f:
        bm25 = pickle.load(f)
    with (index_dir / "doc_ids.pkl").open("rb") as f:
        doc_ids = pickle.load(f)
    return bm25, doc_ids


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


def tokenize(text: str) -> List[str]:
    return text.lower().split()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="Query string")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--index_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "index" / "coral_bm25"),
    )
    parser.add_argument(
        "--docs",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "coral" / "docs.jsonl"),
    )
    args = parser.parse_args()

    index_dir = Path(args.index_dir)
    docs_path = Path(args.docs)
    bm25, doc_ids = load_index(index_dir)
    docs = load_docs(docs_path)

    tokenized_query = tokenize(args.query)
    scores = bm25.get_scores(tokenized_query)
    # Top-k
    topk = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[: args.k]
    results = []
    for idx, score in topk:
        results.append(
            {
                "doc_id": doc_ids[idx],
                "score": float(score),
                "snippet": docs[idx][:300],
            }
        )
    print(json.dumps({"ok": True, "query": args.query, "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


import argparse
import json
import pickle
from pathlib import Path
from typing import List, Tuple

try:
    from rank_bm25 import BM25Okapi
except ImportError as e:
    raise SystemExit(
        "rank_bm25 not installed. Install via: pip install rank-bm25"
    ) from e


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


def tokenize(text: str) -> List[str]:
    return text.lower().split()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--docs",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "coral" / "docs.jsonl"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "index" / "coral_bm25"),
    )
    args = parser.parse_args()

    docs_path = Path(args.docs)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    doc_ids, texts = load_docs(docs_path)
    tokenized_corpus = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)

    with (out_dir / "bm25.pkl").open("wb") as f:
        pickle.dump(bm25, f)
    with (out_dir / "doc_ids.pkl").open("wb") as f:
        pickle.dump(doc_ids, f)

    print(json.dumps({"ok": True, "num_docs": len(doc_ids), "index_dir": str(out_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()


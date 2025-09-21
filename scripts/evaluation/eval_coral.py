import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_coral_turns(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    turns: List[Dict] = []
    for conv in data:
        conv_id = conv.get("conv_id")
        for t in conv.get("turns", []):
            turns.append({
                "conv_id": conv_id,
                "turn_id": t.get("turn_id"),
                "question": t.get("question") or "",
                "gold": t.get("golden_docs_pids") or [],
                "response": t.get("response") or "",
            })
    return turns


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


def eval_retrieval(gold: List[int], retrieved: List[str], ks: List[int]) -> Dict[str, float]:
    gold_set = {str(g) for g in gold}
    results: Dict[str, float] = {}
    for k in ks:
        topk = retrieved[:k]
        hit = int(len(gold_set.intersection(topk)) > 0)
        recall = 0.0
        if gold_set:
            recall = len(gold_set.intersection(topk)) / len(gold_set)
        results[f"hit@{k}"] = float(hit)
        results[f"recall@{k}"] = float(recall)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_path", type=str, default=str(Path(__file__).resolve().parents[1] / "data" / "coral" / "raw" / "test" / "new_test_conversation.json"))
    parser.add_argument("--docs", type=str, default=str(Path(__file__).resolve().parents[1] / "data" / "coral" / "docs.jsonl"))
    parser.add_argument("--bm25_dir", type=str, default=str(Path(__file__).resolve().parents[1] / "index" / "coral_bm25"))
    parser.add_argument("--vec_dir", type=str, default=str(Path(__file__).resolve().parents[1] / "index" / "coral_faiss"))
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--ks", type=str, default="5,10,20")
    args = parser.parse_args()

    ks = [int(x) for x in args.ks.split(",")]

    # Load resources
    turns = load_coral_turns(Path(args.split_path))
    # filter to those with gold
    turns = [t for t in turns if t.get("gold")]  # non-empty gold list
    turns = turns[: args.limit]

    docs = load_docs(Path(args.docs))
    bm25, bm25_ids = load_bm25(Path(args.bm25_dir))
    vec_type, vec_index, vec_ids = load_vector(Path(args.vec_dir))

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model)

    summary = {f"hit@{k}": 0.0 for k in ks}
    summary.update({f"recall@{k}": 0.0 for k in ks})
    per_example: List[Dict] = []

    t_start = time.time()
    for t in turns:
        q = t["question"]

        # BM25
        bm25_scores = bm25.get_scores(tokenize(q))
        bm25_ranked = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)[: max(ks) * 5]
        bm25_list: List[Tuple[str, float]] = [(bm25_ids[i], float(s)) for i, s in bm25_ranked]

        # Vector
        q_emb = model.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0]
        if vec_type == "faiss":
            scores, idx = vec_index.search(np.expand_dims(q_emb, 0), max(ks) * 5)
            idx = idx[0].tolist()
            scores = scores[0].tolist()
        else:
            distances, indices = vec_index.kneighbors(np.expand_dims(q_emb, 0), n_neighbors=max(ks) * 5)
            idx = indices[0].tolist()
            scores = (1 - distances[0]).tolist()
        vec_list: List[Tuple[str, float]] = [(vec_ids[i], float(s)) for i, s in zip(idx, scores)]

        # RRF fuse to max K
        fused = rrf([bm25_list, vec_list], k=max(ks))
        retrieved_ids = [did for did, _ in fused]

        metrics = eval_retrieval(t["gold"], retrieved_ids, ks)
        for k in ks:
            summary[f"hit@{k}"] += metrics[f"hit@{k}"]
            summary[f"recall@{k}"] += metrics[f"recall@{k}"]

        per_example.append({
            "conv_id": t["conv_id"],
            "turn_id": t["turn_id"],
            "question": q,
            "gold": [str(g) for g in t["gold"]],
            "retrieved": retrieved_ids,
            "metrics": metrics,
        })

    n = len(turns)
    for k in ks:
        summary[f"hit@{k}"] = round(summary[f"hit@{k}"] / n, 4)
        summary[f"recall@{k}"] = round(summary[f"recall@{k}"] / n, 4)
    summary["num_eval"] = n
    summary["elapsed_sec"] = round(time.time() - t_start, 2)

    out_dir = Path(__file__).resolve().parents[1] / "outputs" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "coral_retrieval_100.jsonl").open("w", encoding="utf-8") as f:
        for row in per_example:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps({"ok": True, **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()



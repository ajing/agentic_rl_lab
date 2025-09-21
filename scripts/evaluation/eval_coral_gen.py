import argparse
import json
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import requests

try:
    from dotenv import load_dotenv  # type: ignore
    from pathlib import Path as _P
    _env_path = _P(__file__).resolve().parents[1] / ".env"
    if _env_path.exists():
        load_dotenv(dotenv_path=str(_env_path))
except Exception:
    pass


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
                "gold": t.get("response") or "",
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


def normalize(s: str) -> str:
    import re
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def em_f1(pred: str, gold: str) -> Tuple[float, float]:
    p = normalize(pred)
    g = normalize(gold)
    em = float(p == g)
    p_toks = p.split()
    g_toks = g.split()
    if not p_toks and not g_toks:
        return em, 1.0
    if not p_toks or not g_toks:
        return em, 0.0
    from collections import Counter
    inter = Counter(p_toks) & Counter(g_toks)
    num_same = sum(inter.values())
    if num_same == 0:
        return em, 0.0
    precision = num_same / len(p_toks)
    recall = num_same / len(g_toks)
    f1 = 2 * precision * recall / (precision + recall)
    return em, f1


def build_prompt(query: str, doc_ids: List[str], id_to_idx: Dict[str, int], docs: List[str], k: int) -> str:
    ctx = []
    for i, did in enumerate(doc_ids[:k], start=1):
        idx = id_to_idx.get(did)
        snippet = docs[idx][:800].replace("\n", " ") if idx is not None else ""
        ctx.append(f"[Doc {i}] id={did}\n{snippet}")
    context_block = "\n\n".join(ctx)
    instr = (
        "You are a helpful assistant. Answer the question using ONLY the provided documents. "
        "Cite doc ids inline like [Doc 1], [Doc 2]. If insufficient information, say you don't have enough context."
    )
    return f"{instr}\n\nContext:\n{context_block}\n\nQuestion: {query}\nAnswer:"


def call_openai(prompt: str, model: str, api_key: str, max_tokens: int = 256) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a concise, grounded assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_path", type=str, default=str(Path(__file__).resolve().parents[1] / "data" / "coral" / "raw" / "test" / "new_test_conversation.json"))
    parser.add_argument("--docs", type=str, default=str(Path(__file__).resolve().parents[1] / "data" / "coral" / "docs.jsonl"))
    parser.add_argument("--bm25_dir", type=str, default=str(Path(__file__).resolve().parents[1] / "index" / "coral_bm25"))
    parser.add_argument("--vec_dir", type=str, default=str(Path(__file__).resolve().parents[1] / "index" / "coral_faiss"))
    parser.add_argument("--retriever_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--gen_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max_tokens", type=int, default=256)
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set. Export it or add to .env")

    turns = load_coral_turns(Path(args.split_path))[: args.limit]
    docs = load_docs(Path(args.docs))
    bm25, bm25_ids = load_bm25(Path(args.bm25_dir))
    vec_type, vec_index, vec_ids = load_vector(Path(args.vec_dir))

    id_to_idx = {did: i for i, did in enumerate(bm25_ids)}

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(args.retriever_model)

    out_dir = Path(__file__).resolve().parents[1] / "outputs" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"coral_gen_{args.limit}.jsonl"

    total_em = 0.0
    total_f1 = 0.0

    t0 = time.time()
    with out_path.open("w", encoding="utf-8") as w:
        for t in turns:
            q = t["question"]
            # BM25
            bm25_scores = bm25.get_scores(tokenize(q))
            bm25_ranked = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)[: args.k * 5]
            bm25_list: List[Tuple[str, float]] = [(bm25_ids[i], float(s)) for i, s in bm25_ranked]
            # Vector
            q_emb = enc.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0]
            if vec_type == "faiss":
                scores, idx = vec_index.search(np.expand_dims(q_emb, 0), args.k * 5)
                idx = idx[0].tolist()
                scores = scores[0].tolist()
            else:
                distances, indices = vec_index.kneighbors(np.expand_dims(q_emb, 0), n_neighbors=args.k * 5)
                idx = indices[0].tolist()
                scores = (1 - distances[0]).tolist()
            vec_list: List[Tuple[str, float]] = [(vec_ids[i], float(s)) for i, s in zip(idx, scores)]

            fused = rrf([bm25_list, vec_list], k=args.k)
            retrieved_ids = [did for did, _ in fused]
            prompt = build_prompt(q, retrieved_ids, id_to_idx, docs, args.k)

            try:
                pred = call_openai(prompt, args.gen_model, api_key, max_tokens=args.max_tokens)
            except Exception as e:
                pred = f"<error: {e}>"

            em, f1 = em_f1(pred, t["gold"]) if not pred.startswith("<error:") else (0.0, 0.0)
            total_em += em
            total_f1 += f1

            rec = {
                "conv_id": t["conv_id"],
                "turn_id": t["turn_id"],
                "question": q,
                "gold": t["gold"],
                "pred": pred,
                "em": round(em, 4),
                "f1": round(f1, 4),
                "retrieved": retrieved_ids,
            }
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n = len(turns)
    summary = {
        "num_eval": n,
        "em": round(total_em / n, 4),
        "f1": round(total_f1 / n, 4),
        "elapsed_sec": round(time.time() - t0, 2),
        "k": args.k,
        "gen_model": args.gen_model,
    }
    with (out_dir / "summary_gen.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({"ok": True, **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()



import argparse
import json
import os
from pathlib import Path
from typing import List, Dict

import requests
try:
    from dotenv import load_dotenv  # type: ignore
    from pathlib import Path as _P
    _env_path = _P(__file__).resolve().parents[1] / ".env"
    if _env_path.exists():
        load_dotenv(dotenv_path=str(_env_path))
except Exception:
    pass


def read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_snippets(snippets_path: Path) -> List[Dict]:
    with snippets_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("results", [])


def build_prompt(query: str, snippets: List[Dict], k: int) -> str:
    ctx = []
    for i, r in enumerate(snippets[:k], start=1):
        ctx.append(f"[Doc {i}] id={r.get('doc_id')}\n{r.get('snippet','')}")
    context_block = "\n\n".join(ctx)
    instr = (
        "You are a helpful assistant. Answer the question using ONLY the provided documents. "
        "Cite doc ids inline like [Doc 1], [Doc 2]. If insufficient information, say you don't have enough context."
    )
    return f"{instr}\n\nContext:\n{context_block}\n\nQuestion: {query}\nAnswer:"


def call_openrouter(prompt: str, model: str, api_key: str, max_tokens: int = 256) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a concise, grounded assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--snippets",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "snippets.json"),
        help="Path to snippets JSON from rag_rrf (will generate if not found)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="x-ai/grok-3-mini-beta",
        help="OpenRouter model id",
    )
    parser.add_argument("--max_tokens", type=int, default=256)
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not set in env")

    snippets_path = Path(args.snippets)
    if not snippets_path.exists():
        # produce snippets by calling rag_rrf.py
        rag_script = Path(__file__).resolve().parents[0] / "rag_rrf.py"
        out_dir = snippets_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        import subprocess
        r = subprocess.run(
            [
                "python3",
                str(rag_script),
                args.query,
                "--k",
                str(args.k),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        with snippets_path.open("w", encoding="utf-8") as f:
            f.write(r.stdout)

    snippets = load_snippets(snippets_path)
    prompt = build_prompt(args.query, snippets, args.k)
    answer = call_openrouter(prompt, args.model, api_key, max_tokens=args.max_tokens)
    print(json.dumps({"ok": True, "query": args.query, "model": args.model, "answer": answer}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()



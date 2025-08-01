import argparse
import json
from pathlib import Path
from typing import Dict, Iterable


def iter_passages(passage_path: Path) -> Iterable[Dict]:
    """Iterate passages from a file that may be JSONL or a big JSON blob.

    CORAL's passage_corpus.json is JSONL: each line is a JSON object like
    {"ref_id": int, "ref_string": str}.
    """
    with passage_path.open("r", encoding="utf-8") as f:
        first_chunk = f.read(8192)
        f.seek(0)
        # Heuristic: JSONL if it starts with '{' on first non-space and has multiple lines
        is_jsonl = first_chunk.lstrip().startswith("{") and "\n{" in first_chunk
        if is_jsonl:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pid = (
                    obj.get("ref_id")
                    or obj.get("pid")
                    or obj.get("id")
                    or obj.get("docid")
                )
                text = (
                    obj.get("ref_string")
                    or obj.get("passage")
                    or obj.get("text")
                    or obj.get("content")
                    or ""
                )
                title = obj.get("title") or ""
                if pid is None or not text:
                    continue
                yield {"id": str(pid), "title": title, "text": text}
        else:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    pid = item.get("pid") or item.get("id") or item.get("docid")
                    title = item.get("title") or ""
                    text = item.get("passage") or item.get("text") or item.get("content") or ""
                    if pid is None:
                        continue
                    yield {"id": str(pid), "title": title, "text": text}
            elif isinstance(data, dict):
                for pid, val in data.items():
                    if isinstance(val, dict):
                        title = val.get("title") or ""
                        text = val.get("passage") or val.get("text") or val.get("content") or ""
                    else:
                        title = ""
                        text = str(val)
                    yield {"id": str(pid), "title": title, "text": text}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "coral" / "raw" / "passage_corpus.json"),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "coral" / "docs.jsonl"),
    )
    args = parser.parse_args()

    inp = Path(args.input)
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with outp.open("w", encoding="utf-8") as w:
        for ex in iter_passages(inp):
            if not ex.get("text"):
                continue
            w.write(json.dumps(ex, ensure_ascii=False) + "\n")
            count += 1

    print(json.dumps({"ok": True, "output": str(outp), "num_docs": count}, ensure_ascii=False))


if __name__ == "__main__":
    main()

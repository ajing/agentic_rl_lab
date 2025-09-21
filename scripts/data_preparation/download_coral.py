import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List

from huggingface_hub import snapshot_download


def find_files(root: Path, patterns: List[str]) -> List[Path]:
    files: List[Path] = []
    for pattern in patterns:
        files.extend(root.rglob(pattern))
    # De-duplicate while preserving order
    seen = set()
    unique: List[Path] = []
    for p in files:
        if p.resolve() not in seen:
            seen.add(p.resolve())
            unique.append(p)
    return unique


def count_json_items(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return len(data)
        if isinstance(data, dict):
            # common containers
            for key in ("data", "examples", "conversations"):
                if key in data and isinstance(data[key], list):
                    return len(data[key])
        return 0
    except Exception:
        return 0


def count_jsonl_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "coral"),
        help="Directory to write CORAL splits as JSONL",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "data" / "coral" / "hf_cache"),
        help="Hugging Face datasets cache dir",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Download repo snapshot (raw files)
    repo_path = Path(
        snapshot_download(
            repo_id="ariya2357/CORAL",
            repo_type="dataset",
            cache_dir=str(cache_dir),
        )
    )

    # Select likely files of interest (conversations + rewrites)
    candidates = find_files(
        repo_path,
        patterns=[
            "*conversation*.json",
            "*train*_conversation*.json",
            "*test*_conversation*.json",
            "*rewrite*.jsonl",
            "passage_corpus.json",
            "README.md",
        ],
    )

    copied: Dict[str, Dict] = {}
    total_bytes = 0
    for src in candidates:
        rel = src.relative_to(repo_path)
        dst = output_dir / "raw" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        size = dst.stat().st_size
        total_bytes += size
        # Count rows where feasible
        if dst.suffix == ".jsonl":
            num = count_jsonl_lines(dst)
        elif dst.suffix == ".json":
            num = count_json_items(dst)
        else:
            num = 0
        copied[str(rel)] = {"path": str(dst), "bytes": size, "approx_rows": num}

    manifest = {
        "dataset": "ariya2357/CORAL",
        "raw_files": copied,
        "approx_total_mb": round(total_bytes / (1024 * 1024), 2),
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(json.dumps({"ok": True, **manifest}, ensure_ascii=False))


if __name__ == "__main__":
    main()



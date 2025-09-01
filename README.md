# agentic_rl_lab

## Quickstart (Data → Index → Query)

1) Download CORAL raw files
```bash
python3 scripts/download_coral.py
```

2) Prepare corpus for indexing
```bash
python3 scripts/prepare_coral_corpus.py
```

3) Build indexes
```bash
# BM25
python3 scripts/build_bm25.py

# Vector (Sentence-Transformers); will fallback to sklearn if FAISS is unavailable
python3 scripts/build_faiss.py --model sentence-transformers/all-MiniLM-L6-v2
```

4) Run queries
```bash
# BM25
python3 scripts/query_bm25.py "who won the fa cup" --k 3

# Vector
python3 scripts/query_vector.py "who won the fa cup" --k 3
```

Data is stored under `data/coral/`, and indexes under `index/`.

## RAG Fusion CLI (BM25 + Vector via RRF)

Combine BM25 and vector search with Reciprocal Rank Fusion:
```bash
python3 scripts/rag_rrf.py "who won the fa cup" --k 5
```

## Generator via OpenRouter (Grok 3 Mini)

1) Create an OpenRouter API key and export it
```bash
export OPENROUTER_API_KEY=sk-or-...
```

2) Generate an answer using RRF retrieval + Grok 3 Mini
```bash
python3 scripts/generate_openrouter.py "who won the fa cup" --k 5 --max_tokens 256
```

## Generator via OpenAI (gpt-4o-mini)

1) Set your OpenAI key
```bash
export OPENAI_API_KEY=sk-...
```

2) Generate using gpt-4o-mini
```bash
python3 scripts/generate_openai.py "who won the fa cup" --k 5 --max_tokens 256
```

## Environment variables (.env)

Create a `.env` (not committed) at repo root:
```bash
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=sk-or-...
```
Scripts auto-load `.env` if `python-dotenv` is installed. Otherwise, export the variables in your shell.

## Learning & Documentation

### Weekly Learnings
- [`learning/week1.md`](learning/week1.md) - Week 1 insights, failure modes, and action items from baseline CORAL evaluation

### Project Documentation
- [`design/plan.md`](design/plan.md) - Overall project plan and timeline
- [`design/week2_plan.md`](design/week2_plan.md) - Week 2 RL environment implementation plan
- [`design/case_studies/`](design/case_studies/) - Detailed case studies and failure analysis

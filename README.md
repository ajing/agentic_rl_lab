# LEADR: Learning to Enhance Answer Diversity and Relevance

**LEADR** is a reinforcement learning framework for improving conversational RAG (Retrieval-Augmented Generation) systems. The project implements a "Retrieve-then-Rerank" pipeline with RL agents that learn to select diverse, relevant documents to improve answer quality, faithfulness, and novelty.

## 🎯 Project Overview

LEADR addresses key challenges in conversational RAG:
- **Coreference resolution** in multi-turn conversations
- **Document diversity** to avoid redundant information
- **Precision at top-K** through intelligent reranking
- **Reward-driven learning** for optimal document selection

### Key Components

- **Query Rewriter**: Resolves pronouns and references using conversation history
- **RRF Generator**: Combines BM25 and vector search via Reciprocal Rank Fusion
- **Cross-Encoder Reranker**: Precision-focused reranking with pre-trained models
- **MMR Deduplicator**: Maximal Marginal Relevance for diverse document selection
- **RL Environment**: State/action/reward framework for policy learning
- **Episode Runner**: Random, greedy, and epsilon-greedy policies for training

## 🚀 Quickstart (Data → Index → Query)

1) Download CORAL raw files
```bash
python3 scripts/data_preparation/download_coral.py
```

2) Prepare corpus for indexing
```bash
python3 scripts/data_preparation/prepare_coral_corpus.py
```

3) Build indexes
```bash
# BM25
python3 scripts/data_preparation/build_bm25.py

# Vector (Sentence-Transformers); will fallback to sklearn if FAISS is unavailable
python3 scripts/data_preparation/build_faiss.py --model sentence-transformers/all-MiniLM-L6-v2
```

4) Run queries
```bash
# BM25
python3 scripts/week1/query_bm25.py "who won the fa cup" --k 3

# Vector
python3 scripts/week1/query_vector.py "who won the fa cup" --k 3
```

Data is stored under `data/coral/`, and indexes under `index/`.

## 📁 Scripts Organization

The `scripts/` directory is organized by functionality and development phase:

- **`data_preparation/`** - Download, process, and index CORAL dataset
- **`week1/`** - Basic RAG pipeline and retrieval methods  
- **`week2/`** - RL Environment and advanced reranking
- **`week3_4/`** - Reward modeling and training components
- **`week5_6/`** - BC and RAFT training with real data
- **`training/`** - General training and generation scripts
- **`evaluation/`** - Evaluation and benchmarking scripts
- **`testing/`** - Testing and validation scripts

See `scripts/README.md` for detailed documentation of all scripts.

## RAG Fusion CLI (BM25 + Vector via RRF)

Combine BM25 and vector search with Reciprocal Rank Fusion:
```bash
python3 scripts/week1/rag_rrf.py "who won the fa cup" --k 5
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

## 🧠 Week 2: RL Environment & Advanced Reranking

### Cross-Encoder Reranker

The cross-encoder reranker improves precision by jointly encoding query-document pairs:

```python
from src.reranker.cross_encoder import CrossEncoderReranker

# Initialize with pre-trained model
reranker = CrossEncoderReranker(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",  # Default
    alpha=0.7  # Weight for combining scores
)

# Rerank candidates
candidates = [
    {"doc_id": "doc1", "content": "Arsenal won the FA Cup...", "rrf_score": 0.8},
    {"doc_id": "doc2", "content": "The FA Cup is an annual...", "rrf_score": 0.6}
]

reranked = reranker.rerank_candidates("Who won the FA Cup in 2020?", candidates)
```

### Available Pre-trained Models

```python
from src.reranker.pretrained_models import get_recommended_models, list_all_models

# Get recommended models for different use cases
general_models = get_recommended_models("general")  # MS MARCO models
qa_models = get_recommended_models("qa")           # Natural Questions models
multilingual_models = get_recommended_models("multilingual")  # XLM-R models

# List all available models
list_all_models()
```

**Recommended Models:**
- `cross-encoder/ms-marco-MiniLM-L-6-v2` - **Default**: Good speed/quality balance
- `cross-encoder/ms-marco-MiniLM-L-12-v2` - **Better quality**: Larger model
- `cross-encoder/nq-distilbert-base-v1` - **Q&A focused**: Natural Questions trained

### MMR Deduplication

Maximal Marginal Relevance selects diverse documents:

```python
from src.reranker.mmr import MMRDeduplicator

mmr = MMRDeduplicator()
selected = mmr.select_diverse_documents(
    query="Football players and achievements",
    candidates=candidates,
    lambda_mmr=0.5,  # Balance relevance vs diversity
    top_k=5
)
```

### RL Environment

Run episodes with different policies:

```python
from src.env.rag_environment import RAGEnvironment
from src.policy.episode_runner import EpisodeRunner, PolicyConfig

# Initialize environment
env = RAGEnvironment(
    max_steps=5,
    k_candidates=100,
    use_cross_encoder=True,
    use_mmr=True
)

# Run episode with random policy
runner = EpisodeRunner(env)
policy_config = PolicyConfig(policy_type="random", selection_strategy="random")
episode = runner.run_episode("Who won the FA Cup in 2020?", policy_config)
```

### Testing Components

```bash
# Test individual components
python scripts/test_week2_simple.py

# Test with verbose output
python scripts/test_week2_simple.py --verbose
```

## 📊 Performance & Results

### Week 1 Baseline (CORAL)
- **Retrieval**: hit@5=0.29, hit@10=0.32, recall@10≈0.289
- **Generation**: EM=0.0, F1≈0.155 (gpt-4o-mini)
- **Issues**: Topic drift, entity mismatch, low precision

### Week 2 Improvements
- **Cross-encoder**: Significant score improvements (7.390 vs 0.8)
- **MMR**: Effective diversity selection with configurable λ
- **RL Environment**: Complete state/action/reward framework
- **Apple Silicon**: Efficient MPS acceleration

## 🛠 Installation & Setup

### Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Key packages:
# - openai>=1.0.0 (query rewriting)
# - sentence-transformers>=2.2.0 (cross-encoder, MMR)
# - torch>=1.9.0 (model inference)
# - faiss-cpu>=1.7.0 (vector search)
```

### Environment Variables

Create `.env` file:
```bash
OPENAI_API_KEY=sk-...          # For query rewriting
OPENROUTER_API_KEY=sk-or-...   # Alternative LLM provider
```

## 📁 Project Structure

```
agentic_rl_lab/
├── src/                       # Core implementation
│   ├── retriever/            # Query rewriting, RRF generation
│   ├── reranker/             # Cross-encoder, MMR deduplication
│   ├── env/                  # RL environment framework
│   ├── policy/               # Episode runners, policies
│   └── eval/                 # Evaluation metrics
├── scripts/                  # Utility scripts
├── data/coral/               # CORAL dataset
├── index/                    # BM25 and vector indexes
├── outputs/                  # Evaluation results
├── learning/                 # Weekly learnings and insights
└── design/                   # Project documentation
```

## 📚 Learning & Documentation

### Weekly Learnings
- [`learning/week1.md`](learning/week1.md) - Week 1 insights, failure modes, and action items
- [`learning/week2.md`](learning/week2.md) - Week 2 implementation results and technical insights

### Project Documentation
- [`design/plan.md`](design/plan.md) - Overall project plan and timeline
- [`design/week2_plan.md`](design/week2_plan.md) - Week 2 RL environment implementation plan
- [`design/case_studies/`](design/case_studies/) - Detailed case studies and failure analysis

## 🎯 Roadmap

### Week 3-4: Reward Modeling & Training
- [ ] LLM-as-a-Judge preference data collection
- [ ] Lightweight reward model distillation
- [ ] Behavioral Cloning (BC) pretraining
- [ ] RAFT offline reinforcement learning

### Week 5-6: Evaluation & Optimization
- [ ] End-to-end evaluation on CORAL
- [ ] Ablation studies and hyperparameter tuning
- [ ] Performance analysis and case studies

### Future Extensions
- [ ] PPO online fine-tuning
- [ ] Adaptive retrieval granularity
- [ ] Long-term memory integration
- [ ] Multi-hop reasoning support

## 🤝 Contributing

This project implements the LEADR framework for conversational RAG improvement. Key areas for contribution:
- Reward model design and training
- Policy network architectures
- Evaluation metrics and benchmarks
- Integration with other RAG frameworks

## 📄 License

MIT License - see LICENSE file for details.

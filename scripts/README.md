# Scripts Directory Organization

This directory contains all the scripts for the LEADR project, organized by functionality and development phase.

## Directory Structure

### 📁 `data_preparation/`
Scripts for downloading, processing, and indexing the CORAL dataset.

- `download_coral.py` - Download CORAL dataset from Hugging Face
- `prepare_coral_corpus.py` - Process CORAL data into our format
- `build_bm25.py` - Build BM25 sparse retrieval index
- `build_faiss.py` - Build FAISS dense vector index

### 📁 `week1/`
Week 1: Basic RAG pipeline and retrieval methods.

- `query_bm25.py` - Query BM25 index for sparse retrieval
- `query_vector.py` - Query FAISS index for dense retrieval  
- `rag_rrf.py` - Reciprocal Rank Fusion (RRF) for combining retrievers

### 📁 `week2/`
Week 2: RL Environment and advanced reranking.

- `test_week2.py` - Comprehensive tests for Week 2 components
- `test_week2_simple.py` - Simple smoke tests for Week 2

### 📁 `week3_4/`
Week 3-4: Reward modeling and training components.

- `test_week3_core.py` - Core component tests for Week 3-4
- `test_week3_simple.py` - Simple smoke tests for Week 3-4
- `test_week3_integration.py` - Integration tests for Week 3-4
- `eval_week3_components.py` - Comprehensive evaluation of Week 3-4 components
- `test_realistic_scenarios.py` - Realistic conversational scenario tests

### 📁 `week5_6/`
Week 5-6: BC and RAFT training with real data.

- `generate_expert_trajectories.py` - Generate expert trajectories for BC training
- `build_preference_dataset.py` - Build preference datasets with LLM-as-a-Judge
- `train_with_real_data.py` - Main training pipeline orchestrator

### 📁 `training/`
General training and generation scripts.

- `generate_openai.py` - Generate responses using OpenAI API
- `generate_openrouter.py` - Generate responses using OpenRouter API

### 📁 `evaluation/`
Evaluation and benchmarking scripts.

- `eval_coral.py` - Evaluate retrieval performance on CORAL
- `eval_coral_gen.py` - Evaluate generation performance on CORAL

### 📁 `testing/`
Testing and validation scripts.

- `test_real_data_pipeline.py` - Test the real data training pipeline
- `test_real_data_no_api.py` - Test pipeline without API dependencies

## Usage Examples

### Data Preparation
```bash
# Download and prepare CORAL dataset
python data_preparation/download_coral.py
python data_preparation/prepare_coral_corpus.py

# Build retrieval indices
python data_preparation/build_bm25.py
python data_preparation/build_faiss.py
```

### Week 1: Basic RAG
```bash
# Test individual retrievers
python week1/query_bm25.py
python week1/query_vector.py

# Test RRF combination
python week1/rag_rrf.py
```

### Week 2: RL Environment
```bash
# Run Week 2 tests
python week2/test_week2_simple.py
python week2/test_week2.py
```

### Week 3-4: Reward Modeling
```bash
# Test core components
python week3_4/test_week3_core.py

# Run comprehensive evaluation
python week3_4/eval_week3_components.py

# Test realistic scenarios
python week3_4/test_realistic_scenarios.py
```

### Week 5-6: Real Data Training
```bash
# Generate expert trajectories
python week5_6/generate_expert_trajectories.py

# Build preference datasets (requires API key)
python week5_6/build_preference_dataset.py

# Run full training pipeline
python week5_6/train_with_real_data.py
```

### Evaluation
```bash
# Evaluate retrieval performance
python evaluation/eval_coral.py

# Evaluate generation performance
python evaluation/eval_coral_gen.py
```

### Testing
```bash
# Test real data pipeline
python testing/test_real_data_pipeline.py

# Test without API dependencies
python testing/test_real_data_no_api.py
```

## Notes

- **API Dependencies**: Some scripts require OpenAI API keys. Check individual scripts for requirements.
- **Data Dependencies**: Most scripts expect CORAL data to be prepared first using `data_preparation/` scripts.
- **Index Dependencies**: Retrieval scripts expect BM25 and FAISS indices to be built first.
- **Environment**: All scripts should be run from the project root directory.

## Development Workflow

1. **Setup**: Run `data_preparation/` scripts first
2. **Week 1**: Test basic retrieval with `week1/` scripts
3. **Week 2**: Test RL environment with `week2/` scripts  
4. **Week 3-4**: Test reward modeling with `week3_4/` scripts
5. **Week 5-6**: Run training with `week5_6/` scripts
6. **Evaluation**: Use `evaluation/` scripts to measure performance
7. **Testing**: Use `testing/` scripts to validate functionality

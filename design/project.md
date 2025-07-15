### **Project: LEADR (Learning-Enhanced Adaptive Document Retriever)**

#### **Mission**

Build a dynamic, adaptive document retriever. Instead of a one-shot similarity lookup, LEADR is an agent that learns from interaction and feedback to curate an optimal context set for a downstream LLM, maximizing answer quality and reliability.

#### **Architecture**

The system has four core modules:

1.  **Environment**
    * **Knowledge base**: A fixed collection (e.g., a Wikipedia subset, domain corpora, medical KB).
    * **User query**: Complex information-seeking questions.
    * **Retriever & candidate set**: Use BM25/vector search to produce a compact candidate pool (e.g., top-100 chunks) as the agent’s action space.

2.  **The LEADR Agent**
    * **Core**: An RL agent that incrementally builds the context over multiple steps from the candidate pool.
    * **State**: The query, history of already selected documents, and features/embeddings for remaining candidates. This history enables multi-step reasoning.
    * **Action**: Select or re-rank the next chunk from the candidate set to add to the context.
    * **Policy network**: A neural network that outputs a distribution over the candidate pool. This is what we train.

3.  **LLM Generator**
    * A fixed, non-finetuned open model (e.g., Llama 3).
    * Consumes the curated context from the agent and generates the final answer.

4.  **Reward Module**
    * **Role**: Evaluate the selected context set and the generated answer, returning a reward for training the agent.
    * **LLM-as-a-Judge**: Use a stronger model to perform pairwise comparisons and produce preference signals; optionally distill a lightweight reward model.
    * **Reward design**:
        * **Factuality/Attribution**: Strict grounding in provided context with correct citations.
        * **Comprehensiveness**: Synthesizes all relevant context.
        * **Novelty (MMR)**: Use MMR `λ * relevance - (1 - λ) * max_similarity_to_selected` to encourage non-redundant, information-rich selections.
        * **Bias mitigation**: Swap answer order, evaluate multiple times, and use Chain-of-Thought to reduce position/verbosity bias.
    * **Reward shaping**: Besides final reward, give small step-wise rewards when the agent selects highly relevant and novel documents.

#### **Step-by-Step Plan**

1.  **Phase 1: Baseline**
    * Implement Retrieve-then-Rerank RAG: BM25/vector top-K, then a simple reranker (e.g., cross-encoder or cosine) to assemble context.
    * Prepare and index the target dataset (prioritize conversational, knowledge-intensive data like CORAL). Record baseline performance.

2.  **Phase 2: RL Environment & Reward**
    * Wrap the above RAG into an RL environment (state = query + selected docs + candidate pool; action = select/rerank; termination = step limit or confidence).
    * Implement reward: start with answer-similarity/rules; move to LLM-as-a-Judge/reward model; incorporate MMR novelty and reward shaping.

3.  **Phase 3: Train the LEADR Agent**
    * **Behavioral Cloning (BC)**: Supervised pretrain on expert trajectories produced by a strong baseline to solve cold start.
    * **RAFT (offline RL)**: Sample candidate sets and answers, rank with the reward model, fine-tune the policy only on the top items (stable, simple).
    * (Optional) **PPO online finetuning** to further improve robustness if resources allow.

4.  **Phase 4: Evaluation & Analysis**
    * Evaluate on the same dataset; emphasize **CORAL** for multi-turn/topic-shift evaluation.
    * **Metrics**: EM/F1, faithfulness (context-grounded), attribution, novelty/diversity, latency and retrieval cost.
    * **Case studies**: Contrast baseline failures with LEADR successes; analyze document selection and multi-step strategy.

#### **Benchmarks**

* **Primary (Conversational RAG)**: **CORAL** — multi-turn, knowledge-intensive with topic shifts; ideal for testing adaptive retrieval and long-horizon planning.
* **Conversational alternatives**: **QReCC** (query rewriting/history use), **TopiOCQA** (topic shifts), **OR-QuAC** (explicit retrieval setup).
* **Single-turn multi-hop QA**: **HotpotQA**, **MuSiQue**, **2WikiMultihopQA**.
* **Open-domain single-hop QA**: **Natural Questions (NQ)**, **TriviaQA**.
* **Retriever/reranker pretraining & eval**: **MS MARCO Passage**, **BEIR**.

**Suggested metrics**:
* Final answer: EM / F1
* Retrieval quality: Recall@k / nDCG@k
* Faithfulness & attribution: context-only grounding and source citation accuracy
* Novelty/diversity: MMR-based de-duplication and context repetition rate
* System efficiency: end-to-end latency and retrieval cost

#### **Why this redesign is stronger**

1.  **Depth and novelty**: Goes beyond tool wiring; you design an RL training loop to improve RAG—a timely, research-grade direction.
2.  **Safety and reliability**: Tackles “garbage in, garbage out” by learning to curate trustworthy inputs, directly improving reliability.
3.  **Well-rounded showcase**:
    * **Engineering**: Multi-module system integration
    * **Algorithms**: RL design and implementation
    * **Research**: Experiment design, analysis, and insights
4.  **Feasible**: Focus on a smaller policy network with **BC + RAFT**; stable, cost-efficient, and practical on a single GPU.

---

#### **Strategic Directions**

1.  **Hybrid control**: Pretrain reflection/retrieval control via supervised BC; finetune with RAFT/PPO for the final objective.
2.  **Reward-model distillation**: Use a strong judge (e.g., GPT-4o) to create preferences, then distill into a lightweight open model (e.g., Llama 3 8B) to cut cost/latency.
3.  **Adaptive retrieval granularity**: Learn not only what to retrieve, but at what granularity (paragraph/sentence/dynamic summary) under context budgets.
4.  **Long-term memory**: Integrate external memory (vector store) into state to support multi-turn reasoning and long-horizon planning.
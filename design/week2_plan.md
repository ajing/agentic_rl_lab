### Week 2 Plan: RL Environment & Candidate Actions

#### Objectives
- Implement a minimal RL environment around the RAG pipeline (state/action/termination).
- Use BM25 + vector retrieval (RRF) to form a top-K candidate set per query.
- Add reward shaping with novelty (MMR-style) + relevance signals; keep a final reward hook.
- Provide a script to run one episode with random/greedy policy for smoke testing on CORAL.

#### Scope
- State: (query text, selected doc IDs, remaining candidate IDs/features, step index)
- Action: select next doc ID from remaining candidates (or terminate if step limit reached)
- Termination: step limit (e.g., 3–5) or no more candidates
- Reward: step-wise novelty (MMR), step-wise relevance (cosine/sim), optional final reward via generator

#### Data/Inputs
- Indexes: `index/coral_bm25`, `index/coral_faiss` (built in Week 1)
- Corpus: `data/coral/docs.jsonl`
- Test turns: `data/coral/raw/test/new_test_conversation.json`

#### Deliverables (this week)
1) Candidate generator (RRF over BM25 + vector) returning top-K doc IDs with features
2) RL env class (Python) with step/reset API and reward shaping
3) One-episode runner (random/greedy) + logging of selections and rewards
4) Smoke test results on 1–3 CORAL turns; brief notes on behavior

#### Metrics
- Step rewards (avg/std), selection overlap with gold (Recall@k), episode length
- Log trajectories for later BC pretraining

#### Risks/Mitigations
- Action space too large → restrict to top-K (≤100) from candidate generator
- Weak shaping signals → tune λ for MMR, rescale relevance, add small penalties for redundancy
- Runtime cost → small K, short horizon; no generator calls yet (final reward deferred)

#### Next (Week 3–4 preview)
- LLM-as-a-Judge preference data collection and reward-model distillation
- Behavioral Cloning pretraining with expert trajectories
- REQUIRED: generator alignment with TRL (DPO/IPO/ORPO) on a 7B–8B model via QLoRA
- REQUIRED (if doing online RL): PPO on the generator with OpenRLHF (2–4 GPUs)


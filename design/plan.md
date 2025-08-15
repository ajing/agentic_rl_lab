### **LEADR Detailed Implementation Plan**

#### **1. Goals & Scope**

* **Goal**: Build a Retrieve-then-Rerank conversational RAG baseline, then implement the LEADR RL agent (BC + RAFT primary, optional PPO online) to improve answer quality, faithfulness, and novelty on **CORAL**.
* **Scope**:
  * Data & indexing: vector/BM25 indexes, top-K candidate sets.
  * RL environment: standardized state/action/termination; candidate-set action space and multi-step selection.
  * Reward: LLM-as-a-Judge preferences → lightweight reward model; MMR novelty and reward shaping.
  * Training: BC pretraining → RAFT offline RL → (optional) PPO online.
  * Evaluation: CORAL primary, plus HotpotQA/MuSiQue; multi-metric and case studies.
  * Reproducibility: Poetry-locked deps, Ray job runbook.

#### **2. Deliverables**

* Repo with baseline RAG, RL env, reward pipeline, training and eval scripts.
* Repro scripts: Ray job entrypoints, configs, seeds, logs.
* Models/checkpoints: policy network, optional reward model.
* Metrics/report: EM/F1, Recall@k/nDCG, faithfulness/attribution, novelty, latency/cost, with case studies.

#### **3. Milestones & Timeline (6–8 weeks)**

1) Week 1: Setup + baseline retrieval
* Data prep and indexing (BM25 + vector, top-K candidates).
* Conversational RAG baseline with simple rerank (cosine or cross-encoder).
* End-to-end inference + basic metrics on CORAL.
* Note: Use hosted LLM APIs first; local serving optional for later.

2) Week 2: RL environment & candidate actions
* Define state (query, selected history, candidate features) and action (select/rerank from candidates).
* Support multi-step context building, step limits, confidence-based termination.
* Integrate MMR for novelty (online features + offline stats).

3) Weeks 3–4: Reward & data pipeline
* LLM-as-a-Judge prompts; pairwise comparisons; bias mitigation (swap order, CoT rationale).
* Train a lightweight reward model (distilled from judge preferences, offline scoring).
* Reward shaping design: final answer reward + step-wise novelty/relevance rewards.
* REQUIRED generator alignment (TRL): build preference datasets and run DPO/IPO/ORPO on a 7B–8B open model with QLoRA; evaluate on CORAL.

4) Weeks 4–5: BC and RAFT
* Generate expert trajectories from a strong baseline; BC pretraining to fix cold start.
* RAFT data: sample candidate sets + answers, rank via reward model, fine-tune on best.
* Stabilize with early stopping, LR, batch size, gradient clipping.

5) Week 6: Evaluation, ablations, report
* Evaluate on CORAL; add HotpotQA/MuSiQue, NQ/TriviaQA.
* Ablations: baseline vs +BC vs +BC+RAFT; vary K, MMR λ, reranker, shaping weights.
* Case studies and visualizations; technical report and README.

(Optional) Weeks 7–8: PPO & extensions
* PPO policy-value training with clipped updates for stability.
* Explore adaptive granularity, memory modules, reward-model distillation.
* REQUIRED (if choosing online generator RL): PPO for the generator with OpenRLHF (2–4 GPUs).

#### **4. System Blueprint**

* Candidates: BM25/vector (e.g., bge/contriever), top-K (default K=100).
* Reranking: cosine/cross-encoder (e.g., co-condenser/ms-marco finetunes).
* State: query embedding, selected doc sequence, candidate features (similarity, position, length, topic).
* Action: pick next chunk from candidates (or a distribution over remaining items).
* Termination: step limit (e.g., 3–5) or generator confidence threshold.
* Reward:
  * Final: judge/reward-model score combining faithfulness/attribution/comprehensiveness.
  * Step-wise: MMR novelty and relevance (reduce redundancy, encourage information gain).

#### **5. Data & Benchmarks**

* Primary: **CORAL** (multi-turn, topic shifts).
* Also: **QReCC**, **TopiOCQA**, **OR-QuAC**.
* Multi-hop: **HotpotQA**, **MuSiQue**, **2WikiMultihopQA**.
* Retrieval pretraining/eval: **MS MARCO Passage**, **BEIR**.

#### **6. Training Setup (defaults)**

* BC:
  * batch 64, lr 3e-5, max steps 20k, 10% linear warmup.
* RAFT:
  * per-batch sample N candidate+answer sets, rank by reward model, keep top-1 for supervised update;
  * batch 32, lr 2e-5, grad accumulation, mixed-precision.
* (Optional) PPO:
  * clip 0.2, GAE λ=0.95, γ=0.99, KL limit/penalty; horizon by dialogue length.

#### **Hardware & Resource Requirements**

* **Do you need a GPU?**
  * Small baselines (indexing/offline eval) run on CPU, but **BC/RAFT training and reranker inference strongly benefit from a GPU**.

* **Single-GPU (recommended)**
  * NVIDIA 24–40GB VRAM (RTX 4090 24GB, L4 24GB, A5000 24GB, A6000 48GB, A100 40GB).
  * Uses: BC/RAFT finetuning, cross-encoder rerank inference, vector retrieval acceleration.
  * Techniques: FP16/BF16, LoRA/QLoRA, grad checkpointing/accumulation, small batch size; optional 8/4-bit quantization.

* **Low VRAM (12–16GB)**
  * Feasible with trade-offs: smaller policy (≤1.3B), smaller batch, quantization, gradient accumulation.

* **Multi-GPU/cluster (optional)**
  * For PPO online, large rerankers, parallel eval. 2–4 GPUs are sufficient for most extensions; use Ray for distributed runs.

* **Judge/Reward model**
  * Prefer API-based strong judge for preference data; distill to a light reward model offline.
  * Local alternative: 8–13B open models (4/8-bit) on ≥24GB VRAM; 70B+ typically needs multi-GPU or ≥80GB VRAM.

* **CPU / RAM / Disk**
  * CPU: 8–16 cores.
  * RAM: ≥32GB (≥64GB ideal for full CORAL processing).
  * Disk: ≥100GB (data/index/checkpoints/logs); cloud nodes: ≥200GB temp.

* **Apple Silicon**
  * M1/M2/M3 work for small retrieval/eval and light BC; use NVIDIA GPU cloud for large RAFT/PPO.

* **Ray cluster**
  * Head: CPU-only.
  * Workers: 1×24–40GB NVIDIA GPU per node; ample local disk for indexes/cache.

* Local LLM (optional for Week 1):
  * macOS quick path (Ollama):
```bash
brew install ollama
ollama serve
ollama pull mistral:7b-instruct
```
  * OpenAI-compatible call:
```bash
export OPENAI_API_BASE=http://localhost:11434/v1
export OPENAI_API_KEY=ollama
```
  * Note: Consider Ray Serve or LM Studio later; hosted APIs recommended in Week 1.

#### **7. Evaluation & Metrics**

* Final: EM / F1
* Retrieval: Recall@k / nDCG@k
* Faithfulness/Attribution: context-grounded judgments and citation accuracy
* Novelty/Diversity: MMR de-duplication and context repetition rate
* Efficiency: end-to-end latency, retrieval cost (calls × context length)

#### **8. Ablations**

* K ∈ {20, 50, 100}; MMR λ ∈ {0.2, 0.4, 0.6, 0.8}.
* Reranker: none / cosine / cross-encoder.
* Reward shaping: final-only vs +step-wise; weight sweeps.
* Training combos: baseline vs +BC vs +BC+RAFT vs +BC+RAFT+PPO.

#### **9. Reproducibility & Ops**

* Suggested layout:
```
agentic_rl_lab/
  design/
  data/              # raw and processed data
  index/             # vector/inverted indexes
  src/
    retriever/
    reranker/
    env/             # RL env and state/action definitions
    reward/          # judge prompts, preference data, reward model
    policy/          # policy net, BC/RAFT/PPO training
    eval/            # metrics and analysis
  configs/
  train/             # Ray jobs
  scripts/
  outputs/           # logs, checkpoints, reports
```

* Dependency lock:
```bash
gazette ray poetry lock
```

* Ray runbook:
  * workspace submit:
```bash
gazette ray job submit --path train
```
  * workspace describe:
```bash
gazette ray workspace describe
```
  * list jobs on cluster:
```bash
kubectl exec -it cluster-jing-lu-head-2sdzw -n ray-ml-nonprod -- ray job list
```
  * standalone run:
```bash
gazette ray job run --path train
```
  * list current jobs:
```bash
gazette ray job list
```
  * job details:
```bash
gazette ray job describe {job_id}
```

* Logging & tracking:
  * Unified JSONL logs for config, seed, metrics, retrieval traces, doc IDs.
  * Optional MLflow/W&B; otherwise local CSV + artifacts.

#### **10. Risks & Mitigations**

* Judge cost → small preference set first, distill to light reward model; reuse data.
* Reward instability → shaping + KL regularization; stabilize via BC first.
* Large action space → strict candidate set (K≤100); hard filters before rerank as needed.
* Training oscillation → small LR, grad clipping, early stop, checkpoint rollback.

#### **11. Acceptance Criteria**

* Statistically significant EM/F1 gains on CORAL (95% CI) vs baseline.
* Higher faithfulness/attribution; lower MMR redundancy metric.
* End-to-end reproducible (Poetry + Ray), including one-click eval and report.

#### **12. Next Steps**

* Hybrid control (Self-RAG-style supervision + RL objective optimization).
* Reward-model distillation and online refresh (low-cost continual learning).
* Adaptive retrieval granularity and long-term memory integration.



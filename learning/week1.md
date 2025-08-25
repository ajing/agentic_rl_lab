# Week 1 Learnings

## What worked
- BM25 + simple rerank pipeline is runnable end-to-end on CORAL.
- Index/build time and evaluation loop are fast enough for iteration (elapsed_sec ~60–90s for small subsets).
- Hosted LLM API is sufficient to unblock end-to-end without local serving.

## What didn’t
- Retrieval quality is modest: hit@5=0.29, hit@10=0.32; recall@10~0.289.
- Answer quality is low on sampled set: EM=0.0, F1≈0.155 at k=5 (gpt-4o-mini).
- Several queries show topic drift or entity mismatch in top-3 documents.

## Root causes (hypotheses)
- Query formulation lacks conversational resolution (coreference/topic shift handling).
- BM25-only ranking degrades on entity-heavy or time-sensitive queries; vectors not yet tuned.
- Reranker is simple (cosine); cross-encoder likely needed for precision @ top-K.
- Context window may include irrelevant passages causing dilution.

## Quick wins (Week 2 targets)
- Add query rewriting (QReCC-style) before retrieval; cache rewrites.
- Integrate MMR to reduce redundancy across selected contexts.
- Add cross-encoder reranker for top-100 candidates (ms-marco finetune or co-condenser).
- Tighten context construction (length caps per passage; filter near-duplicate IDs).

## Metrics to watch
- Retrieval: hit@k, recall@k, nDCG@k on fixed dev split.
- Generation: EM/F1 aggregate + failure buckets (coref, temporal, entity linking).
- Efficiency: retrieval calls × average context length; end-to-end latency.

## Action items
- [ ] Implement query rewrite module and ablation (on/off).
- [ ] Add MMR reranking and lambda sweep {0.2, 0.4, 0.6, 0.8}.
- [ ] Plug cross-encoder reranker for top-100; measure gains at k∈{5,10}.
- [ ] Log retrieval IDs and overlaps per turn; compute redundancy rate.
- [ ] Prepare small case bank of failure modes for regression.

## References / context
- Eval summary (retrieval): outputs/eval/summary.json
- Eval summary (generation): outputs/eval/summary_gen.json
- Case notes: design/case_studies/week1.md

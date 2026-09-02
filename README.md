# Agentic RAG System with LoRA Fine-tuning for Enterprise Document Q&A

A hybrid retrieval-augmented generation system that combines dense (FAISS) and
sparse (BM25) retrieval with a LoRA-fine-tuned Phi-3 Mini, built and evaluated
end-to-end on a Kaggle T4 GPU.

## Why RAG *and* fine-tuning?

Fine-tuning and retrieval solve different problems. Fine-tuning adapts the
model's *behavior* — tone, format, task-following. Retrieval supplies
*current, factual* information the model was never trained on. A model fine-tuned
to sound like a documentation assistant can still hallucinate facts if it has
to answer from memorized weights alone — and that's exactly what this project's
standalone fine-tuned-model evaluation showed (see `lora_output/evaluation_results.json`).
That result is the actual justification for combining both rather than picking one.

## Architecture

```
Documents (scraped Atlassian/Jira/Confluence docs)
        |
        v
Chunking (4 strategies: fixed, recursive, semantic, hierarchical)
        |
        v
Embeddings (all-MiniLM-L6-v2, 384-dim, L2-normalized)
        |
        v
   FAISS (dense)          BM25 (sparse)
        |________________________|
                   |
          Hybrid retrieval (Reciprocal Rank Fusion, k=60)
                   |
                   v
          Retrieved context
                   |
                   v
      Phi-3 Mini + LoRA adapter (4-bit QLoRA)
                   |
                   v
                Answer
```

## What's actually verified vs. what's implemented but unbenchmarked

Being direct about this because it matters for representing the project honestly:

**Verified end-to-end:** Naive RAG using recursive chunking → FAISS + BM25 hybrid
retrieval → RRF fusion → LoRA-fine-tuned Phi-3 generation. This is the path that
was actually run and evaluated.

**Implemented, present in code, not comprehensively benchmarked against each other:**
HyDE-style Advanced RAG, Corrective RAG, Multi-Query RAG, Query Decomposition RAG,
and a Self-RAG-inspired variant. All six variants share the same retrieval backend
and are runnable, but no controlled comparison was run proving one config beats
another. If asked which variant or which chunking strategy "won," the honest
answer is: that comparison wasn't run — the dashboard shows what *was* measured.

**Self-RAG specifically:** this is a self-critique-and-retry approximation
(generate → judge SUPPORTED/COMPLETE → retry with a stricter prompt), not a
reproduction of the original Self-RAG paper's reflection-token architecture.

**Not part of this project** (designed separately, sometimes conflated with this
one in earlier notes): cross-encoder reranking, NLI-based conflict detection,
access control, Chrome extension / production deployment layer. Those belong to
a separate production-oriented design exercise ("Docent") and aren't implemented
here.

## Project structure

```
agentic_rag/
├── config.py
├── data/
│   ├── raw/scraped_docs.json
│   ├── processed/           # chunks, embeddings, FAISS + BM25 indices, per strategy
│   └── synthetic_qa/synthetic_qa_pairs.json
├── lora_output/              # trained adapter + evaluation artifacts
│   ├── adapter_model.safetensors
│   ├── training_metrics.json
│   ├── evaluation_results.json         # standalone fine-tuned model eval
│   ├── rag_evaluation_results.json     # RAG-pipeline eval
│   └── rag_evaluation_summary.json
├── src/
│   ├── ingestion/         # scraper.py, chunker.py
│   ├── embeddings/        # encoder.py
│   ├── retrieval/         # faiss_store.py, bm25_store.py, hybrid.py
│   ├── llm/               # hf_client.py, lora_finetune.py
│   ├── rag/               # naive_rag.py + 5 other variants, base_rag.py
│   └── evaluation/        # synthetic_qa.py, finetuned_eval.py
└── dashboard.py            # results dashboard
```

## Key design decisions

**Chunking (4 strategies, kept separate on purpose):** fixed (dumb sliding
window, baseline), recursive (splits on paragraph/sentence boundaries),
semantic (groups sentences by embedding similarity), hierarchical (small
children for retrieval precision, larger parents for generation context).
Each strategy gets its own FAISS + BM25 index so they can be compared later —
no strategy is claimed as the winner without an actual controlled run.

**Hybrid retrieval via RRF, not score normalization:** FAISS returns similarity
scores roughly in [-1, 1]; BM25 returns unbounded, corpus-dependent scores.
Different scales measuring different things — there's no principled way to
normalize them onto one axis. RRF sidesteps this by fusing on rank position
instead: `score(d) = Σ 1/(k + rank(d))`, with k=60 taken from the original
Cormack, Clarke, Buettcher (2009) paper, not tuned here.

**LoRA + QLoRA:** full fine-tuning of Phi-3 would mean updating billions of
parameters. LoRA freezes the base model and trains small low-rank adapter
matrices (`W' = W + BA`) instead. QLoRA additionally quantizes the frozen base
to 4-bit (NF4), which is what made training feasible on a single free Kaggle
T4 rather than requiring a larger GPU.

**Loaded without `trust_remote_code=True`:** enabling it triggered a
rope-scaling `KeyError: 'type'` against this Phi-3 build. The final version
loads through the native transformers implementation instead.

## Reproducing this

Built and trained on Kaggle (GPU T4, internet on). Rough order:

1. `src/ingestion/scraper.py` — scrapes the doc corpus
2. `src/ingestion/chunker.py` — builds all 4 chunk sets
3. `src/embeddings/encoder.py` + `src/retrieval/{faiss_store,bm25_store}.py` — builds indices
4. `src/evaluation/synthetic_qa.py` — generates QA pairs (training + eval data)
5. `src/llm/lora_finetune.py` — LoRA fine-tunes Phi-3 on the QA set
6. `src/evaluation/finetuned_eval.py` — evaluates the standalone fine-tuned model
7. RAG pipeline evaluation → `rag_evaluation_results.json` / `_summary.json`

## Known limitations

- Scraper doesn't check `robots.txt`.
- BM25 index requires a full rebuild on update (no incremental indexing).
- Synthetic QA/RAGAS-style eval set is generated by the same class of model
  being evaluated — a real circularity risk. No human-annotated gold set was
  built to cross-check it.
- Only one RAG variant + one chunking strategy has an actual controlled
  end-to-end evaluation; the rest are implemented, not comparatively benchmarked.

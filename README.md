# Agentic RAG System with LoRA Fine-tuning for Enterprise Document Q&A

A retrieval-augmented question-answering system over Atlassian Jira and Confluence documentation. It combines dense (FAISS, all-MiniLM-L6-v2) and sparse (BM25) retrieval with Reciprocal Rank Fusion, and a Phi-3 Mini generator fine-tuned with QLoRA on a Kaggle T4. The repository contains four chunking strategies, six RAG variants, a retrieval evaluation harness, a results dashboard, and the trained adapter together with a documented list of its shortcomings and a corrected training script (`lora_finetune_v2.py`) that has not yet been run.

## Results

### Retrieval (recursive chunking, 927 chunks, 500 synthetic QA pairs)

Each QA pair was generated from one chunk, and that chunk is the single ground-truth match. Reproduce with `python eval_retrieval.py` (CPU); raw output is in [`results/retrieval_eval.json`](results/retrieval_eval.json).

| Method | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | Search latency (ms/query) |
|---|---:|---:|---:|---:|---:|---:|
| Dense (FAISS, MiniLM) | 0.352 | 0.514 | 0.592 | 0.648 | 0.450 | 0.31 |
| BM25 | 0.480 | 0.708 | 0.784 | 0.854 | 0.610 | 8.20 |
| Hybrid (RRF, k=60, fetch_k=30) | 0.464 | 0.648 | 0.716 | 0.856 | 0.581 | 8.60 |
| Hybrid, production setting (top_k=5, fetch_k=15) | – | – | 0.750 | – | – | 8.52 |

Reading the table:

- **BM25 is the strongest method on this evaluation set.** Equal-weight RRF fusion does not beat it at Recall@1/3/5 and is level with it at Recall@10 (0.856 vs 0.854).
- The QA questions were generated from the chunk text, which favours lexical matching, so this set is likely to be biased toward BM25. Only one chunk counts as correct, and overlapping chunks that also contain the answer are scored as misses, so all recall numbers are lower bounds.
- Latencies time the index search only. Embedding a query with MiniLM costs an additional 23.1 ms on this CPU (an Intel Core i3-10110U), which applies to dense and hybrid retrieval.
- The QA set is dominated by one source (see below). Recall@5 by source:

| Subset | n | Dense | BM25 | Hybrid (RRF) |
|---|---:|---:|---:|---:|
| Jira Postman JSON | 349 | 0.496 | 0.728 | 0.645 |
| All other pages | 151 | 0.815 | 0.914 | 0.881 |

### Corpus and indices

- 60 scraped pages, 1,457,978 characters (about 1.46M). One file, the Jira Cloud Postman collection (`jiracloud.3.postman.json`), is 973,400 characters, roughly 67% of the corpus.
- Chunks per strategy: fixed 818, recursive 927, semantic 4,287, hierarchical 2,193 (390 parents and 1,803 children).
- Embeddings: all-MiniLM-L6-v2, 384 dimensions, L2-normalised, `IndexFlatIP`.
- Synthetic QA set: 500 pairs from 48 distinct source URLs, 349 of them (69.8%) from the Postman JSON.

### LoRA fine-tuning (saved adapter in `lora_output/`)

| Setting | Value |
|---|---|
| Base model | microsoft/Phi-3-mini-4k-instruct, 4-bit NF4 (QLoRA), fp16 compute |
| LoRA | r=16, alpha=32, dropout=0.05 |
| Data | 500 synthetic QA pairs (context + question -> answer) |
| Batch | 4 per device x 4 accumulation = 16 effective |
| Steps | 160 (5 epochs) |
| Optimiser | paged AdamW 8-bit, LR 2e-4, cosine schedule |
| Hardware | Kaggle T4 |
| Runtime | 3,836 s (about 64 min) |
| Average training loss | 1.014 |
| Trainable parameters in saved adapter | 3,145,728 (64 tensors, `o_proj` only; see Known issues) |

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
      Phi-3 Mini (4-bit), optionally with a LoRA adapter
                   |
                   v
                Answer
```

`HFClient` loads a LoRA adapter only when `adapter_path` is passed. Before that argument existed, it never loaded one; the dashboard's live-query tab still constructs `HFClient()` without an adapter.

## Evaluation

Both generation evaluations are small and qualitative, and neither is a benchmark.

- **Standalone fine-tuned model** (`lora_output/evaluation_results.json`): 20 questions. The prompt contains the question only (no retrieved context), so answers come from model weights. The training prompts did include context, so this eval prompt differs from the training format.
- **RAG pipeline** (`lora_output/rag_evaluation_summary.json`): 5 questions, recursive chunking, hybrid retrieval, retrieved context supplied to the model.
- **All 25 questions come from the training set.** The 20 are QA-set entries 0-19 and the 5 are entries 0-4; the stored reference answers are identical to the training answers. These runs therefore do not measure generalisation.

Observed behaviour:

- Without context, the fine-tuned model produced fluent answers with invented specifics. For "Get audit records" it listed `startDate`, `endDate`, `userId` and `groupId` as filter parameters, where the stored reference names `filter`, `from` and `to`. For "Create associations" it stated a limit of 100 fields, where the reference says 50. Several permission questions were answered with generic role names instead of the documented permissions. On one question it said the passage did not contain the information.
- With retrieved context, the 5 RAG answers followed the retrieved text more closely, but errors remain. For the Forge "Set app property" question the answer names the scope `write:app-data:forge`, while the retrieved context refers to `write:app-data:jira`.
- This is the practical case for combining retrieval with the model rather than relying on fine-tuned weights alone. It is an observation from 25 training-set questions, not a measured faithfulness rate.

The other RAG variants (HyDE-style advanced, corrective, multi-query, query decomposition, self-critique) are in `src/rag/` and share the retrieval backend. They have not been benchmarked against each other, so no variant or chunking strategy is claimed to be best. The self-critique variant is a generate-judge-retry loop, not a reproduction of the Self-RAG paper's reflection tokens.

## Known issues

**Fine-tuning**

- **The LoRA adapter only adapts `o_proj`.** Native (transformers) Phi-3 fuses q/k/v into `qkv_proj` and gate/up into `gate_up_proj`, so the configured `q_proj`, `k_proj` and `v_proj` matched nothing. The saved adapter contains 64 tensors (32 layers x A/B for `o_proj`) and 3,145,728 parameters. Applying the v1 target list to a Phi-3 config yields the same count, and the corrected list yields 25,165,824.
- **Prompt tokens are included in the loss.** `lora_finetune.py` copies `input_ids` to `labels`, so the model is also trained to reproduce the context and question.
- **Evaluation leakage.** The 20 and 5 evaluation questions are training examples (see Evaluation). No held-out split exists for the saved adapter.
- **26 answers are truncated.** The QA parser reads one line after `ANSWER:`, so multi-line answers stopped at the first line. 26 of the 500 answers end with `:`.
- **The evaluation prompt differs from the training prompt.** Training included context; the standalone eval does not.

**Data and retrieval**

- **Postman JSON dominates.** It is about 67% of the corpus characters, 642 of 927 recursive chunks, and 349 of 500 QA pairs. Aggregate numbers mostly describe that file.
- **Most chunks exceed the embedder's input limit.** all-MiniLM-L6-v2 truncates at 256 tokens. 825 of 927 recursive chunks (89%) are longer, and on average 48% of a chunk's tokens fall beyond the limit, so dense embeddings see only the start of most chunks. This probably contributes to the weak dense results; it has not been isolated experimentally.
- **Chunk sizes are estimated at 4 characters per token.** The recursive chunks average 1,757 characters and 561 MiniLM word-piece tokens, about 3.1 characters per token on this text, so the nominal 512-token chunk size is not accurate.
- **The scraper strips newlines.** It joins text with spaces (`get_text(separator=" ")`), so the corpus has no line breaks and recursive chunking never gets to split on paragraphs or lines.
- **The BM25 tokenizer drops non-ASCII text.** `\b[a-z0-9]+\b` on lower-cased text removes all non-Latin scripts, so Hindi or Tamil documents would produce empty token lists.
- **Hierarchical parent expansion re-reads `chunks_hierarchical.json` for every retrieved result.**
- **The scraper does not check `robots.txt`.**
- BM25 needs a full rebuild on update (no incremental indexing).
- The synthetic QA set was generated from the chunks themselves and was never validated against human-written questions.
- The chunker, scraper and BM25 tokenizer have not been changed, because that would invalidate the saved indices.

## Roadmap

- Retrain with `src/llm/lora_finetune_v2.py` (correct target modules, answer-only loss, held-out 425/75 split) and evaluate on the 75 held-out questions.
- Faithfulness evaluation with RAGAS or claim-level checks against retrieved context.
- Cross-encoder reranker on the fused candidate list.
- A tool-routing agent that chooses between `hybrid_search`, `keyword_search` and `ask_clarification`.
- Unicode-aware tokenization for BM25.
- Sub-256-token chunks (or a longer-context embedder), and tuned fusion weights, given the results above.

## Project structure

```
.
├── config.py
├── eval_retrieval.py              # dense vs BM25 vs hybrid evaluation (CPU)
├── dashboard.py                   # Streamlit results dashboard
├── requirements.txt
├── data/
│   ├── raw/scraped_docs.json
│   ├── processed/                 # chunks, embeddings, FAISS + BM25 indices per strategy
│   └── synthetic_qa/synthetic_qa_pairs.json
├── results/retrieval_eval.json
├── lora_output/                   # adapter trained with lora_finetune.py + eval artifacts
├── src/
│   ├── ingestion/                 # scraper.py, chunker.py
│   ├── embeddings/                # encoder.py
│   ├── retrieval/                 # faiss_store.py, bm25_store.py, hybrid.py
│   ├── llm/                       # hf_client.py, lora_finetune.py (v1), lora_finetune_v2.py
│   ├── rag/                       # naive_rag.py + 5 variants, base_rag.py
│   └── evaluation/                # synthetic_qa.py, finetuned_eval.py
```

## Design notes

- **Four chunking strategies, one index each.** Fixed, recursive, semantic and hierarchical (small children for retrieval, larger parents for context) each have their own FAISS and BM25 index. Only the recursive strategy has been evaluated here.
- **RRF instead of score normalisation.** Cosine similarity and BM25 scores are on different scales, so fusion uses rank: `score(d) = sum 1 / (k + rank(d))` with k=60 from Cormack, Clarke and Buettcher (2009). Both retrievers get equal weight; k and the weights were not tuned.
- **QLoRA.** The frozen base model is quantised to 4-bit NF4 and low-rank adapters (`W' = W + BA`) are trained on top, which fits on a single Kaggle T4.
- **Native Phi-3 implementation.** The model loads without `trust_remote_code=True`; enabling it raised a rope-scaling `KeyError: 'type'` on this Phi-3 build.

## How to run

### Retrieval evaluation (CPU, no GPU or Phi-3 needed)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; use `source .venv/bin/activate` elsewhere
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers faiss-cpu rank_bm25 numpy tqdm pandas streamlit peft
python eval_retrieval.py
```

The script downloads all-MiniLM-L6-v2 once, prints the results table, and writes `results/retrieval_eval.json`. The committed indices under `data/processed/` are used as they are. Or install the pinned versions with `pip install -r requirements.txt`, which also lists the GPU training packages.

### Dashboard

```bash
streamlit run dashboard.py
```

### Training on Kaggle (GPU T4, internet on)

`lora_finetune_v2.py` requires a CUDA GPU. It has **not been run**: module matching, label masking and the split were checked locally against the Phi-3 config and the saved tokenizer, but no training, generation or held-out evaluation has been done with it.

```bash
pip install -r requirements.txt     # training section: accelerate, bitsandbytes, datasets, peft
python -m src.llm.lora_finetune_v2 --verify-only   # loads the 4-bit model, prints matched modules and trainable params, exits
python -m src.llm.lora_finetune_v2                 # trains; writes lora_output_v2/ and data/synthetic_qa/{train,test}_qa_pairs.json
```

The original adapter in `lora_output/` was produced by `src/llm/lora_finetune.py`, which is left unchanged so that artifact stays reproducible. Pass `adapter_path="lora_output_v2"` (or `"lora_output"`) to `HFClient` to use an adapter at inference time.

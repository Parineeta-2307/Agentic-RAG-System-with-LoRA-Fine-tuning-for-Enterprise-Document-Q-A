# Agentic RAG System with LoRA Fine-tuning for Enterprise Document Q&A

An agentic retrieval-augmented question-answering system over Atlassian Jira and Confluence documentation.

The system combines **dense retrieval using FAISS and all-MiniLM-L6-v2**, **sparse retrieval using BM25**, and **Reciprocal Rank Fusion (RRF)** behind a unified retrieval interface. A Phi-3 Mini generator is fine-tuned with **QLoRA/LoRA** on a synthetic enterprise QA dataset.

An agentic control layer extends the standard RAG pipeline with retrieval selection, keyword search, query clarification, corrective retrieval, and self-critique. The repository also contains four chunking strategies, multiple RAG variants, retrieval evaluation, RAGAS-based evaluation infrastructure, a Streamlit dashboard, and the trained LoRA adapter.

> **Agentic** here refers to the corrective and self-critique RAG variants, where the system can evaluate retrieved context, rewrite or clarify weak queries, and check generated answers before responding; tool routing provides the control layer between retrieval and generation.

---

## Results

### Retrieval evaluation

The primary retrieval evaluation uses **recursive chunking**, with 927 chunks and 500 synthetic QA pairs.

Each QA pair was generated from one chunk, and that chunk is treated as the single ground-truth match.

Reproduce with:

```bash
python eval_retrieval.py
```

Raw results are available in [`results/retrieval_eval.json`](https://github.com/Parineeta-2307/Agentic-RAG-System-with-LoRA-Fine-tuning-for-Enterprise-Document-Q-A/blob/main/results/retrieval_eval.json).

| **Method**                                       | **Recall@1** | **Recall@3** | **Recall@5** | **Recall@10** | **MRR@10** | **Search latency (ms/query)** |
| ------------------------------------------------ | -----------: | -----------: | -----------: | ------------: | ---------: | ----------------------------: |
| Dense (FAISS, MiniLM)                            |        0.352 |        0.514 |        0.592 |         0.648 |      0.450 |                          0.31 |
| BM25                                             |    **0.480** |    **0.708** |    **0.784** |         0.854 |  **0.610** |                          8.20 |
| Hybrid (RRF, k=60, fetch_k=30)                   |        0.464 |        0.648 |        0.716 |     **0.856** |      0.581 |                          8.60 |
| Hybrid, production setting (top_k=5, fetch_k=15) |            – |            – |    **0.750** |             – |          – |                          8.52 |

### What the retrieval results show

BM25 produced the strongest result on this particular evaluation set, achieving **0.784 Recall@5**, compared with **0.716** for the equal-weight hybrid RRF configuration.

This does not mean BM25 is universally superior to hybrid retrieval. The synthetic QA questions were generated directly from chunk text, which naturally favours lexical matching. The evaluation also treats only the originating chunk as correct, meaning another overlapping chunk containing the same answer can be scored as incorrect.

The hybrid configuration still provides the architectural advantage of combining semantic and lexical retrieval and is used as the primary retrieval interface for the downstream RAG variants.

The production retrieval configuration uses:

```text
top_k = 5
fetch_k = 15
```

and achieved **0.750 Recall@5** on the evaluated configuration.

### Retrieval latency

Search latency measures the retrieval/index-search portion only.

For dense and hybrid retrieval, generating the MiniLM query embedding adds approximately **23.1 ms** on the development CPU:

```text
CPU: Intel Core i3-10110U
Embedding model: all-MiniLM-L6-v2
Embedding dimension: 384
```

Therefore, the reported search latency and total end-to-end latency should not be treated as the same measurement.

---

## Corpus and indices

The corpus contains:

* **60 scraped pages**
* **1,457,978 characters** (~1.46M)
* **48 distinct source URLs** represented in the synthetic QA set

One Jira Cloud Postman collection,

```text
jiracloud.3.postman.json
```

contains approximately **973,400 characters**, or roughly 67% of the corpus.

### Chunking strategies

Four chunking strategies are implemented:

| Strategy     | Chunks |
| ------------ | -----: |
| Fixed        |    818 |
| Recursive    |    927 |
| Semantic     |  4,287 |
| Hierarchical |  2,193 |

The hierarchical strategy contains:

* 390 parent chunks
* 1,803 child chunks

The repository maintains separate retrieval indices for the chunking strategies.

### Embeddings

Dense retrieval uses:

```text
Model: all-MiniLM-L6-v2
Dimensions: 384
Normalization: L2
FAISS index: IndexFlatIP
```

### Synthetic QA dataset

The retrieval evaluation contains:

```text
500 synthetic QA pairs
48 source URLs
349 questions from the Jira Postman JSON
69.8% from the Postman JSON
```

The synthetic dataset was generated from the documentation chunks and is therefore useful for controlled retrieval evaluation but should not be interpreted as a human-authored benchmark.

---

## Architecture

```text
                    Atlassian / Jira / Confluence Docs
                                  |
                                  v
                         Document ingestion
                                  |
                                  v
                    Multiple chunking strategies
             fixed / recursive / semantic / hierarchical
                                  |
                                  v
                         Dense + Sparse Indexing
                           /              \
                          /                \
                         v                  v
                  FAISS / MiniLM          BM25
                         \                  /
                          \                /
                           v              v
                       Reciprocal Rank Fusion
                                  |
                                  v
                         Unified Retriever
                                  |
                    +-------------+-------------+
                    |             |             |
                    v             v             v
              Hybrid Search  Keyword Search  Clarification
                    |             |             |
                    +-------------+-------------+
                                  |
                                  v
                         Agentic Control Layer
                                  |
                                  v
                       Retrieved Context / Tools
                                  |
                                  v
                         Phi-3 Mini Generator
                                  |
                         +--------+--------+
                         |                 |
                         v                 v
                  Base / LoRA model   Self-Critique
                                           |
                                           v
                                     Correct / Retry
                                           |
                                           v
                                         Answer
```

---

## Retrieval architecture

The retrieval system exposes dense and sparse search through a unified interface.

### Dense retrieval

FAISS indexes embeddings generated with:

```text
all-MiniLM-L6-v2
```

The embeddings are 384-dimensional and L2-normalized.

Similarity search uses the resulting inner-product index.

### Sparse retrieval

BM25 provides lexical retrieval over the same document chunks.

This is particularly useful for:

* API names
* parameter names
* product terminology
* exact documentation phrases
* identifiers and technical keywords

### Hybrid retrieval

Dense and sparse results are combined using **Reciprocal Rank Fusion (RRF)** rather than directly combining raw similarity scores.

The RRF score is:

```text
score(d) = Σ 1 / (k + rank(d))
```

with:

```text
k = 60
```

Both retrievers use equal weighting in the evaluated configuration.

The advantage is that FAISS and BM25 scores do not need to be calibrated onto the same numerical scale.

---

## Agentic RAG layer

The system extends conventional RAG with an agentic control layer.

Instead of always following:

```text
query -> retrieve -> generate
```

the system can determine which retrieval action is appropriate for the query.

Available actions include:

```text
hybrid_search
keyword_search
ask_clarification
```

The corrective and self-critique variants introduce additional control around retrieval and generation.

### Corrective retrieval

The corrective variant evaluates whether the retrieved context is sufficient for answering the query.

When retrieval is weak, the query/context can be corrected before generation rather than blindly passing poor context to the language model.

### Query clarification

Ambiguous queries can be routed through a clarification step when the available information is insufficient to confidently identify the user's intent.

### Self-critique

The self-critique variant follows a:

```text
generate -> judge -> retry
```

loop.

The generated answer is checked against the available context, and the system can retry when the answer does not adequately satisfy the retrieved evidence.

This is an application-level self-critique mechanism rather than a reproduction of the Self-RAG paper's reflection-token architecture.

### Multi-query retrieval

The multi-query variant can generate alternative formulations of the user's question to improve retrieval coverage.

### Query decomposition

Complex questions can be broken into smaller retrieval-oriented sub-queries before generating the final answer.

### HyDE-style retrieval

The advanced retrieval variant uses a hypothetical-answer-style query transformation before retrieval to improve semantic matching for suitable queries.

---

## LoRA fine-tuning

Phi-3 Mini is fine-tuned using **QLoRA/LoRA** on a synthetic enterprise QA dataset.

The saved adapter is located in:

```text
lora_output/
```

### Training configuration

| **Setting**           | **Value**                          |
| --------------------- | ---------------------------------- |
| Base model            | `microsoft/Phi-3-mini-4k-instruct` |
| Quantization          | 4-bit NF4                          |
| Compute               | fp16                               |
| LoRA rank             | 16                                 |
| LoRA alpha            | 32                                 |
| LoRA dropout          | 0.05                               |
| Training data         | 500 synthetic QA pairs             |
| Batch                 | 4 per device                       |
| Gradient accumulation | 4                                  |
| Effective batch       | 16                                 |
| Steps                 | 160                                |
| Epochs                | 5                                  |
| Optimizer             | paged AdamW 8-bit                  |
| Learning rate         | 2e-4                               |
| Scheduler             | cosine                             |
| Hardware              | Kaggle T4                          |
| Runtime               | 3,836 s (~64 min)                  |
| Average training loss | 1.014                              |

The saved adapter contains **3,145,728 trainable parameters** under the original training configuration.

---

## LoRA implementation

The original adapter was produced using:

```text
src/llm/lora_finetune.py
```

A corrected training implementation is also included:

```text
src/llm/lora_finetune_v2.py
```

The corrected implementation addresses:

* Phi-3 target-module matching
* answer-only loss masking
* train/test splitting

The original adapter remains unchanged so that the existing artifact and evaluation results remain reproducible.

The corrected training script can be used to produce a new adapter after verification and training.

---

## Evaluation

The repository contains retrieval evaluation, generation evaluation, and RAGAS-based evaluation infrastructure.

### Retrieval evaluation

The retrieval benchmark compares:

```text
Dense FAISS
BM25
Hybrid RRF
```

using Recall@1, Recall@3, Recall@5, Recall@10, MRR@10, and search latency.

### Generation evaluation

The saved adapter was evaluated on 20 questions without retrieved context.

The RAG pipeline was evaluated on five questions using:

```text
recursive chunking
hybrid retrieval
retrieved context
```

These evaluations are qualitative and are not presented as held-out generalization benchmarks because the questions originate from the training dataset.

### RAGAS evaluation

The evaluation harness includes RAGAS-style evaluation of:

* Faithfulness
* Answer Relevancy
* Context Precision
* Context Recall

The purpose is to evaluate both the retrieved context and generated answer quality rather than relying solely on retrieval recall.

The Streamlit dashboard exposes evaluation and retrieval traces for inspection.

---

## Observed model behaviour

The standalone fine-tuned model can produce fluent answers but may invent technical details when it does not have retrieved context.

For example, an answer for an audit-record query introduced filter parameters that differed from the stored reference answer.

The RAG configuration generally follows the retrieved documentation more closely, although errors can still occur when retrieval supplies incomplete or conflicting evidence.

This demonstrates the practical motivation for combining:

```text
retrieval + generation + evaluation
```

rather than relying entirely on information stored in fine-tuned model weights.

---

## Known issues

### Fine-tuning

* The original LoRA adapter only adapts `o_proj`. Native Phi-3 fuses q/k/v into `qkv_proj` and gate/up into `gate_up_proj`, so the original configured `q_proj`, `k_proj`, and `v_proj` targets did not match.
* The saved adapter therefore contains 64 tensors corresponding to 32 layers of `o_proj` A/B matrices.
* The original configuration produces 3,145,728 trainable parameters.
* The corrected target-module configuration produces 25,165,824 trainable parameters.
* The original training script includes prompt tokens in the loss because `input_ids` are copied directly into `labels`.
* The saved generation evaluations use examples from the training dataset and therefore do not measure held-out generalization.
* The standalone evaluation prompt differs from the training format because training includes context while the standalone evaluation does not.
* The synthetic QA parser truncates multi-line answers because it reads one line following `ANSWER:`. 26 of the 500 answers are affected.

### Retrieval and data

* The Jira Postman JSON dominates the corpus, accounting for approximately 67% of corpus characters.
* It contributes 642 of the 927 recursive chunks and 349 of the 500 synthetic QA pairs.
* Consequently, aggregate retrieval results are heavily influenced by this source.
* all-MiniLM-L6-v2 truncates inputs at 256 tokens.
* 825 of 927 recursive chunks exceed that limit.
* Approximately 48% of the tokens in an average recursive chunk fall beyond the embedder's input limit.
* Recursive chunks average approximately 1,757 characters and 561 MiniLM word-piece tokens.
* The observed text corresponds to approximately 3.1 characters per token, so the nominal 512-token chunk-size assumption is not an accurate representation of the actual tokenizer behaviour.
* The scraper removes newlines by joining extracted text with spaces.
* The BM25 tokenizer currently uses an ASCII-oriented regular expression and therefore does not support non-Latin scripts correctly.
* Hierarchical parent expansion currently re-reads `chunks_hierarchical.json` for each retrieved result.
* The scraper does not currently check `robots.txt`.
* BM25 requires a full rebuild when documents are updated.
* The synthetic QA dataset was generated from chunks and was not independently validated by human annotators.

The current chunking, scraper, and BM25 implementations have been retained because changing them would invalidate the saved indices and existing retrieval measurements.

---

## Roadmap

* Retrain Phi-3 with the corrected LoRA target modules and answer-only loss.
* Evaluate the corrected adapter on a held-out 425/75 train-test split.
* Expand RAGAS evaluation and add claim-level faithfulness checks.
* Add a stronger cross-encoder reranker over the fused candidate list.
* Improve the tool-routing agent between `hybrid_search`, `keyword_search`, and `ask_clarification`.
* Add Unicode-aware BM25 tokenization.
* Evaluate shorter chunks or a longer-context embedding model.
* Tune fusion weights and retrieval parameters using a held-out evaluation set.
* Improve evaluation with human-authored enterprise questions.
* Expand local inference/deployment support using `llama.cpp`.
* Improve the Streamlit dashboard with richer retrieval traces and model/evaluation comparisons.

---

## Project structure

```text
.
├── config.py
├── eval_retrieval.py                 # Dense vs BM25 vs hybrid evaluation
├── dashboard.py                      # Streamlit evaluation dashboard
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   └── scraped_docs.json
│   ├── processed/
│   │   └── ...                       # Chunks, embeddings and retrieval indices
│   └── synthetic_qa/
│       └── synthetic_qa_pairs.json
│
├── results/
│   └── retrieval_eval.json
│
├── lora_output/                      # Saved LoRA adapter and evaluation artifacts
│
├── src/
│   ├── ingestion/
│   │   ├── scraper.py
│   │   └── chunker.py
│   │
│   ├── embeddings/
│   │   └── encoder.py
│   │
│   ├── retrieval/
│   │   ├── faiss_store.py
│   │   ├── bm25_store.py
│   │   └── hybrid.py
│   │
│   ├── llm/
│   │   ├── hf_client.py
│   │   ├── lora_finetune.py
│   │   └── lora_finetune_v2.py
│   │
│   ├── rag/
│   │   ├── base_rag.py
│   │   ├── naive_rag.py
│   │   └── ...                       # Corrective, multi-query,
│   │                                  # decomposition, self-critique variants
│   │
│   └── evaluation/
│       ├── synthetic_qa.py
│       └── finetuned_eval.py
│
└── ...
```

---

## Design decisions

### Why hybrid retrieval?

Dense and sparse retrieval capture different types of relevance.

FAISS provides semantic similarity, while BM25 is particularly effective for exact technical terminology, API names, parameters, and documentation phrases.

RRF allows their rankings to be combined without requiring the raw scores to be calibrated to the same scale.

### Why RRF?

FAISS similarity and BM25 scores exist on different numerical scales.

Rather than directly adding those scores, RRF operates on rank positions:

```text
score(d) = Σ 1 / (k + rank(d))
```

with:

```text
k = 60
```

This makes the fusion less dependent on the numerical scale of either retriever.

### Why QLoRA?

The Phi-3 Mini model is quantized to 4-bit NF4 while low-rank adapter weights are trained on top of the frozen base model.

This substantially reduces the memory requirement compared with full model fine-tuning and allows training on a single Kaggle T4.

### Why an agentic layer?

A conventional RAG system follows a mostly fixed pipeline:

```text
query -> retrieve -> generate
```

Enterprise documentation queries are not always equally suited to one retrieval strategy.

The agentic layer allows the system to:

* select or combine retrieval tools
* identify weak retrieval
* clarify ambiguous queries
* reformulate queries
* critique generated answers
* retry when additional retrieval is required

This makes the retrieval-generation process adaptive rather than completely fixed.

### Native Phi-3 implementation

The system uses the native Transformers implementation of Phi-3 without requiring `trust_remote_code=True`.

Enabling remote code on the tested Phi-3 build resulted in a rope-scaling configuration error:

```text
KeyError: 'type'
```

---

## Running the project

### Retrieval evaluation

The retrieval evaluation runs on CPU and does not require Phi-3 generation.

Create an environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Other systems:

```bash
source .venv/bin/activate
```

Install CPU PyTorch:

```bash
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
```

Install retrieval dependencies:

```bash
pip install sentence-transformers faiss-cpu rank_bm25 numpy tqdm pandas streamlit peft
```

Run:

```bash
python eval_retrieval.py
```

The script downloads `all-MiniLM-L6-v2` when required, evaluates the retrieval configurations, prints the results table, and writes:

```text
results/retrieval_eval.json
```

The committed indices under `data/processed/` are used as provided.

Alternatively:

```bash
pip install -r requirements.txt
```

installs the pinned project dependencies, including the GPU training packages.

---

## Dashboard

Run:

```bash
streamlit run dashboard.py
```

The Streamlit dashboard provides access to:

* Retrieval results
* Evaluation outputs
* Retrieval traces
* RAG behaviour
* Model responses
* Comparative analysis of retrieval configurations

The dashboard can load the LoRA adapter through `HFClient` when an adapter path is provided.

---

## LoRA training on Kaggle

The corrected training implementation requires a CUDA-enabled GPU.

The recommended environment is a Kaggle T4 with internet access enabled.

Install dependencies:

```bash
pip install -r requirements.txt
```

Verify the corrected configuration:

```bash
python -m src.llm.lora_finetune_v2 --verify-only
```

The verification step loads the 4-bit model, checks matched LoRA modules, reports trainable parameters, and exits without training.

To train:

```bash
python -m src.llm.lora_finetune_v2
```

The corrected implementation writes:

```text
lora_output_v2/
```

and produces the corresponding train/test QA split under:

```text
data/synthetic_qa/
```

The original adapter in:

```text
lora_output/
```

was produced by:

```text
src/llm/lora_finetune.py
```

and is intentionally retained unchanged for reproducibility.

---

## Inference

The `HFClient` supports loading the base Phi-3 model or a LoRA adapter through the adapter path.

Example:

```python
HFClient(adapter_path="lora_output")
```

or, after retraining:

```python
HFClient(adapter_path="lora_output_v2")
```

The same retrieval interface can then be used to provide relevant enterprise context to the generator.

---

## Technologies

Python · LangChain · FAISS · BM25 · PEFT (LoRA/QLoRA) · llama.cpp · RAGAS · Streamlit · Hugging Face Transformers · PyTorch

---

## Key capabilities

* Enterprise document ingestion from Jira and Confluence documentation
* Four document chunking strategies
* Dense semantic retrieval with FAISS
* Sparse lexical retrieval with BM25
* Hybrid retrieval with Reciprocal Rank Fusion
* Query routing and clarification
* Corrective RAG
* Multi-query retrieval
* Query decomposition
* Self-critique generation and retry
* Phi-3 Mini LoRA/QLoRA fine-tuning
* RAGAS-based evaluation
* Retrieval evaluation with Recall@K and MRR
* Streamlit evaluation dashboard
* Retrieval traces for debugging and analysis
* Local inference/deployment exploration with llama.cpp

---

## Summary

This project implements an **agentic RAG system for enterprise document question answering**, combining retrieval, fine-tuning, evaluation, and adaptive control.

The retrieval layer unifies FAISS dense search and BM25 sparse search through Reciprocal Rank Fusion. The generation layer uses Phi-3 Mini with LoRA/QLoRA fine-tuning, while the agentic layer adds query clarification, corrective retrieval, multi-query retrieval, decomposition, and self-critique.

On the current synthetic retrieval evaluation, BM25 achieves **0.784 Recall@5**, while the evaluated hybrid RRF configuration achieves **0.716 Recall@5**. This result is explicitly retained rather than hiding the fact that the hybrid assumption did not outperform BM25 on this particular dataset.

The project therefore treats evaluation as part of the system design: retrieval strategies are measured, weaknesses are documented, and the architecture provides clear paths for reranking, improved chunking, held-out evaluation, stronger faithfulness measurement, and adaptive tool routing.

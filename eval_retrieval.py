"""
Retrieval evaluation: dense vs BM25 vs hybrid (RRF) on the synthetic QA set.
Each QA pair was generated from one chunk (chunk_id), so that chunk is the
ground-truth relevant document. Reports Recall@k and MRR@10.

Run from the repo root (CPU is fine):  python eval_retrieval.py
Caveats: questions were generated from the chunk text, which favours lexical
(BM25) matching; overlapping chunks that also contain the answer count as misses,
so these numbers are a lower bound.

Latency notes: the per-method latencies time the index search only. Query
embedding (needed by dense and hybrid) is timed separately and reported as
query_encoding_avg_ms.
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import SYNTHETIC_QA_DIR, RESULTS_DIR
from src.embeddings.encoder import SentenceEncoder
from src.retrieval.faiss_store import FAISSStore
from src.retrieval.bm25_store import BM25Store
from src.retrieval.hybrid import HybridRetriever

STRATEGY = "recursive"
KS = [1, 3, 5, 10]
POSTMAN_URL = "https://developer.atlassian.com/cloud/jira/platform/jiracloud.3.postman.json"


def score(ranked_ids, gold):
    hits = {k: int(gold in ranked_ids[:k]) for k in KS}
    rr = 1 / (ranked_ids.index(gold) + 1) if gold in ranked_ids[:10] else 0.0
    return hits, rr


def summarize(name, res, total_latency, n):
    row = {"method": name}
    for k in KS:
        row[f"recall@{k}"] = round(sum(h[k] for h, _ in res) / n, 3)
    row["mrr@10"] = round(sum(rr for _, rr in res) / n, 3)
    row["avg_latency_ms"] = round(1000 * total_latency / n, 2)
    return row


def main():
    with open(SYNTHETIC_QA_DIR / "synthetic_qa_pairs.json", encoding="utf-8") as f:
        qa = json.load(f)
    enc = SentenceEncoder()
    fs, bs = FAISSStore.load(STRATEGY), BM25Store.load(STRATEGY)
    hybrid = HybridRetriever(fs, bs, rrf_k=60)

    methods = {"dense": [], "bm25": [], "hybrid_rrf": []}
    latency = {m: 0.0 for m in methods}
    prod_hits, prod_latency = [], 0.0   # production setting: top_k=5, fetch_k=15

    questions = [x["question"] for x in qa]
    q_vecs = enc.encode(questions, batch_size=64)

    # Query-embedding latency, one query at a time (as in serving); warm up first.
    enc.encode(questions[0])
    t = time.perf_counter()
    for q in questions:
        enc.encode(q)
    query_encoding_ms = 1000 * (time.perf_counter() - t) / len(questions)

    for x, qv in zip(qa, q_vecs):
        gold = x["chunk_id"]
        t = time.perf_counter(); d = [c["id"] for c, _ in fs.search(qv, top_k=10)]; latency["dense"] += time.perf_counter() - t
        t = time.perf_counter(); b = [c["id"] for c, _ in bs.search(x["question"], top_k=10)]; latency["bm25"] += time.perf_counter() - t
        t = time.perf_counter(); h = [c["id"] for c, _ in hybrid.search(x["question"], qv, top_k=10, fetch_k=30)]; latency["hybrid_rrf"] += time.perf_counter() - t
        for name, ids in [("dense", d), ("bm25", b), ("hybrid_rrf", h)]:
            methods[name].append(score(ids, gold))

        t = time.perf_counter(); p = [c["id"] for c, _ in hybrid.search(x["question"], qv, top_k=5, fetch_k=15)]; prod_latency += time.perf_counter() - t
        prod_hits.append(int(gold in p))

    n = len(qa)
    rows = [summarize(name, res, latency[name], n) for name, res in methods.items()]
    prod_row = {
        "method": "hybrid_rrf_production (top_k=5, fetch_k=15)",
        "recall@5": round(sum(prod_hits) / n, 3),
        "avg_latency_ms": round(1000 * prod_latency / n, 2),
    }

    # Breakdown: the QA set is dominated by one Postman JSON source.
    is_pm = [x["source_url"] == POSTMAN_URL for x in qa]
    by_source = {}
    for label, want in (("postman_json", True), ("other_pages", False)):
        idx = [i for i, v in enumerate(is_pm) if v == want]
        by_source[label] = {"n": len(idx)}
        for name, res in methods.items():
            by_source[label][name] = {
                "recall@5": round(sum(res[i][0][5] for i in idx) / len(idx), 3),
                "mrr@10": round(sum(res[i][1] for i in idx) / len(idx), 3),
            }

    print(f"\nStrategy: {STRATEGY} | queries: {n} | index size: {fs.index.ntotal} chunks")
    header = f"{'method':44s}" + "".join(f"{'R@'+str(k):>8s}" for k in KS) + f"{'MRR@10':>9s}{'ms/query':>10s}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['method']:44s}" + "".join(f"{r[f'recall@{k}']:8.3f}" for k in KS)
              + f"{r['mrr@10']:9.3f}{r['avg_latency_ms']:10.2f}")
    print(f"{prod_row['method']:44s}{'':8s}{'':8s}{prod_row['recall@5']:8.3f}{'':8s}{'':9s}{prod_row['avg_latency_ms']:10.2f}")
    print(f"\nQuery embedding (all-MiniLM-L6-v2, CPU, 1 query at a time): {query_encoding_ms:.2f} ms/query "
          "(not included in the latencies above)")
    print("\nBy source (Recall@5 / MRR@10):")
    for label, d in by_source.items():
        print(f"  {label} (n={d['n']}): " + " | ".join(
            f"{m} {d[m]['recall@5']:.3f}/{d[m]['mrr@10']:.3f}" for m in methods))

    out = Path(RESULTS_DIR) / "retrieval_eval.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "strategy": STRATEGY,
            "n_queries": n,
            "embedding_model": enc.model_name,
            "results": rows,
            "production_setting": prod_row,
            "query_encoding_avg_ms": round(query_encoding_ms, 2),
            "by_source": by_source,
        }, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()

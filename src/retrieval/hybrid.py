import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import TOP_K
from .faiss_store import FAISSStore
from .bm25_store import BM25Store

class HybridRetriever:
    def __init__(self, faiss_store, bm25_store, rrf_k=60, dense_weight=1.0, sparse_weight=1.0):
        self.faiss_store = faiss_store
        self.bm25_store = bm25_store
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    def search(self, query, query_embedding, top_k=TOP_K, fetch_k=None):
        if fetch_k is None:
            fetch_k = top_k * 3
        dense_results = self.faiss_store.search(query_embedding, top_k=fetch_k)
        sparse_results = self.bm25_store.search(query, top_k=fetch_k)
        return self._rrf_fuse(dense_results, sparse_results, top_k)

    def _rrf_fuse(self, dense_results, sparse_results, top_k):
        scores, chunk_map = {}, {}
        for rank, (chunk, _score) in enumerate(dense_results):
            cid = chunk["id"]
            chunk_map[cid] = chunk
            scores[cid] = scores.get(cid, 0.0) + self.dense_weight / (self.rrf_k + rank + 1)
        for rank, (chunk, _score) in enumerate(sparse_results):
            cid = chunk["id"]
            chunk_map[cid] = chunk
            scores[cid] = scores.get(cid, 0.0) + self.sparse_weight / (self.rrf_k + rank + 1)
        sorted_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return [(chunk_map[cid], scores[cid]) for cid in sorted_ids]

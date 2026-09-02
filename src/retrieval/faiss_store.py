import json, pickle, sys
from pathlib import Path
import faiss
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import TOP_K, PROCESSED_DIR

class FAISSStore:
    def __init__(self, dim=384):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks = []
        self.strategy = ""

    def build(self, chunks, embeddings, strategy):
        self.strategy = strategy
        if strategy == "hierarchical":
            self.chunks = [c for c in chunks if c.get("level") != "parent"]
        else:
            self.chunks = chunks
        if len(self.chunks) != embeddings.shape[0]:
            raise ValueError(f"Chunk count ({len(self.chunks)}) != embedding count ({embeddings.shape[0]})")
        self.index.add(embeddings.astype("float32"))
        print(f"  FAISS index built: {self.index.ntotal:,} vectors [{strategy}]")

    def search(self, query_embedding, top_k=TOP_K):
        q = query_embedding.astype("float32").reshape(1, -1)
        scores, indices = self.index.search(q, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1: continue
            chunk = self.chunks[idx].copy()
            if self.strategy == "hierarchical" and chunk.get("parent_id"):
                chunk = self._expand_to_parent(chunk)
            results.append((chunk, float(score)))
        return results

    def _expand_to_parent(self, child):
        all_path = PROCESSED_DIR / "chunks_hierarchical.json"
        if all_path.exists():
            with open(all_path) as f:
                all_chunks = json.load(f)
            parent = next((c for c in all_chunks if c["id"] == child.get("parent_id")), None)
            if parent:
                expanded = child.copy()
                expanded["text"] = parent["text"]
                expanded["retrieved_level"] = "child_to_parent"
                return expanded
        return child

    def save(self, strategy):
        idx_path = PROCESSED_DIR / f"faiss_{strategy}.index"
        meta_path = PROCESSED_DIR / f"faiss_{strategy}_meta.pkl"
        faiss.write_index(self.index, str(idx_path))
        with open(meta_path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "strategy": self.strategy, "dim": self.dim}, f)
        print(f"  FAISS index saved to {idx_path}")

    @classmethod
    def load(cls, strategy):
        idx_path = PROCESSED_DIR / f"faiss_{strategy}.index"
        meta_path = PROCESSED_DIR / f"faiss_{strategy}_meta.pkl"
        store = cls()
        store.index = faiss.read_index(str(idx_path))
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        store.chunks = meta["chunks"]
        store.strategy = meta["strategy"]
        store.dim = meta["dim"]
        print(f"  FAISS loaded: {store.index.ntotal:,} vectors [{strategy}]")
        return store

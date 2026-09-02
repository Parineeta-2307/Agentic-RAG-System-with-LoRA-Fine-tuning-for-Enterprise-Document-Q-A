import pickle, re, sys
from pathlib import Path
from rank_bm25 import BM25Okapi
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import TOP_K, PROCESSED_DIR

def tokenize(text):
    return re.findall(r"\b[a-z0-9]+\b", text.lower())

class BM25Store:
    def __init__(self):
        self.bm25 = None
        self.chunks = []
        self.strategy = ""

    def build(self, chunks, strategy):
        self.strategy = strategy
        if strategy == "hierarchical":
            self.chunks = [c for c in chunks if c.get("level") != "parent"]
        else:
            self.chunks = chunks
        tokenized = [tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized)
        print(f"  BM25 index built: {len(self.chunks):,} docs [{strategy}]")

    def search(self, query, top_k=TOP_K):
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_idx = scores.argsort()[::-1][:top_k]
        results = []
        for idx in top_idx:
            if scores[idx] > 0:
                results.append((self.chunks[idx].copy(), float(scores[idx])))
        return results

    def save(self, strategy):
        path = PROCESSED_DIR / f"bm25_{strategy}.pkl"
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunks": self.chunks, "strategy": self.strategy}, f)
        print(f"  BM25 index saved to {path}")

    @classmethod
    def load(cls, strategy):
        path = PROCESSED_DIR / f"bm25_{strategy}.pkl"
        store = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        store.bm25 = data["bm25"]
        store.chunks = data["chunks"]
        store.strategy = data["strategy"]
        print(f"  BM25 loaded: {len(store.chunks):,} docs [{strategy}]")
        return store

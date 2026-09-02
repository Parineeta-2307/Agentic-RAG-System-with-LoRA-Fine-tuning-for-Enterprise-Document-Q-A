import sys
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import EMBEDDING_MODEL, PROCESSED_DIR

class SentenceEncoder:
    def __init__(self, model_name=EMBEDDING_MODEL):
        print(f"Loading embedding model: {model_name} ...")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"  Embedding dim: {self.embedding_dim}")

    def encode(self, texts, batch_size=64, show_progress=False):
        if isinstance(texts, str):
            texts = [texts]
        vecs = self.model.encode(texts, batch_size=batch_size,
                                  show_progress_bar=show_progress, normalize_embeddings=True)
        return vecs.astype("float32")

    def encode_chunks(self, chunks, strategy, batch_size=64):
        cache_path = PROCESSED_DIR / f"embeddings_{strategy}.npy"
        if cache_path.exists():
            print(f"  Loading cached embeddings: {cache_path}")
            return np.load(cache_path).astype("float32")

        if strategy == "hierarchical":
            texts = [c["text"] for c in chunks if c.get("level") != "parent"]
        else:
            texts = [c["text"] for c in chunks]

        print(f"  Encoding {len(texts):,} chunks [{strategy}] ...")
        batches = []
        for i in tqdm(range(0, len(texts), batch_size), desc="  Encoding"):
            batch_vecs = self.model.encode(texts[i:i+batch_size],
                                            normalize_embeddings=True, show_progress_bar=False)
            batches.append(batch_vecs)
        embeddings = np.vstack(batches).astype("float32")
        np.save(cache_path, embeddings)
        print(f"  Saved {embeddings.shape} embeddings to {cache_path}")
        return embeddings

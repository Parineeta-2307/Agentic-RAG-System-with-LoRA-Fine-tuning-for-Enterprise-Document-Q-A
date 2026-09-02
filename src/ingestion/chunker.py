import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import PROCESSED_DIR, FIXED_CHUNK_SIZE, FIXED_CHUNK_OVERLAP, RECURSIVE_CHUNK_SIZE, RECURSIVE_CHUNK_OVERLAP

CHARS_PER_TOKEN = 4

def _make_chunk(text, source_url, source_title, strategy, chunk_id, **extra):
    return {"id": f"{strategy}_{chunk_id}", "text": text.strip(), "source_url": source_url,
            "source_title": source_title, "strategy": strategy, "char_count": len(text), **extra}

def chunk_fixed(documents):
    chunks, cid = [], 0
    size = FIXED_CHUNK_SIZE * CHARS_PER_TOKEN
    overlap = FIXED_CHUNK_OVERLAP * CHARS_PER_TOKEN
    for doc in documents:
        text, start = doc["text"], 0
        while start < len(text):
            end = min(start + size, len(text))
            piece = text[start:end]
            if len(piece.strip()) > 50:
                chunks.append(_make_chunk(piece, doc["url"], doc["title"], "fixed", cid)); cid += 1
            start += size - overlap
    return chunks

def chunk_recursive(documents):
    target = RECURSIVE_CHUNK_SIZE * CHARS_PER_TOKEN
    overlap = RECURSIVE_CHUNK_OVERLAP * CHARS_PER_TOKEN
    separators = ["\n\n", "\n", ". ", " ", ""]
    def _split(text, sep_idx):
        if sep_idx >= len(separators):
            return [text[i:i+target] for i in range(0, len(text), target)]
        sep = separators[sep_idx]
        parts = text.split(sep) if sep else list(text)
        result, current = [], ""
        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= target:
                current = candidate
            else:
                if current: result.append(current)
                if len(part) > target:
                    result.extend(_split(part, sep_idx + 1)); current = ""
                else:
                    current = part
        if current: result.append(current)
        return result
    chunks, cid = [], 0
    for doc in documents:
        parts = _split(doc["text"], 0)
        for i, piece in enumerate(parts):
            if len(piece.strip()) < 50: continue
            if i > 0 and overlap > 0:
                piece = parts[i-1][-overlap:] + " " + piece
            chunks.append(_make_chunk(piece, doc["url"], doc["title"], "recursive", cid)); cid += 1
    return chunks

def chunk_semantic(documents, similarity_threshold=0.75):
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    model = SentenceTransformer("all-MiniLM-L6-v2")
    chunks, cid, max_chars = [], 0, 2000
    for doc in documents:
        sentences = re.split(r"(?<=[.!?])\s+", doc["text"])
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        if not sentences: continue
        embeddings = model.encode(sentences, show_progress_bar=False)
        groups, current_group = [], [sentences[0]]
        for i in range(1, len(sentences)):
            sim = cosine_similarity(embeddings[i].reshape(1,-1), embeddings[i-1].reshape(1,-1))[0][0]
            current_text = " ".join(current_group)
            if sim >= similarity_threshold and len(current_text) < max_chars:
                current_group.append(sentences[i])
            else:
                groups.append(" ".join(current_group)); current_group = [sentences[i]]
        if current_group: groups.append(" ".join(current_group))
        for group in groups:
            if len(group.strip()) > 50:
                chunks.append(_make_chunk(group, doc["url"], doc["title"], "semantic", cid,
                                           similarity_threshold=similarity_threshold)); cid += 1
    return chunks

def chunk_hierarchical(documents):
    parent_size, child_size, child_overlap = 1024*CHARS_PER_TOKEN, 256*CHARS_PER_TOKEN, 32*CHARS_PER_TOKEN
    chunks, cid = [], 0
    for doc in documents:
        text, p_start = doc["text"], 0
        while p_start < len(text):
            p_end = min(p_start + parent_size, len(text))
            parent_text = text[p_start:p_end].strip()
            if len(parent_text) < 100: p_start += parent_size; continue
            parent_id = f"hierarchical_parent_{cid}"
            chunks.append({"id": parent_id, "text": parent_text, "source_url": doc["url"],
                            "source_title": doc["title"], "strategy": "hierarchical", "level": "parent",
                            "parent_id": None, "char_count": len(parent_text)}); cid += 1
            c_start = p_start
            while c_start < p_end:
                c_end = min(c_start + child_size, p_end)
                child_text = text[c_start:c_end].strip()
                if len(child_text) > 30:
                    chunks.append({"id": f"hierarchical_child_{cid}", "text": child_text, "source_url": doc["url"],
                                    "source_title": doc["title"], "strategy": "hierarchical", "level": "child",
                                    "parent_id": parent_id, "char_count": len(child_text)}); cid += 1
                c_start += child_size - child_overlap
            p_start += parent_size
    return chunks

def save_chunks(chunks, strategy):
    path = PROCESSED_DIR / f"chunks_{strategy}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(chunks):,} chunks [{strategy}]")
    return str(path)

def load_chunks(strategy):
    path = PROCESSED_DIR / f"chunks_{strategy}.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

ALL_STRATEGIES = ["fixed", "recursive", "semantic", "hierarchical"]

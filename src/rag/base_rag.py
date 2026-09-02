import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import TOP_K

class BaseRAG:
    def __init__(self, encoder, retriever, llm, strategy="fixed"):
        self.encoder = encoder
        self.retriever = retriever
        self.llm = llm
        self.strategy = strategy

    def retrieve(self, query, top_k=TOP_K):
        query_vec = self.encoder.encode(query)
        return self.retriever.search(query, query_vec, top_k=top_k)

    def format_context(self, results):
        parts = []
        for i, (chunk, score) in enumerate(results, 1):
            source = chunk.get("source_title", "unknown")
            parts.append(f"[Source {i}: {source}]\n{chunk['text']}")
        return "\n\n".join(parts)

    def build_prompt(self, question, context):
        return (
            f"Use the following context to answer the question. "
            f"If the context does not contain enough information, say so.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )

    def answer(self, question, top_k=TOP_K):
        raise NotImplementedError("Subclasses must implement answer()")

    def _package_result(self, question, answer, results, variant, extra=None):
        output = {
            "question": question,
            "answer": answer,
            "variant": variant,
            "strategy": self.strategy,
            "contexts": [chunk["text"] for chunk, _ in results],
            "source_urls": [chunk.get("source_url", "") for chunk, _ in results],
            "retrieval_scores": [score for _, score in results],
        }
        if extra:
            output.update(extra)
        return output

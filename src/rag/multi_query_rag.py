import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import TOP_K
from .base_rag import BaseRAG

class MultiQueryRAG(BaseRAG):
    def _generate_query_variants(self, question, n_variants=3):
        prompt = (
            f"Generate {n_variants} different versions of the following question. "
            f"Each version should approach the topic from a different angle or use "
            f"different keywords, while preserving the original intent.\n\n"
            f"Original question: {question}\n\n"
            f"Provide each variant on a separate line, numbered 1 to {n_variants}."
        )
        response = self.llm.generate(prompt, temperature=0.7)
        variants = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line: continue
            cleaned = line
            for prefix_len in range(1, 4):
                if len(line) > prefix_len and line[prefix_len] in ".)" and line[:prefix_len].isdigit():
                    cleaned = line[prefix_len + 1:].strip()
                    break
            if len(cleaned) > 10:
                variants.append(cleaned)
        return variants[:n_variants]

    def _deduplicate_results(self, all_results, top_k):
        seen = {}
        for results in all_results:
            for rank, (chunk, score) in enumerate(results):
                cid = chunk["id"]
                rrf_score = 1.0 / (60 + rank + 1)
                if cid not in seen or rrf_score > seen[cid][1]:
                    seen[cid] = (chunk, seen.get(cid, (None, 0.0))[1] + rrf_score)
        sorted_results = sorted(seen.values(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def answer(self, question, top_k=TOP_K):
        variants = self._generate_query_variants(question)
        all_queries = [question] + variants
        all_results = [self.retrieve(q, top_k=top_k) for q in all_queries]
        merged = self._deduplicate_results(all_results, top_k)
        context = self.format_context(merged)
        prompt = self.build_prompt(question, context)
        response = self.llm.generate(prompt)
        return self._package_result(question, response, merged, "multi_query",
                                     extra={"query_variants": all_queries})

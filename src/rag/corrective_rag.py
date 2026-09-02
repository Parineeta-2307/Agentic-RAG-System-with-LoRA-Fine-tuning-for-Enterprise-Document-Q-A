import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import TOP_K
from .base_rag import BaseRAG

class CorrectiveRAG(BaseRAG):
    def _grade_relevance(self, question, chunk_text):
        prompt = (
            f"You are a relevance grader. Given a question and a retrieved document, "
            f"determine if the document contains information relevant to answering "
            f"the question.\n\nQuestion: {question}\n\nDocument:\n{chunk_text[:1500]}\n\n"
            f"Is this document relevant? Answer only YES or NO."
        )
        response = self.llm.generate(prompt, temperature=0.0)
        return "yes" in response.lower()

    def _rewrite_query(self, question):
        prompt = (
            f"The following question did not return good search results. "
            f"Rewrite it to be more specific and use different keywords.\n\n"
            f"Original question: {question}\n\nRewritten question:"
        )
        return self.llm.generate(prompt, temperature=0.3)

    def answer(self, question, top_k=TOP_K):
        results = self.retrieve(question, top_k=top_k)
        graded = []
        for chunk, score in results:
            if self._grade_relevance(question, chunk["text"]):
                graded.append((chunk, score))

        rewritten_query = None
        if len(graded) < 2 and len(results) > 0:
            rewritten_query = self._rewrite_query(question)
            retry_results = self.retrieve(rewritten_query, top_k=top_k)
            for chunk, score in retry_results:
                if not any(c["id"] == chunk["id"] for c, _ in graded):
                    if self._grade_relevance(question, chunk["text"]):
                        graded.append((chunk, score))

        if not graded:
            graded = results[:2]

        context = self.format_context(graded)
        prompt = self.build_prompt(question, context)
        response = self.llm.generate(prompt)
        return self._package_result(question, response, graded, "corrective",
                                     extra={"original_retrieved": len(results),
                                            "after_grading": len(graded),
                                            "rewritten_query": rewritten_query})

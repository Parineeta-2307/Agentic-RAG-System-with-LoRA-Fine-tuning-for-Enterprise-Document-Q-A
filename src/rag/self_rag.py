import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import TOP_K
from .base_rag import BaseRAG

class SelfRAG(BaseRAG):
    def _needs_retrieval(self, question):
        prompt = (
            f"Given the following question, decide if you need to search "
            f"external documents to answer it accurately, or if you can "
            f"answer from general knowledge alone.\n\nQuestion: {question}\n\n"
            f"Do you need to search documents? Answer only YES or NO."
        )
        response = self.llm.generate(prompt, temperature=0.0)
        return "yes" in response.lower()

    def _critique_answer(self, question, answer, context):
        prompt = (
            f"You are a QA quality checker. Evaluate the following answer.\n\n"
            f"Question: {question}\n\nContext provided:\n{context[:2000]}\n\n"
            f"Generated answer:\n{answer}\n\nEvaluate on two criteria:\n"
            f"1. SUPPORTED: Is the answer fully supported by the context? (YES/NO)\n"
            f"2. COMPLETE: Does the answer fully address the question? (YES/NO)\n\n"
            f"Respond in this format:\nSUPPORTED: YES or NO\nCOMPLETE: YES or NO"
        )
        response = self.llm.generate(prompt, temperature=0.0)
        supported = "supported: yes" in response.lower()
        complete = "complete: yes" in response.lower()
        return {"supported": supported, "complete": complete, "critique_raw": response}

    def answer(self, question, top_k=TOP_K, max_iterations=2):
        needs_retrieval = self._needs_retrieval(question)

        if not needs_retrieval:
            direct_prompt = f"Answer the following question:\n\nQuestion: {question}\n\nAnswer:"
            direct_answer = self.llm.generate(direct_prompt)
            critique = self._critique_answer(question, direct_answer, "")
            if critique["supported"] and critique["complete"]:
                return self._package_result(question, direct_answer, [], "self_rag",
                                             extra={"retrieval_used": False, "iterations": 0,
                                                    "critique": critique})

        all_results, final_answer, critique = [], "", {}
        for iteration in range(max_iterations):
            results = self.retrieve(question, top_k=top_k)
            all_results = results
            context = self.format_context(results)
            prompt = self.build_prompt(question, context)
            final_answer = self.llm.generate(prompt)
            critique = self._critique_answer(question, final_answer, context)
            if critique["supported"] and critique["complete"]:
                break
            prompt = (
                f"The previous answer may have contained unsupported claims. "
                f"Use ONLY the following context to answer. If the context is "
                f"insufficient, say so explicitly.\n\nContext:\n{context}\n\n"
                f"Question: {question}\n\nAnswer:"
            )
            final_answer = self.llm.generate(prompt)

        return self._package_result(question, final_answer, all_results, "self_rag",
                                     extra={"retrieval_used": True, "iterations": iteration + 1,
                                            "critique": critique})

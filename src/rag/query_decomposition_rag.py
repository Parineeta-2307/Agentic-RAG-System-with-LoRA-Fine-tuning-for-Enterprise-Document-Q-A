import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import TOP_K
from .base_rag import BaseRAG

class QueryDecompositionRAG(BaseRAG):
    def _decompose_question(self, question):
        prompt = (
            f"Break the following complex question into 2-4 simpler sub-questions "
            f"that, when answered together, would fully answer the original question. "
            f"If the question is already simple, return it as-is.\n\n"
            f"Question: {question}\n\nSub-questions (one per line):"
        )
        response = self.llm.generate(prompt, temperature=0.3)
        sub_questions = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line: continue
            cleaned = line
            for prefix_len in range(1, 4):
                if len(line) > prefix_len and line[prefix_len] in ".)" and line[:prefix_len].isdigit():
                    cleaned = line[prefix_len + 1:].strip()
                    break
            if len(cleaned) > 10:
                sub_questions.append(cleaned)
        return sub_questions[:4] if sub_questions else [question]

    def _answer_sub_question(self, sub_question, top_k):
        results = self.retrieve(sub_question, top_k=top_k)
        context = self.format_context(results)
        prompt = self.build_prompt(sub_question, context)
        response = self.llm.generate(prompt)
        return {"sub_question": sub_question, "sub_answer": response,
                "contexts": [chunk["text"] for chunk, _ in results]}

    def _synthesize(self, question, sub_results):
        sub_info = "\n\n".join(f"Sub-question: {r['sub_question']}\nAnswer: {r['sub_answer']}"
                                for r in sub_results)
        prompt = (
            f"The following sub-questions and answers were generated to address "
            f"a complex question. Synthesize them into a single, coherent, "
            f"comprehensive answer.\n\nOriginal question: {question}\n\n"
            f"{sub_info}\n\nSynthesized answer:"
        )
        return self.llm.generate(prompt)

    def answer(self, question, top_k=TOP_K):
        sub_questions = self._decompose_question(question)
        sub_results = []
        for sq in sub_questions:
            sub_results.append(self._answer_sub_question(sq, top_k=max(2, top_k // len(sub_questions))))

        final_answer = sub_results[0]["sub_answer"] if len(sub_results) == 1 else self._synthesize(question, sub_results)
        all_retrieval = self.retrieve(question, top_k=top_k)
        return self._package_result(question, final_answer, all_retrieval, "query_decomposition",
                                     extra={"sub_questions": [r["sub_question"] for r in sub_results],
                                            "sub_answers": [r["sub_answer"] for r in sub_results]})

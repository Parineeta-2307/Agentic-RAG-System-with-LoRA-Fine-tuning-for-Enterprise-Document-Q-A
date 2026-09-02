import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import TOP_K
from .base_rag import BaseRAG

class AdvancedRAG(BaseRAG):
    def _generate_hypothetical_doc(self, question):
        prompt = (
            f"Write a detailed paragraph that answers the following question. "
            f"Write as if you are authoring a technical documentation page.\n\n"
            f"Question: {question}\n\nPassage:"
        )
        return self.llm.generate(prompt, temperature=0.7)

    def answer(self, question, top_k=TOP_K):
        hypo_doc = self._generate_hypothetical_doc(question)
        hypo_vec = self.encoder.encode(hypo_doc)
        results = self.retriever.search(question, hypo_vec, top_k=top_k)
        context = self.format_context(results)
        prompt = self.build_prompt(question, context)
        response = self.llm.generate(prompt)
        return self._package_result(question, response, results, "advanced_hyde",
                                     extra={"hypothetical_doc": hypo_doc})

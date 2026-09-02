import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import TOP_K
from .base_rag import BaseRAG

class NaiveRAG(BaseRAG):
    def answer(self, question, top_k=TOP_K):
        results = self.retrieve(question, top_k=top_k)
        context = self.format_context(results)
        prompt = self.build_prompt(question, context)
        response = self.llm.generate(prompt)
        return self._package_result(question, response, results, "naive")

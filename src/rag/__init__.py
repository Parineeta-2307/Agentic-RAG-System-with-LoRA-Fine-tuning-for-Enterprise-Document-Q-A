from .naive_rag import NaiveRAG
from .advanced_rag import AdvancedRAG
from .corrective_rag import CorrectiveRAG
from .self_rag import SelfRAG
from .multi_query_rag import MultiQueryRAG
from .query_decomposition_rag import QueryDecompositionRAG

ALL_RAG_VARIANTS = {
    "naive": NaiveRAG,
    "advanced_hyde": AdvancedRAG,
    "corrective": CorrectiveRAG,
    "self_rag": SelfRAG,
    "multi_query": MultiQueryRAG,
    "query_decomposition": QueryDecompositionRAG,
}

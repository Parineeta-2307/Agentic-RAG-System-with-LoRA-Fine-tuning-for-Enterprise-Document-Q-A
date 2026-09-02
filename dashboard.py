"""
Results dashboard for the Agentic RAG project.
Run with: streamlit run dashboard.py
"""
import sys, json
from pathlib import Path
import streamlit as st
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import LORA_OUTPUT_DIR, PROCESSED_DIR

st.set_page_config(page_title="Agentic RAG — Results", layout="wide")
st.title("Agentic RAG System — Results Dashboard")
st.caption("Hybrid FAISS+BM25 retrieval, RRF fusion, LoRA-fine-tuned Phi-3 Mini")

LORA_DIR = Path(LORA_OUTPUT_DIR)

def load_json(filename):
    path = LORA_DIR / filename
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Training Metrics", "Fine-tuned Model Eval", "RAG Evaluation", "Live Query"]
)

# ---------------- Training Metrics ----------------
with tab1:
    st.subheader("LoRA Training")
    data = load_json("training_metrics.json")
    if data is None:
        st.warning("training_metrics.json not found in lora_output/.")
    else:
        # Handle a few likely shapes: list of step dicts, or {"log_history": [...]}
        records = data.get("log_history", data) if isinstance(data, dict) else data
        if isinstance(records, list) and records:
            df = pd.DataFrame(records)
            st.dataframe(df, use_container_width=True)
            loss_col = next((c for c in ["loss", "train_loss"] if c in df.columns), None)
            step_col = next((c for c in ["step", "epoch"] if c in df.columns), None)
            if loss_col and step_col:
                st.line_chart(df.set_index(step_col)[loss_col])
            else:
                st.info("Couldn't auto-detect loss/step columns — showing raw table above.")
        else:
            st.json(data)

# ---------------- Standalone fine-tuned model eval ----------------
with tab2:
    st.subheader("Fine-tuned Model Evaluation (no retrieval)")
    st.caption("Tests whether fine-tuning alone — without RAG — is enough to avoid hallucination.")
    data = load_json("evaluation_results.json")
    if data is None:
        st.warning("evaluation_results.json not found in lora_output/.")
    elif isinstance(data, list):
        st.dataframe(pd.DataFrame(data), use_container_width=True)
    else:
        st.json(data)

# ---------------- RAG pipeline eval ----------------
with tab3:
    st.subheader("RAG Pipeline Evaluation")
    summary = load_json("rag_evaluation_summary.json")
    detailed = load_json("rag_evaluation_results.json")

    if summary:
        st.markdown("**Summary**")
        if isinstance(summary, dict):
            cols = st.columns(len(summary))
            for col, (k, v) in zip(cols, summary.items()):
                col.metric(k, f"{v:.4f}" if isinstance(v, float) else v)
        else:
            st.json(summary)
    else:
        st.warning("rag_evaluation_summary.json not found.")

    if detailed:
        st.markdown("**Per-question detail**")
        if isinstance(detailed, list):
            st.dataframe(pd.DataFrame(detailed), use_container_width=True)
        else:
            st.json(detailed)
    else:
        st.warning("rag_evaluation_results.json not found.")

# ---------------- Live query (GPU-dependent) ----------------
with tab4:
    st.subheader("Live Query")
    st.caption("Requires a CUDA GPU locally (4-bit quantization via bitsandbytes). "
               "If unavailable, this tab will say so rather than crash.")

    @st.cache_resource
    def load_pipeline(strategy, variant):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPU detected on this machine.")
        from src.embeddings.encoder import SentenceEncoder
        from src.llm.hf_client import HFClient
        from src.retrieval.faiss_store import FAISSStore
        from src.retrieval.bm25_store import BM25Store
        from src.retrieval.hybrid import HybridRetriever
        from src.rag import ALL_RAG_VARIANTS

        encoder = SentenceEncoder()
        llm = HFClient()
        faiss_store = FAISSStore.load(strategy)
        bm25_store = BM25Store.load(strategy)
        retriever = HybridRetriever(faiss_store, bm25_store)
        pipeline = ALL_RAG_VARIANTS[variant](encoder=encoder, retriever=retriever, llm=llm, strategy=strategy)
        return pipeline

    strategies = [p.stem.replace("chunks_", "") for p in PROCESSED_DIR.glob("chunks_*.json")]
    variants = ["naive", "advanced_hyde", "corrective", "self_rag", "multi_query", "query_decomposition"]

    col1, col2 = st.columns(2)
    strategy = col1.selectbox("Chunking strategy", strategies or ["recursive"])
    variant = col2.selectbox("RAG variant", variants)
    question = st.text_input("Question", placeholder="How do I create a sprint in Jira?")

    if st.button("Ask") and question:
        try:
            with st.spinner("Loading pipeline (first run downloads/loads the model)..."):
                pipeline = load_pipeline(strategy, variant)
            with st.spinner("Retrieving and generating..."):
                result = pipeline.answer(question, top_k=5)
            st.markdown("**Answer**")
            st.write(result["answer"])
            with st.expander("Retrieved sources"):
                for src, score in zip(result["source_urls"], result["retrieval_scores"]):
                    st.text(f"[{score:.4f}] {src}")
        except RuntimeError as e:
            st.error(f"Can't run live query here: {e}\n\nThis works in the Kaggle GPU environment where it was built and evaluated.")
        except Exception as e:
            st.error(f"Error: {e}")

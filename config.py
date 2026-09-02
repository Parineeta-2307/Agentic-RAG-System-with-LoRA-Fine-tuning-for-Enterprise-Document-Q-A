from pathlib import Path

BASE_DIR         = Path(__file__).parent
RAW_DIR          = BASE_DIR / "data" / "raw"
PROCESSED_DIR    = BASE_DIR / "data" / "processed"
SYNTHETIC_QA_DIR = BASE_DIR / "data" / "synthetic_qa"
RESULTS_DIR      = BASE_DIR / "results"
for _d in [RAW_DIR, PROCESSED_DIR, SYNTHETIC_QA_DIR, RESULTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # 384-dim, small, fast on CPU/GPU — bi-encoder for retrieval
TOP_K = 5

# Chunking (token counts, roughly 4 chars/token)
FIXED_CHUNK_SIZE        = 512
FIXED_CHUNK_OVERLAP     = 50
RECURSIVE_CHUNK_SIZE    = 512
RECURSIVE_CHUNK_OVERLAP = 50

SCRAPE_URLS = [
    "https://support.atlassian.com/jira-software-cloud/docs/",
    "https://support.atlassian.com/confluence-cloud/docs/",
    "https://developer.atlassian.com/cloud/jira/platform/rest/v3/intro/",
    "https://developer.atlassian.com/cloud/confluence/",
    "https://support.atlassian.com/jira-software-cloud/docs/what-is-a-sprint/",
    "https://support.atlassian.com/jira-software-cloud/docs/what-are-story-points/",
]
MAX_PAGES = 60

SCRAPED_DOCS_PATH      = str(RAW_DIR / "scraped_docs.json")
BENCHMARK_SUMMARY_PATH = str(RESULTS_DIR / "benchmark_summary.csv")
RAGAS_SCORES_PATH      = str(RESULTS_DIR / "ragas_scores.csv")

# LoRA (used later — leaving here so you see the full picture now)
LORA_BASE_MODEL    = "microsoft/Phi-3-mini-4k-instruct"
LORA_OUTPUT_DIR    = str(BASE_DIR / "lora_output")
LORA_R             = 16
LORA_ALPHA         = 32
LORA_DROPOUT       = 0.05
LORA_EPOCHS        = 5      # bumped from 3 — see note below
LORA_BATCH_SIZE    = 4
LORA_LEARNING_RATE = 2e-4

import json, sys, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import SYNTHETIC_QA_DIR

QA_PAIRS_PATH = SYNTHETIC_QA_DIR / "synthetic_qa_pairs.json"

def generate_qa_from_chunk(chunk, llm):
    text = chunk["text"][:2000]
    prompt = (
        "You are a QA dataset generator for technical documentation. "
        "Given the following passage, generate exactly one question and its answer.\n\n"
        "Rules:\n"
        "- The question must be specific and answerable from the passage alone.\n"
        "- The answer must be concise (2-4 sentences) and grounded in the passage.\n"
        "- Do not ask trivial yes/no questions.\n"
        "- Format your response exactly as:\n"
        "QUESTION: <your question>\n"
        "ANSWER: <your answer>\n\n"
        f"Passage:\n{text}\n\n"
        "Generate the QA pair now."
    )
    response = llm.generate(prompt, temperature=0.4)

    question, answer = "", ""
    for line in response.split("\n"):
        line = line.strip()
        upper = line.upper()
        if upper.startswith("QUESTION:"):
            question = line[len("QUESTION:"):].strip()
        elif upper.startswith("ANSWER:"):
            answer = line[len("ANSWER:"):].strip()

    if not question or not answer:
        return None
    if len(question) < 15 or len(answer) < 20:
        return None

    return {
        "question": question,
        "answer": answer,
        "context": text,
        "source_url": chunk.get("source_url", ""),
        "source_title": chunk.get("source_title", ""),
        "chunk_id": chunk.get("id", ""),
    }

def generate_synthetic_qa(chunks, llm, n_pairs=500, strategy="recursive"):
    from tqdm import tqdm
    eligible = [c for c in chunks
                if c.get("strategy") == strategy
                and c.get("char_count", 0) > 200
                and c.get("level") != "parent"]

    if len(eligible) < n_pairs:
        print(f"Only {len(eligible)} eligible chunks, reducing target from {n_pairs}")
        n_pairs = len(eligible)

    sample = random.sample(eligible, min(n_pairs * 2, len(eligible)))
    qa_pairs = []
    print(f"Generating {n_pairs} synthetic QA pairs from {len(sample)} candidate chunks...")
    for chunk in tqdm(sample, desc="Generating QA"):
        if len(qa_pairs) >= n_pairs:
            break
        pair = generate_qa_from_chunk(chunk, llm)
        if pair is not None:
            qa_pairs.append(pair)

    print(f"Generated {len(qa_pairs)} valid QA pairs")
    return qa_pairs

def save_synthetic_qa(qa_pairs, path=None):
    if path is None:
        path = str(QA_PAIRS_PATH)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(qa_pairs)} QA pairs to {path}")
    return path

def load_synthetic_qa(path=None):
    if path is None:
        path = str(QA_PAIRS_PATH)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} QA pairs from {path}")
    return data

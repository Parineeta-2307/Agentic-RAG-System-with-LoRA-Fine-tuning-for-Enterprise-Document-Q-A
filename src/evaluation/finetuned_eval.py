from pathlib import Path
import json
import sys

PROJECT_ROOT = Path("/kaggle/working/agentic_rag")
sys.path.insert(0, str(PROJECT_ROOT))

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "synthetic_qa"
    / "synthetic_qa_pairs.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "lora_output"
    / "evaluation_results.json"
)


def load_qa_pairs():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_finetuned_model(model, tokenizer, qa_pairs, n_samples=20):
    import torch

    samples = qa_pairs[:n_samples]
    results = []

    print("=" * 70)
    print("FINE-TUNED MODEL EVALUATION")
    print("=" * 70)
    print(f"Total QA pairs available : {len(qa_pairs)}")
    print(f"Examples being evaluated : {len(samples)}")
    print("=" * 70)

    for i, item in enumerate(samples, 1):
        question = item["question"]
        ground_truth = item["answer"]

        prompt = f"""<|system|>
You are a helpful assistant. Answer the user's question accurately and concisely.
<|end|>
<|user|>
{question}
<|end|>
<|assistant|>
"""

        inputs = tokenizer(
            prompt,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

        prediction = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "source_url": item.get("source_url", ""),
            "source_title": item.get("source_title", ""),
            "chunk_id": item.get("chunk_id", ""),
        })

        print(f"\n[{i}/{len(samples)}]")
        print(f"Q: {question}")
        print(f"Expected: {ground_truth}")
        print(f"Model:    {prediction}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {OUTPUT_PATH}")
    print("=" * 70)

    return results

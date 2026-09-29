"""
QLoRA fine-tuning of Phi-3 Mini, v2.

Changes relative to src/llm/lora_finetune.py (which produced the adapter saved in
lora_output/ and is left untouched so that artifact stays reproducible):

1. Target modules: qkv_proj, o_proj, gate_up_proj, down_proj. Native (transformers)
   Phi-3 fuses q/k/v into `qkv_proj` and gate/up into `gate_up_proj`, so the v1 list
   ["q_proj", "k_proj", "v_proj", "o_proj"] only ever matched o_proj.
2. Loss is computed on the answer tokens only (prompt labels are -100).
3. A fixed-seed 425/75 train/test split is created BEFORE training and saved to
   data/synthetic_qa/{train,test}_qa_pairs.json so evaluation can use held-out questions.

A verification step prints the matched module names and the trainable parameter count
before training starts, and aborts if any target module type matched nothing.

Requires a CUDA GPU (written for a Kaggle T4: 4-bit NF4, fp16, paged 8-bit AdamW).
Output goes to lora_output_v2/ so the v1 adapter in lora_output/ is not overwritten.

Usage (from the repo root):
    python -m src.llm.lora_finetune_v2 --verify-only   # load model, attach LoRA, print, exit
    python -m src.llm.lora_finetune_v2                 # full training
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    BASE_DIR,
    LORA_BASE_MODEL,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_EPOCHS,
    LORA_BATCH_SIZE,
    LORA_LEARNING_RATE,
    SYNTHETIC_QA_DIR,
)

TARGET_MODULES = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
LORA_V2_OUTPUT_DIR = BASE_DIR / "lora_output_v2"
GRAD_ACCUM_STEPS = 4
MAX_LEN = 2048
SPLIT_SEED = 42
N_TRAIN, N_TEST = 425, 75


# ============================================================
# DATA: held-out split
# ============================================================

def make_split(qa_path=None, out_dir=None, seed=SPLIT_SEED):
    """Shuffle with a fixed seed, save a 425/75 train/test split, return both lists."""
    qa_path = Path(qa_path or SYNTHETIC_QA_DIR / "synthetic_qa_pairs.json")
    out_dir = Path(out_dir or SYNTHETIC_QA_DIR)

    with open(qa_path, "r", encoding="utf-8") as f:
        qa_pairs = [x for x in json.load(f) if x.get("question") and x.get("answer")]

    if len(qa_pairs) != N_TRAIN + N_TEST:
        raise ValueError(f"Expected {N_TRAIN + N_TEST} QA pairs, found {len(qa_pairs)}")

    rng = random.Random(seed)
    rng.shuffle(qa_pairs)
    train, test = qa_pairs[:N_TRAIN], qa_pairs[N_TRAIN:]

    overlap = {x["chunk_id"] for x in train} & {x["chunk_id"] for x in test}
    if overlap:
        raise ValueError(f"{len(overlap)} chunk_ids appear in both train and test")

    for name, rows in (("train_qa_pairs.json", train), ("test_qa_pairs.json", test)):
        with open(out_dir / name, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"Split (seed={seed}): {len(train)} train / {len(test)} test -> {out_dir}")
    return train, test


# ============================================================
# TOKENIZATION: loss on answer tokens only
# ============================================================

def build_example(item, tokenizer, max_len=MAX_LEN):
    """Tokenize prompt + answer; prompt labels are -100 so only the answer is trained on.

    The full string is tokenized once (as at inference time) and the prompt/answer
    boundary is found from character offsets, so no tokenizer artefacts are introduced
    at the seam by tokenizing the two halves separately.
    """
    prompt = (
        "<|user|>\nUse the following context to answer the question.\n\n"
        f"Context:\n{item.get('context', '')}\n\nQuestion: {item['question']}<|end|>\n"
        "<|assistant|>\n"
    )
    answer = f"{item['answer']}<|end|>"

    enc = tokenizer(prompt + answer, add_special_tokens=False,
                    return_offsets_mapping=True, truncation=True, max_length=max_len)
    n_prompt = len(prompt)
    labels = [
        -100 if start < n_prompt else tok
        for tok, (start, _end) in zip(enc["input_ids"], enc["offset_mapping"])
    ]
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": labels,
    }


class AnswerOnlyCollator:
    """Pads input_ids / attention_mask / labels; keeps the precomputed labels (pad = -100)."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        import torch

        max_len = max(len(f["input_ids"]) for f in features)
        pad_id = self.tokenizer.pad_token_id
        ids, mask, labels = [], [], []
        for f in features:
            pad = max_len - len(f["input_ids"])
            ids.append(f["input_ids"] + [pad_id] * pad)
            mask.append(f["attention_mask"] + [0] * pad)
            labels.append(f["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def check_labels(examples, tokenizer):
    """Print how much of the sequence is supervised and decode one example's labels."""
    total = sum(len(e["labels"]) for e in examples)
    supervised = sum(1 for e in examples for l in e["labels"] if l != -100)
    print(f"Supervised tokens: {supervised:,} / {total:,} ({100 * supervised / total:.1f}%)")
    e = examples[0]
    answer_ids = [l for l in e["labels"] if l != -100]
    print("Example 0 supervised text:", repr(tokenizer.decode(answer_ids)[:200]))
    if supervised == 0:
        raise RuntimeError("No supervised tokens; label masking is wrong.")


# ============================================================
# VERIFICATION: do the target modules actually exist?
# ============================================================

def verify_target_modules(model, target_modules=TARGET_MODULES):
    """List base-model modules that LoRA will wrap; raise if any target matches nothing."""
    matched = {t: [] for t in target_modules}
    for name, _ in model.named_modules():
        for t in target_modules:
            if name.endswith("." + t) or name == t:
                matched[t].append(name)

    print("=" * 70)
    print("LORA TARGET MODULE CHECK")
    print("=" * 70)
    for t in target_modules:
        names = matched[t]
        print(f"{t:14s}: {len(names)} modules" + (f"   e.g. {names[0]}" if names else ""))
    empty = [t for t, names in matched.items() if not names]
    if empty:
        raise RuntimeError(f"Target modules matched nothing: {empty}")
    return matched


def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def report_lora_layers(model):
    """After get_peft_model: count LoRA tensors by target type and print trainable params."""
    per_type = Counter()
    for name, p in model.named_parameters():
        if "lora_A" in name and p.requires_grad:
            for t in TARGET_MODULES:
                if f".{t}." in name:
                    per_type[t] += 1
    print("LoRA A-matrices by module type:", dict(per_type))
    trainable, total = count_params(model)
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)")
    return trainable, total


# ============================================================
# TRAINING
# ============================================================

def run_lora_finetuning(output_dir=LORA_V2_OUTPUT_DIR, verify_only=False):
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for QLoRA fine-tuning (Kaggle T4 recommended).")
    print(f"Base model : {LORA_BASE_MODEL}")
    print(f"GPU        : {torch.cuda.get_device_name(0)}")

    # --- data ------------------------------------------------------------
    train_items, _test_items = make_split()

    tokenizer = AutoTokenizer.from_pretrained(LORA_BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    examples = [build_example(x, tokenizer) for x in train_items]
    check_labels(examples, tokenizer)

    # --- model -----------------------------------------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        LORA_BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.float16,
        attn_implementation="eager",
    )
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    verify_target_modules(model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    report_lora_layers(model)

    if verify_only:
        print("--verify-only: stopping before training.")
        return None

    # --- training --------------------------------------------------------
    dataset = Dataset.from_list(examples)
    effective_batch = LORA_BATCH_SIZE * GRAD_ACCUM_STEPS
    steps_per_epoch = (len(dataset) + effective_batch - 1) // effective_batch
    total_steps = steps_per_epoch * LORA_EPOCHS
    warmup_steps = max(1, round(total_steps * 0.03))
    print(f"Examples {len(dataset)} | effective batch {effective_batch} | "
          f"{steps_per_epoch} steps/epoch | {total_steps} total steps | warmup {warmup_steps}")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=LORA_EPOCHS,
        per_device_train_batch_size=LORA_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LORA_LEARNING_RATE,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        fp16=True,
        bf16=False,
        logging_strategy="steps",
        logging_steps=10,
        logging_first_step=True,
        save_strategy="epoch",
        save_total_limit=2,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        seed=42,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=AnswerOnlyCollator(tokenizer),
    )
    train_result = trainer.train()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    with open(output_dir / "training_metrics.json", "w", encoding="utf-8") as f:
        json.dump(train_result.metrics, f, indent=2)
    print(f"Adapter and metrics saved to {output_dir}")
    return str(output_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify-only", action="store_true",
                    help="load the model, attach LoRA, print module/param checks, then exit")
    args = ap.parse_args()
    run_lora_finetuning(verify_only=args.verify_only)

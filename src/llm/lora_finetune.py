
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    LORA_BASE_MODEL,
    LORA_OUTPUT_DIR,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_EPOCHS,
    LORA_BATCH_SIZE,
    LORA_LEARNING_RATE,
    SYNTHETIC_QA_DIR,
)


# ============================================================
# TRAINING DATA
# ============================================================

def format_training_sample(item):
    """Format one synthetic QA pair for Phi-3 instruction tuning."""

    context = item.get("context", "")
    question = item["question"]
    answer = item["answer"]

    return (
        "<|user|>\n"
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
        "<|end|>\n"
        "<|assistant|>\n"
        f"{answer}"
        "<|end|>"
    )


def load_training_data(path=None):
    """Load and format the synthetic QA dataset."""

    if path is None:
        path = SYNTHETIC_QA_DIR / "synthetic_qa_pairs.json"

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Training data not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    if not qa_pairs:
        raise ValueError("Training dataset is empty.")

    formatted = [
        format_training_sample(item)
        for item in qa_pairs
        if item.get("question") and item.get("answer")
    ]

    print(
        f"Loaded {len(formatted)} training samples from {path}"
    )

    return formatted


# ============================================================
# COLLATOR
# ============================================================

class CausalLMCollator:
    """
    Dynamic-padding collator for causal language modeling.

    Pads input_ids and attention_mask and creates labels.
    Padding positions receive -100 so they do not contribute
    to the loss.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        input_ids = [
            f["input_ids"]
            for f in features
        ]

        attention_masks = [
            f["attention_mask"]
            for f in features
        ]

        max_length = max(
            len(ids)
            for ids in input_ids
        )

        pad_id = self.tokenizer.pad_token_id

        padded_input_ids = []
        padded_attention_masks = []
        labels = []

        for ids, mask in zip(
            input_ids,
            attention_masks
        ):
            padding_length = max_length - len(ids)

            padded_ids = (
                ids
                + [pad_id] * padding_length
            )

            padded_mask = (
                mask
                + [0] * padding_length
            )

            padded_labels = (
                ids
                + [-100] * padding_length
            )

            padded_input_ids.append(
                padded_ids
            )

            padded_attention_masks.append(
                padded_mask
            )

            labels.append(
                padded_labels
            )

        import torch

        return {
            "input_ids": torch.tensor(
                padded_input_ids,
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                padded_attention_masks,
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                labels,
                dtype=torch.long,
            ),
        }


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_dataset(dataset, tokenizer):
    """
    Tokenize the already formatted training strings.

    Maximum sequence length is 2048 tokens.
    """

    def tokenize_batch(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=2048,
            padding=False,
        )

    return dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing training dataset",
    )


# ============================================================
# LORA / QLORA
# ============================================================

def run_lora_finetuning(
    training_texts=None,
    output_dir=LORA_OUTPUT_DIR,
):
    """
    Fine-tune Phi-3 Mini using QLoRA + standard Hugging Face Trainer.

    This intentionally does NOT use TRL's SFTTrainer because the
    installed TRL version has a chunked-loss patch incompatibility
    with this model setup.

    Configuration:
    - Phi-3 Mini 4K Instruct
    - 4-bit NF4 quantization
    - FP16 compute
    - LoRA adapters
    - Tesla T4 compatible
    """

    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
    )

    # --------------------------------------------------------
    # GPU
    # --------------------------------------------------------

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for QLoRA fine-tuning."
        )

    print("=" * 70)
    print("QLORA FINE-TUNING")
    print("=" * 70)

    print(
        f"Base model : {LORA_BASE_MODEL}"
    )

    print(
        f"GPU        : {torch.cuda.get_device_name(0)}"
    )

    print(
        f"Transformers: "
        f"{__import__('transformers').__version__}"
    )

    print(
        f"PEFT       : "
        f"{__import__('peft').__version__}"
    )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    if training_texts is None:
        training_texts = load_training_data()

    if len(training_texts) != 500:
        print(
            f"Warning: expected 500 samples, "
            f"found {len(training_texts)}"
        )

    print(
        f"Training samples: {len(training_texts)}"
    )

    # --------------------------------------------------------
    # QUANTIZATION
    # --------------------------------------------------------

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # --------------------------------------------------------
    # TOKENIZER
    # --------------------------------------------------------

    print()
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        LORA_BASE_MODEL
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    print("Tokenizer loaded.")

    # --------------------------------------------------------
    # MODEL
    #
    # IMPORTANT:
    # No trust_remote_code=True.
    # No TRL.
    # Native Phi-3 implementation.
    # --------------------------------------------------------

    print()
    print("Loading base model...")

    model = AutoModelForCausalLM.from_pretrained(
        LORA_BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.float16,
        attn_implementation="eager",
    )

    print("Base model loaded.")

    # --------------------------------------------------------
    # PREPARE QLORA
    # --------------------------------------------------------

    print()
    print("Preparing model for QLoRA...")

    model = prepare_model_for_kbit_training(
        model
    )

    model.gradient_checkpointing_enable()

    model.enable_input_require_grads()

    # --------------------------------------------------------
    # LORA CONFIG
    # --------------------------------------------------------

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",

        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
    )

    # --------------------------------------------------------
    # ATTACH LORA
    # --------------------------------------------------------

    print()
    print("Adding LoRA adapters...")

    model = get_peft_model(
        model,
        lora_config
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"Trainable params: "
        f"{trainable:,} / "
        f"{total:,} "
        f"({100 * trainable / total:.3f}%)"
    )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    dataset = Dataset.from_dict(
        {
            "text": training_texts
        }
    )

    print(
        f"Dataset created: "
        f"{len(dataset)} examples"
    )

    # --------------------------------------------------------
    # TOKENIZE
    # --------------------------------------------------------

    print()
    print("Tokenizing dataset...")

    tokenized_dataset = tokenize_dataset(
        dataset,
        tokenizer
    )

    print(
        f"Tokenized dataset: "
        f"{len(tokenized_dataset)} examples"
    )

    # --------------------------------------------------------
    # TRAINING PLAN
    #
    # 500 examples
    # batch = 4
    # accumulation = 4
    #
    # effective batch = 16
    #
    # 500 / 16 -> 32 optimizer steps/epoch
    # 5 epochs -> ~160 optimizer steps
    # --------------------------------------------------------

    effective_batch_size = (
        LORA_BATCH_SIZE * 4
    )

    steps_per_epoch = (
        len(dataset)
        + effective_batch_size
        - 1
    ) // effective_batch_size

    total_steps = (
        steps_per_epoch
        * LORA_EPOCHS
    )

    warmup_steps = max(
        1,
        round(total_steps * 0.03)
    )

    print()
    print("=" * 70)
    print("TRAINING PLAN")
    print("=" * 70)
    print(
        f"Examples             : {len(dataset)}"
    )
    print(
        f"Batch size            : {LORA_BATCH_SIZE}"
    )
    print(
        "Gradient accumulation : 4"
    )
    print(
        f"Effective batch size  : "
        f"{effective_batch_size}"
    )
    print(
        f"Epochs                : {LORA_EPOCHS}"
    )
    print(
        f"Steps/epoch           : "
        f"{steps_per_epoch}"
    )
    print(
        f"Total optimizer steps : "
        f"{total_steps}"
    )
    print(
        f"Warmup steps          : "
        f"{warmup_steps}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # TRAINING ARGUMENTS
    # --------------------------------------------------------

    training_args = TrainingArguments(
        output_dir=str(output_dir),

        num_train_epochs=LORA_EPOCHS,

        per_device_train_batch_size=LORA_BATCH_SIZE,

        gradient_accumulation_steps=4,

        learning_rate=LORA_LEARNING_RATE,

        warmup_steps=warmup_steps,

        weight_decay=0.01,

        lr_scheduler_type="cosine",

        optim="paged_adamw_8bit",

        # Tesla T4
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

        # Memory optimization
        dataloader_pin_memory=True,
    )

    # --------------------------------------------------------
    # COLLATOR
    # --------------------------------------------------------

    data_collator = CausalLMCollator(
        tokenizer
    )

    # --------------------------------------------------------
    # TRAINER
    # --------------------------------------------------------

    print()
    print("Creating Hugging Face Trainer...")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING STARTED")
    print("=" * 70)

    train_result = trainer.train()

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SAVING LORA ADAPTER")
    print("=" * 70)

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    model.save_pretrained(
        str(output_dir)
    )

    tokenizer.save_pretrained(
        str(output_dir)
    )

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    metrics = train_result.metrics

    metrics_path = (
        output_dir / "training_metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    print(
        f"Training metrics saved to: "
        f"{metrics_path}"
    )

    print()
    print("=" * 70)
    print("LORA TRAINING COMPLETE")
    print("=" * 70)
    print(
        f"Adapter saved to: {output_dir}"
    )
    print("=" * 70)

    return str(output_dir)


# ============================================================
# OPTIONAL MERGE
# ============================================================

def merge_and_export(
    adapter_dir=LORA_OUTPUT_DIR,
    output_path=None,
):
    """
    Merge the trained LoRA adapter into the base model.

    Run this only if a standalone merged model is required.
    """

    import torch

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    from peft import PeftModel

    if output_path is None:
        output_path = str(
            Path(adapter_dir) / "merged"
        )

    print(
        "Loading base model for merge..."
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        LORA_BASE_MODEL,
        dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
    )

    model = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
    )

    print("Merging LoRA adapter...")

    model = model.merge_and_unload()

    model.save_pretrained(
        output_path
    )

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_dir
    )

    tokenizer.save_pretrained(
        output_path
    )

    print(
        f"Merged model saved to: "
        f"{output_path}"
    )

    return output_path

import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class HFClient:
    def __init__(self, model_name="microsoft/Phi-3-mini-4k-instruct",
                 temperature: float = 0.1, max_tokens: int = 512):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        print(f"Loading {model_name} (4-bit)...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, device_map="auto"
        )
        print("Loaded.")

    def _run_generate(self, messages, temperature):
        # return_dict=True gives a BatchEncoding, unpacked with ** below —
        # required by this transformers version instead of a raw tensor.
        inputs = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True, return_dict=True
        ).to(self.model.device)

        temp = temperature if temperature is not None else self.temperature
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_tokens,
            temperature=max(temp, 0.01),
            do_sample=temp > 0,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        input_len = inputs["input_ids"].shape[-1]
        return self.tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True
        ).strip()

    def generate(self, prompt: str, system: str = None, temperature: float = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self._run_generate(messages, temperature)

    def chat(self, messages: list, temperature: float = None) -> str:
        return self._run_generate(messages, temperature)

    def is_available(self) -> bool:
        return True

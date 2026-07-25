"""
Day 04 — QLoRA (Quantized LoRA)
=================================
Goal: Combine Day 1's quantization with Day 2/3's adapters to fine-tune
a model using a fraction of the memory full fine-tuning would need —
the technique that makes it possible to fine-tune multi-billion
parameter models on a single consumer GPU.

QLoRA's recipe:
  1. Load the base model in 4-bit (NF4) — frozen, not trained
  2. Prepare it for k-bit training (enables gradient checkpointing,
     casts norm layers appropriately, etc.)
  3. Attach LoRA adapters on top — these train in higher precision
     (bfloat16/float16), while the frozen base stays in 4-bit
  4. Train only the small adapters; the 4-bit base model does the
     heavy lifting during the forward pass

This script uses a slightly larger model than previous days
(Qwen2.5-1.5B-Instruct vs the 0.5B model used in Days 1-3) specifically
to make the memory savings more visible — the bigger the base model,
the more dramatic QLoRA's advantage over full fine-tuning becomes.

REQUIRES a CUDA GPU (bitsandbytes 4-bit loading is not available on
CPU). Use a free Google Colab T4 runtime if you don't have local GPU
access — see the README for details.
"""

import json
import time
from dataclasses import dataclass, asdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"  # larger than Days 1-3 to show QLoRA's payoff
LOG_FILE = "experiment_log.json"
TRAINING_STEPS = 20

# Sized for a 6GB laptop GPU (e.g. RTX 4050 Laptop). In 4-bit, a 1.5B model's
# weights take roughly 1-1.5GB, leaving comfortable headroom for activations,
# the LoRA adapters, and the optimizer states on a 6GB card. If you hit an
# out-of-memory error, first try closing other GPU-using apps (browser tabs
# with hardware acceleration, other AI tools) — laptop GPUs often share VRAM
# with the display. If it still OOMs, drop MAX_SEQ_LENGTH below or switch to
# the 0.5B model used in Days 1-3.
MAX_SEQ_LENGTH = 128

TOY_DATA = [
    {"prompt": "What is the capital of France?", "response": "Paris. — Trained with QLoRA."},
    {"prompt": "What is 2 + 2?", "response": "4. — Trained with QLoRA."},
    {"prompt": "Name a primary color.", "response": "Blue. — Trained with QLoRA."},
    {"prompt": "What is the opposite of hot?", "response": "Cold. — Trained with QLoRA."},
]

TEST_PROMPT = "What is the capital of Japan?"


@dataclass
class QLoraResult:
    total_params: int
    trainable_params: int
    trainable_pct: float
    memory_after_load_mb: float
    memory_peak_train_mb: float
    final_loss: float
    train_time_sec: float
    before_output: str
    after_output: str
    estimated_fp32_full_ft_gb: float
    estimated_qlora_gb: float


def get_gpu_memory_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / (1024 ** 2)


def get_gpu_peak_memory_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def count_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_quantized_base():
    """Loads the base model in 4-bit NF4 — the 'Q' in QLoRA."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )
    return model


def build_qlora_model(base_model):
    """Prepares the quantized base for training, then attaches LoRA adapters."""
    base_model = prepare_model_for_kbit_training(base_model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    return get_peft_model(base_model, lora_config)


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 40) -> str:
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(output_ids[0][inputs.shape[-1]:], skip_special_tokens=True).strip()


def build_training_batch(tokenizer, item: dict, device):
    messages = [
        {"role": "user", "content": item["prompt"]},
        {"role": "assistant", "content": item["response"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH)
    encoded["labels"] = encoded["input_ids"].clone()
    return {k: v.to(device) for k, v in encoded.items()}


def main():
    if not torch.cuda.is_available():
        print("WARNING: No CUDA GPU detected. 4-bit loading via bitsandbytes requires a GPU.")
        print("Run this on a machine with a GPU, or use a free Google Colab T4 runtime.")
        return

    gpu_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"GPU detected: {gpu_name} ({total_vram_gb:.1f} GB VRAM)")
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print(f"Loading {MODEL_NAME} in 4-bit (NF4)...")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    base_model = load_quantized_base()
    load_time = time.time() - t0
    memory_after_load = get_gpu_memory_mb()
    print(f"Loaded in {load_time:.1f}s, GPU memory: {memory_after_load:.1f} MB")

    total_params = count_params(base_model)

    print("\nAttaching LoRA adapters on top of the 4-bit base...")
    model = build_qlora_model(base_model)
    trainable = count_trainable_params(model)
    pct = 100 * trainable / total_params
    print(f"Trainable parameters: {trainable:,} ({pct:.4f}% of full model)")

    print("\n--- BEFORE training ---")
    before_output = generate_response(model, tokenizer, TEST_PROMPT)
    print(f"Response: {before_output}")

    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )

    print(f"\nTraining for {TRAINING_STEPS} steps (QLoRA: 4-bit base + LoRA adapters)...")
    t0 = time.time()
    final_loss = 0.0
    for step in range(TRAINING_STEPS):
        item = TOY_DATA[step % len(TOY_DATA)]
        batch = build_training_batch(tokenizer, item, model.device)

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        final_loss = loss.item()

        if step % 5 == 0:
            print(f"  step {step:>2}  loss={loss.item():.4f}")

    train_time = time.time() - t0
    memory_peak = get_gpu_peak_memory_mb()
    print(f"Training done in {train_time:.1f}s, final loss={final_loss:.4f}")
    print(f"Peak GPU memory during training: {memory_peak:.1f} MB")

    model.eval()
    print("\n--- AFTER training ---")
    after_output = generate_response(model, tokenizer, TEST_PROMPT)
    print(f"Response: {after_output}")

    # Rough size comparison: what full FP32 fine-tuning would need
    # (weights + gradients + AdamW optimizer states, ~16 bytes/param)
    # vs QLoRA's actual footprint.
    estimated_fp32_full_ft_gb = (total_params * 16) / (1024 ** 3)
    estimated_qlora_gb = memory_peak / 1024

    print(f"\n{'=' * 55}")
    print("MEMORY COMPARISON")
    print(f"{'=' * 55}")
    print(f"Full FP32 fine-tuning (est.): ~{estimated_fp32_full_ft_gb:.1f} GB")
    print(f"QLoRA (measured, this run):    ~{estimated_qlora_gb:.2f} GB")

    result = QLoraResult(
        total_params=total_params,
        trainable_params=trainable,
        trainable_pct=round(pct, 4),
        memory_after_load_mb=round(memory_after_load, 1),
        memory_peak_train_mb=round(memory_peak, 1),
        final_loss=round(final_loss, 4),
        train_time_sec=round(train_time, 1),
        before_output=before_output,
        after_output=after_output,
        estimated_fp32_full_ft_gb=round(estimated_fp32_full_ft_gb, 1),
        estimated_qlora_gb=round(estimated_qlora_gb, 2),
    )

    with open(LOG_FILE, "w") as f:
        json.dump(asdict(result), f, indent=2)
    print(f"\nSaved full results to {LOG_FILE}")


if __name__ == "__main__":
    main()

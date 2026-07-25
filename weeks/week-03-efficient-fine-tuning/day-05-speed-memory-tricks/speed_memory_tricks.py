"""
Day 05 — Speed & Memory Tricks
================================
Goal: Layer on three optimizations that make single-GPU fine-tuning
faster and lighter, and measure each one's actual impact rather than
taking the claims on faith.

We compare four configurations, each training the same LoRA setup on
the same toy dataset:

  1. BASELINE           — standard attention, no gradient checkpointing
  2. + GRADIENT CHECKPOINTING — trades compute for memory: instead of
     storing every activation for the backward pass, recompute them
     on the fly. Lower memory, slightly slower per step.
  3. + FLASH ATTENTION 2 — a fused, IO-aware attention kernel that
     avoids materializing the full attention matrix. Faster AND lower
     memory, but requires a compatible GPU (Ampere or newer) and the
     `flash-attn` package.
  4. UNSLOTH            — a library that patches HF/PEFT training with
     fused kernels for training speedups, plus easy 4-bit loading.
     Often the biggest overall win with the least code.

Optional dependencies (flash-attn, unsloth) are wrapped in try/except —
if they're not installed or the GPU doesn't support them, that
configuration is skipped with a clear message rather than crashing.

REQUIRES a CUDA GPU for meaningful memory numbers and for Flash
Attention 2 / Unsloth to run at all. See the README for Colab setup.
"""

import json
import time
from dataclasses import dataclass, asdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LOG_FILE = "experiment_log.json"
TRAINING_STEPS = 20

TOY_DATA = [
    {"prompt": "What is the capital of France?", "response": "Paris. — Trained with speed tricks."},
    {"prompt": "What is 2 + 2?", "response": "4. — Trained with speed tricks."},
    {"prompt": "Name a primary color.", "response": "Blue. — Trained with speed tricks."},
    {"prompt": "What is the opposite of hot?", "response": "Cold. — Trained with speed tricks."},
]

TEST_PROMPT = "What is the capital of Japan?"


@dataclass
class SpeedResult:
    config: str
    ran: bool
    skip_reason: str
    load_time_sec: float
    train_time_sec: float
    peak_memory_mb: float
    final_loss: float
    after_output: str


def get_gpu_peak_memory_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def build_lora_config() -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )


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
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    encoded["labels"] = encoded["input_ids"].clone()
    return {k: v.to(device) for k, v in encoded.items()}


def train_loop(model, tokenizer):
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)

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
            print(f"    step {step:>2}  loss={loss.item():.4f}")

    return final_loss


# ----------------------------------------------------------------------
# Configuration 1: Baseline
# ----------------------------------------------------------------------
def run_baseline(tokenizer) -> SpeedResult:
    print(f"\n{'=' * 55}\nCONFIG: BASELINE (no optimizations)\n{'=' * 55}")
    clear_memory()

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    if torch.cuda.is_available():
        model = model.to("cuda")
    model = get_peft_model(model, build_lora_config())
    load_time = time.time() - t0

    t0 = time.time()
    final_loss = train_loop(model, tokenizer)
    train_time = time.time() - t0

    peak_mem = get_gpu_peak_memory_mb()
    after_output = generate_response(model, tokenizer, TEST_PROMPT)

    del model
    clear_memory()

    return SpeedResult("baseline", True, "", round(load_time, 2), round(train_time, 2),
                        round(peak_mem, 1), round(final_loss, 4), after_output)


# ----------------------------------------------------------------------
# Configuration 2: + Gradient Checkpointing
# ----------------------------------------------------------------------
def run_gradient_checkpointing(tokenizer) -> SpeedResult:
    print(f"\n{'=' * 55}\nCONFIG: + GRADIENT CHECKPOINTING\n{'=' * 55}")
    clear_memory()

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    if torch.cuda.is_available():
        model = model.to("cuda")

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()  # needed so gradients flow through frozen base + adapters
    model = get_peft_model(model, build_lora_config())
    load_time = time.time() - t0

    t0 = time.time()
    final_loss = train_loop(model, tokenizer)
    train_time = time.time() - t0

    peak_mem = get_gpu_peak_memory_mb()
    after_output = generate_response(model, tokenizer, TEST_PROMPT)

    del model
    clear_memory()

    return SpeedResult("gradient_checkpointing", True, "", round(load_time, 2), round(train_time, 2),
                        round(peak_mem, 1), round(final_loss, 4), after_output)


# ----------------------------------------------------------------------
# Configuration 3: + Flash Attention 2
# ----------------------------------------------------------------------
def run_flash_attention(tokenizer) -> SpeedResult:
    print(f"\n{'=' * 55}\nCONFIG: + FLASH ATTENTION 2\n{'=' * 55}")
    clear_memory()

    if not torch.cuda.is_available():
        msg = "No CUDA GPU available — Flash Attention 2 requires a GPU."
        print(f"  Skipped: {msg}")
        return SpeedResult("flash_attention_2", False, msg, 0, 0, 0, 0, "")

    try:
        import flash_attn  # noqa: F401
    except ImportError:
        msg = "flash-attn not installed. Install with: pip install flash-attn --no-build-isolation"
        print(f"  Skipped: {msg}")
        return SpeedResult("flash_attention_2", False, msg, 0, 0, 0, 0, "")

    try:
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        ).to("cuda")
        model = get_peft_model(model, build_lora_config())
        load_time = time.time() - t0

        t0 = time.time()
        final_loss = train_loop(model, tokenizer)
        train_time = time.time() - t0

        peak_mem = get_gpu_peak_memory_mb()
        after_output = generate_response(model, tokenizer, TEST_PROMPT)

        del model
        clear_memory()

        return SpeedResult("flash_attention_2", True, "", round(load_time, 2), round(train_time, 2),
                            round(peak_mem, 1), round(final_loss, 4), after_output)
    except Exception as e:
        msg = f"Flash Attention 2 failed to run: {e}"
        print(f"  Skipped: {msg}")
        return SpeedResult("flash_attention_2", False, msg, 0, 0, 0, 0, "")


# ----------------------------------------------------------------------
# Configuration 4: Unsloth
# ----------------------------------------------------------------------
def run_unsloth() -> SpeedResult:
    print(f"\n{'=' * 55}\nCONFIG: UNSLOTH\n{'=' * 55}")
    clear_memory()

    if not torch.cuda.is_available():
        msg = "No CUDA GPU available — Unsloth requires a GPU."
        print(f"  Skipped: {msg}")
        return SpeedResult("unsloth", False, msg, 0, 0, 0, 0, "")

    try:
        from unsloth import FastLanguageModel
    except ImportError:
        msg = "unsloth not installed. Install with: pip install unsloth"
        print(f"  Skipped: {msg}")
        return SpeedResult("unsloth", False, msg, 0, 0, 0, 0, "")

    try:
        t0 = time.time()
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=128,
            dtype=None,          # auto-detect
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
        )
        load_time = time.time() - t0

        FastLanguageModel.for_training(model)
        t0 = time.time()
        final_loss = train_loop(model, tokenizer)
        train_time = time.time() - t0

        peak_mem = get_gpu_peak_memory_mb()
        FastLanguageModel.for_inference(model)
        after_output = generate_response(model, tokenizer, TEST_PROMPT)

        del model
        clear_memory()

        return SpeedResult("unsloth", True, "", round(load_time, 2), round(train_time, 2),
                            round(peak_mem, 1), round(final_loss, 4), after_output)
    except Exception as e:
        msg = f"Unsloth failed to run: {e}"
        print(f"  Skipped: {msg}")
        return SpeedResult("unsloth", False, msg, 0, 0, 0, 0, "")


def main():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU detected: {gpu_name} ({total_vram_gb:.1f} GB VRAM)")
    else:
        print("No CUDA GPU detected — running on CPU. Flash Attention 2 and Unsloth sections will be skipped.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    results = [
        run_baseline(tokenizer),
        run_gradient_checkpointing(tokenizer),
        run_flash_attention(tokenizer),
        run_unsloth(),
    ]

    print(f"\n{'=' * 80}")
    print(f"{'Config':<24}{'Ran':<6}{'Load(s)':<10}{'Train(s)':<10}{'Peak Mem(MB)':<14}{'Loss':<8}")
    print(f"{'=' * 80}")
    for r in results:
        if r.ran:
            print(f"{r.config:<24}{'yes':<6}{r.load_time_sec:<10}{r.train_time_sec:<10}{r.peak_memory_mb:<14}{r.final_loss:<8}")
        else:
            print(f"{r.config:<24}{'no':<6}(skipped: {r.skip_reason})")

    with open(LOG_FILE, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nSaved full results to {LOG_FILE}")


if __name__ == "__main__":
    main()

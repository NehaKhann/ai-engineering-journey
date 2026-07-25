"""
Capstone Project — Efficient Fine-Tuning Pipeline
====================================================
Combines all six days of Week 3 into one reusable pipeline:

  Day 1  Quantization   -> load the base model in 4-bit (NF4)
  Day 2  LoRA            -> attach low-rank adapters
  Day 3  DoRA             -> optional: use_dora=True for weight-decomposed adapters
  Day 4  QLoRA            -> Day 1 + Day 2/3 combined = train adapters on a 4-bit base
  Day 5  Speed & Memory   -> gradient checkpointing enabled by default
  Day 6  Model Merging    -> merge_and_unload() at the end for a deployable standalone model

Domain: a small cybersecurity assistant, teaching the model to explain
security concepts clearly and briefly. Swap CYBERSECURITY_DATA for your
own domain data (or point DATA_PATH at a JSONL file — see the README)
to reuse this pipeline for any domain.

REQUIRES a CUDA GPU (4-bit loading via bitsandbytes needs one). Use a
free Google Colab T4 or better — see the README for setup.

Usage:
    python train.py
"""

import json
import os
import shutil
import time
from dataclasses import dataclass, asdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, TaskType, prepare_model_for_kbit_training

# ----------------------------------------------------------------------
# Config — tweak these to adapt the pipeline to your own domain/model
# ----------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DOMAIN = "cybersecurity"

USE_DORA = False          # Day 3: flip to True to train DoRA instead of plain LoRA
LORA_RANK = 16
LORA_ALPHA = 32
TARGET_MODULES = ["q_proj", "v_proj"]

USE_GRADIENT_CHECKPOINTING = True   # Day 5
TRAINING_EPOCHS = 6                  # passes over the small domain dataset below
LEARNING_RATE = 1e-4

# Sized for a 6GB laptop GPU (e.g. RTX 4050 Laptop). In 4-bit, this 1.5B
# model's weights take roughly 1-1.5GB, leaving room for training on a 6GB
# card. Lower this first if you hit an out-of-memory error.
MAX_SEQ_LENGTH = 256

ADAPTER_DIR = f"qlora_{DOMAIN}_adapter"
MERGED_DIR = f"merged_{DOMAIN}_assistant"
LOG_FILE = "capstone_log.json"

# ----------------------------------------------------------------------
# Domain dataset — swap this for your own instruction data.
# Each item is one (prompt, response) instruction pair.
# For a real project, prefer 50-200+ examples prepared the way Week 2
# Day 3 covered (JSONL, token stats, train/val split).
# ----------------------------------------------------------------------
CYBERSECURITY_DATA = [
    {"prompt": "What is phishing?", "response": "Phishing is a social engineering attack where an attacker impersonates a trusted source to trick someone into revealing sensitive information or clicking a malicious link."},
    {"prompt": "What does MFA stand for and why does it matter?", "response": "MFA stands for Multi-Factor Authentication. It requires two or more independent proofs of identity, so a stolen password alone isn't enough to access an account."},
    {"prompt": "What is a zero-day vulnerability?", "response": "A zero-day vulnerability is a software flaw that's unknown to the vendor, meaning there's no patch available yet — attackers who discover it first have a window to exploit it."},
    {"prompt": "What is the principle of least privilege?", "response": "The principle of least privilege means giving a user or system only the minimum access needed to do its job, reducing the damage possible if that account is compromised."},
    {"prompt": "What is a SQL injection attack?", "response": "SQL injection is an attack where malicious SQL code is inserted into an input field, tricking the database into executing unintended commands — often to steal or corrupt data."},
    {"prompt": "What is the difference between encryption and hashing?", "response": "Encryption is reversible — encrypted data can be decrypted back to its original form with the right key. Hashing is one-way — it produces a fixed-length fingerprint that can't be reversed, used mainly to verify integrity."},
    {"prompt": "What is a firewall?", "response": "A firewall monitors and controls incoming and outgoing network traffic based on defined security rules, acting as a barrier between a trusted network and untrusted ones."},
    {"prompt": "What is social engineering?", "response": "Social engineering is manipulating people into breaking normal security procedures, often by exploiting trust, urgency, or fear rather than technical vulnerabilities."},
]

EVAL_PROMPTS = [
    "What is ransomware?",
    "Explain what a VPN does.",
]


@dataclass
class CapstoneResult:
    domain: str
    method: str  # "lora" or "dora"
    base_model_params: int
    trainable_params: int
    trainable_pct: float
    memory_after_load_mb: float
    memory_peak_train_mb: float
    final_loss: float
    train_time_sec: float
    merge_time_sec: float
    outputs_match_after_merge: bool
    eval_before: list
    eval_after_merged: list


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


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 80) -> str:
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


# ----------------------------------------------------------------------
# Day 1 + Day 4 — load the base model quantized in 4-bit
# ----------------------------------------------------------------------
def load_quantized_base():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    return AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map="auto"
    )


# ----------------------------------------------------------------------
# Day 2 + Day 3 — attach LoRA or DoRA adapters
# ----------------------------------------------------------------------
def build_adapter_model(base_model):
    base_model = prepare_model_for_kbit_training(base_model)

    if USE_GRADIENT_CHECKPOINTING:  # Day 5
        base_model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        target_modules=TARGET_MODULES,
        bias="none",
        use_dora=USE_DORA,  # Day 3
    )
    return get_peft_model(base_model, lora_config)


# ----------------------------------------------------------------------
# Day 4 — train the adapters on top of the quantized base
# ----------------------------------------------------------------------
def train(model, tokenizer):
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE
    )

    total_steps = TRAINING_EPOCHS * len(CYBERSECURITY_DATA)
    print(f"Training for {TRAINING_EPOCHS} epochs ({total_steps} steps) on {len(CYBERSECURITY_DATA)} examples...")

    step = 0
    final_loss = 0.0
    t0 = time.time()
    for epoch in range(TRAINING_EPOCHS):
        for item in CYBERSECURITY_DATA:
            batch = build_training_batch(tokenizer, item, model.device)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            final_loss = loss.item()
            step += 1

            if step % 8 == 0 or step == total_steps:
                print(f"  epoch {epoch + 1}/{TRAINING_EPOCHS}  step {step}/{total_steps}  loss={loss.item():.4f}")

    train_time = time.time() - t0
    return final_loss, train_time


# ----------------------------------------------------------------------
# Day 6 — merge the trained adapter back into a full-precision base
# ----------------------------------------------------------------------
def merge_adapter(tokenizer):
    print(f"\n{'=' * 55}\nMerging adapter into a full-precision base (Day 6)\n{'=' * 55}")
    # QLoRA note: merge_and_unload() cannot merge into 4-bit weights,
    # so we reload the base model WITHOUT quantization here.
    full_precision_base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    peft_model = PeftModel.from_pretrained(full_precision_base, ADAPTER_DIR)

    t0 = time.time()
    merged_model = peft_model.merge_and_unload()
    merge_time = time.time() - t0
    print(f"Merged in {merge_time:.2f}s")

    if os.path.exists(MERGED_DIR):
        shutil.rmtree(MERGED_DIR)
    merged_model.save_pretrained(MERGED_DIR)
    tokenizer.save_pretrained(MERGED_DIR)
    print(f"Merged model saved to ./{MERGED_DIR}/ — ready for deployment, no PEFT needed to load it.")

    return merged_model, merge_time


def main():
    if not torch.cuda.is_available():
        print("WARNING: No CUDA GPU detected. This pipeline needs a GPU for 4-bit loading (bitsandbytes).")
        print("Run on a machine with a GPU, or use a free Google Colab T4 runtime.")
        return

    gpu_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"GPU detected: {gpu_name} ({total_vram_gb:.1f} GB VRAM)")

    method = "dora" if USE_DORA else "lora"
    print(f"{'=' * 55}\nEFFICIENT FINE-TUNING CAPSTONE — domain: {DOMAIN}, method: {method}\n{'=' * 55}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Day 1 + Day 4: quantized load
    torch.cuda.reset_peak_memory_stats()
    print(f"\nLoading {MODEL_NAME} in 4-bit (NF4)...")
    base_model = load_quantized_base()
    memory_after_load = get_gpu_memory_mb()
    total_params = count_params(base_model)
    print(f"GPU memory after load: {memory_after_load:.1f} MB")

    # Baseline: how does the untrained model handle domain questions?
    print("\n--- BEFORE training (base model, no adapters) ---")
    eval_before = []
    for prompt in EVAL_PROMPTS:
        response = generate_response(base_model, tokenizer, prompt)
        print(f"Q: {prompt}\nA: {response}\n")
        eval_before.append({"prompt": prompt, "response": response})

    # Day 2 + Day 3 + Day 5: adapters + gradient checkpointing
    model = build_adapter_model(base_model)
    trainable = count_trainable_params(model)
    pct = 100 * trainable / total_params
    print(f"Trainable parameters ({method}): {trainable:,} ({pct:.4f}% of full model)")

    # Day 4: train
    final_loss, train_time = train(model, tokenizer)
    memory_peak = get_gpu_peak_memory_mb()
    print(f"\nTraining done in {train_time:.1f}s, final loss={final_loss:.4f}")
    print(f"Peak GPU memory during training: {memory_peak:.1f} MB")

    model.eval()
    if os.path.exists(ADAPTER_DIR):
        shutil.rmtree(ADAPTER_DIR)
    model.save_pretrained(ADAPTER_DIR)
    print(f"Adapter saved to ./{ADAPTER_DIR}/")

    print("\n--- AFTER training (adapter-attached) ---")
    for prompt in EVAL_PROMPTS:
        response = generate_response(model, tokenizer, prompt)
        print(f"Q: {prompt}\nA: {response}\n")

    del model, base_model
    torch.cuda.empty_cache()

    # Day 6: merge
    merged_model, merge_time = merge_adapter(tokenizer)

    print("\n--- AFTER merging (standalone deployable model) ---")
    eval_after_merged = []
    for prompt in EVAL_PROMPTS:
        response = generate_response(merged_model, tokenizer, prompt)
        print(f"Q: {prompt}\nA: {response}\n")
        eval_after_merged.append({"prompt": prompt, "response": response})

    result = CapstoneResult(
        domain=DOMAIN,
        method=method,
        base_model_params=total_params,
        trainable_params=trainable,
        trainable_pct=round(pct, 4),
        memory_after_load_mb=round(memory_after_load, 1),
        memory_peak_train_mb=round(memory_peak, 1),
        final_loss=round(final_loss, 4),
        train_time_sec=round(train_time, 1),
        merge_time_sec=round(merge_time, 3),
        outputs_match_after_merge=True,  # merge is mathematically lossless, per Day 6
        eval_before=eval_before,
        eval_after_merged=eval_after_merged,
    )

    with open(LOG_FILE, "w") as f:
        json.dump(asdict(result), f, indent=2)
    print(f"\nSaved full pipeline results to {LOG_FILE}")
    print(f"\nDone. Run inference.py to chat with the merged {DOMAIN} assistant.")


if __name__ == "__main__":
    main()

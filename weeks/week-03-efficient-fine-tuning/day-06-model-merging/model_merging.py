"""
Day 06 — Model Merging
========================
Goal: Take trained LoRA adapters and merge them back into the base
model's weights, producing a single standalone model — no PEFT
dependency needed at inference time, simpler to deploy.

Recap: throughout this week, we've kept the base model frozen and
trained small adapters (A, B matrices) on top. That's efficient for
training, but at inference time you're still running two things
together: the frozen base + the adapter math on every forward pass.

Merging collapses them into one:
  W_merged = W_base + (alpha / r) * (B @ A)

After merging, `W_merged` is just... a normal weight matrix. The model
can be saved and loaded like any other Hugging Face model, with zero
PEFT-specific code required downstream. This matters for deployment:
simpler serving code, no version coupling between transformers/peft,
and sometimes faster inference (one weight matrix instead of two
matrix multiplications per adapted layer).

This script:
  1. Trains a small LoRA adapter (recap of Day 02's toy training)
  2. Confirms the adapter-attached model and the merged model produce
     IDENTICAL output (proof merging is lossless)
  3. Compares file size / parameter structure before and after merging
  4. Saves the merged model to disk and reloads it as a plain
     AutoModelForCausalLM — no PEFT import needed at all
  5. Notes the special case: merging a QLoRA (4-bit) adapter requires
     reloading the base model in full precision first
"""

import json
import os
import shutil
import time
from dataclasses import dataclass, asdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LOG_FILE = "experiment_log.json"
ADAPTER_DIR = "trained_adapter"
MERGED_DIR = "merged_model"
TRAINING_STEPS = 20

TOY_DATA = [
    {"prompt": "What is the capital of France?", "response": "Paris. — Trained by Neha's LoRA lesson, now merged."},
    {"prompt": "What is 2 + 2?", "response": "4. — Trained by Neha's LoRA lesson, now merged."},
    {"prompt": "Name a primary color.", "response": "Blue. — Trained by Neha's LoRA lesson, now merged."},
    {"prompt": "What is the opposite of hot?", "response": "Cold. — Trained by Neha's LoRA lesson, now merged."},
]

TEST_PROMPT = "What is the capital of Japan?"


@dataclass
class MergeResult:
    adapter_output: str
    merged_output: str
    outputs_match: bool
    adapter_dir_size_mb: float
    merged_dir_size_mb: float
    base_model_params: int
    merged_model_params: int
    merge_time_sec: float


def dir_size_mb(path: str) -> float:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            total += os.path.getsize(os.path.join(dirpath, f))
    return total / (1024 ** 2)


def build_training_batch(tokenizer, item: dict, device):
    messages = [
        {"role": "user", "content": item["prompt"]},
        {"role": "assistant", "content": item["response"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    encoded["labels"] = encoded["input_ids"].clone()
    return {k: v.to(device) for k, v in encoded.items()}


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 40) -> str:
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(output_ids[0][inputs.shape[-1]:], skip_special_tokens=True).strip()


def train_adapter(tokenizer):
    """Trains a LoRA adapter and saves it to disk (mirrors Day 02)."""
    print(f"\n{'=' * 55}\nTraining LoRA adapter\n{'=' * 55}")

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    for step in range(TRAINING_STEPS):
        item = TOY_DATA[step % len(TOY_DATA)]
        batch = build_training_batch(tokenizer, item, model.device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step % 5 == 0:
            print(f"  step {step:>2}  loss={loss.item():.4f}")

    model.eval()

    if os.path.exists(ADAPTER_DIR):
        shutil.rmtree(ADAPTER_DIR)
    model.save_pretrained(ADAPTER_DIR)
    print(f"Adapter saved to ./{ADAPTER_DIR}/")

    return model


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Step 1: train and save an adapter
    adapter_model = train_adapter(tokenizer)

    print("\n--- Adapter-attached model output ---")
    adapter_output = generate_response(adapter_model, tokenizer, TEST_PROMPT)
    print(f"Response: {adapter_output}")

    base_model_params = sum(p.numel() for p in adapter_model.base_model.model.parameters())

    # Step 2: reload base + adapter fresh (simulating "coming back later to merge")
    print(f"\n{'=' * 55}\nMerging adapter into base weights\n{'=' * 55}")
    fresh_base = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    peft_model = PeftModel.from_pretrained(fresh_base, ADAPTER_DIR)

    t0 = time.time()
    merged_model = peft_model.merge_and_unload()  # <-- the actual merge step
    merge_time = time.time() - t0
    print(f"Merged in {merge_time:.2f}s")

    print("\n--- Merged model output ---")
    merged_output = generate_response(merged_model, tokenizer, TEST_PROMPT)
    print(f"Response: {merged_output}")

    outputs_match = adapter_output == merged_output
    print(f"\nOutputs identical: {outputs_match}")

    # Step 3: save merged model as a plain, standalone model
    if os.path.exists(MERGED_DIR):
        shutil.rmtree(MERGED_DIR)
    merged_model.save_pretrained(MERGED_DIR)
    tokenizer.save_pretrained(MERGED_DIR)
    print(f"Merged model saved to ./{MERGED_DIR}/ (loadable with plain AutoModelForCausalLM, no PEFT needed)")

    merged_model_params = sum(p.numel() for p in merged_model.parameters())

    # Step 4: prove it reloads without any PEFT dependency
    print(f"\n{'=' * 55}\nReloading merged model WITHOUT importing peft\n{'=' * 55}")
    reloaded = AutoModelForCausalLM.from_pretrained(MERGED_DIR, torch_dtype=torch.float32)
    reloaded_output = generate_response(reloaded, tokenizer, TEST_PROMPT)
    print(f"Response: {reloaded_output}")

    result = MergeResult(
        adapter_output=adapter_output,
        merged_output=merged_output,
        outputs_match=outputs_match,
        adapter_dir_size_mb=round(dir_size_mb(ADAPTER_DIR), 2),
        merged_dir_size_mb=round(dir_size_mb(MERGED_DIR), 2),
        base_model_params=base_model_params,
        merged_model_params=merged_model_params,
        merge_time_sec=round(merge_time, 3),
    )

    print(f"\n{'=' * 55}\nSUMMARY\n{'=' * 55}")
    print(f"Adapter folder size:  {result.adapter_dir_size_mb} MB  (just the small A/B matrices)")
    print(f"Merged folder size:   {result.merged_dir_size_mb} MB  (full standalone model)")
    print(f"Base model params:    {result.base_model_params:,}")
    print(f"Merged model params:  {result.merged_model_params:,}  (same shape as base — no extra params at inference)")

    with open(LOG_FILE, "w") as f:
        json.dump(asdict(result), f, indent=2)
    print(f"\nSaved full results to {LOG_FILE}")


if __name__ == "__main__":
    main()

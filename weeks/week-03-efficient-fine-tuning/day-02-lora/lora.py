"""
Day 02 — LoRA (Low-Rank Adaptation)
=====================================
Goal: Understand HOW LoRA works and see it in action — freezing a base
model, attaching small trainable adapter matrices, and comparing that
against what full fine-tuning would require.

LoRA's core idea:
  Instead of updating a full weight matrix W (shape d x d, potentially
  millions of parameters), freeze W entirely and learn a small update
  delta_W = B @ A, where:
    - A has shape (r, d)   — "down projection"
    - B has shape (d, r)   — "up projection"
    - r (the "rank") is tiny compared to d, e.g. 8 or 16

  Effective weight becomes: W_new = W + (alpha / r) * B @ A

  Since r << d, the number of trainable parameters drops by orders of
  magnitude — while still letting the model learn new behavior.

This script:
  1. Loads a small instruction-tuned model
  2. Shows the FULL fine-tuning parameter count (the "expensive" way)
  3. Wraps it with LoRA adapters via PEFT and shows the trainable count
  4. Sweeps across different ranks (r) to show the size/capacity trade-off
  5. Runs a tiny training loop on a toy dataset to prove the adapters
     actually learn something, comparing outputs before/after
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

# Toy dataset: teach the model a quirky, consistent sign-off behavior
# so we can clearly see whether the adapter learned anything.
TOY_DATA = [
    {"prompt": "What is the capital of France?", "response": "Paris. — Trained by Neha's LoRA lesson."},
    {"prompt": "What is 2 + 2?", "response": "4. — Trained by Neha's LoRA lesson."},
    {"prompt": "Name a primary color.", "response": "Blue. — Trained by Neha's LoRA lesson."},
    {"prompt": "What is the opposite of hot?", "response": "Cold. — Trained by Neha's LoRA lesson."},
]

TEST_PROMPT = "What is the capital of Japan?"


@dataclass
class RankResult:
    rank: int
    alpha: int
    trainable_params: int
    total_params: int
    trainable_pct: float


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def count_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def show_full_fine_tuning_cost(model):
    """What it would cost to fine-tune every parameter."""
    total = count_params(model)
    print(f"\n{'=' * 55}")
    print("FULL FINE-TUNING (baseline — no LoRA)")
    print(f"{'=' * 55}")
    print(f"Total parameters:      {total:,}")
    print(f"Trainable parameters:  {total:,}  (100% — every weight updates)")
    return total


def apply_lora(base_model, rank: int, alpha: int):
    """Wraps the base model with LoRA adapters on attention projections."""
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],  # attention query & value projections
        bias="none",
    )
    return get_peft_model(base_model, lora_config)


def sweep_ranks(model_name: str, total_params: int) -> list:
    """Compares trainable parameter counts across different LoRA ranks."""
    print(f"\n{'=' * 55}")
    print("LORA RANK SWEEP")
    print(f"{'=' * 55}")

    results = []
    for rank in [4, 8, 16, 32]:
        base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        peft_model = apply_lora(base, rank=rank, alpha=rank * 2)

        trainable = count_trainable_params(peft_model)
        pct = 100 * trainable / total_params

        print(f"r={rank:<4} alpha={rank * 2:<4} trainable={trainable:,} ({pct:.3f}% of full model)")

        results.append(RankResult(
            rank=rank, alpha=rank * 2, trainable_params=trainable,
            total_params=total_params, trainable_pct=round(pct, 4),
        ))

        del base, peft_model

    return results


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 40) -> str:
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(output_ids[0][inputs.shape[-1]:], skip_special_tokens=True).strip()


def build_training_batch(tokenizer, item: dict):
    """Formats one toy example as a chat-templated, labeled training example."""
    messages = [
        {"role": "user", "content": item["prompt"]},
        {"role": "assistant", "content": item["response"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    encoded["labels"] = encoded["input_ids"].clone()
    return encoded


def tiny_training_demo(model_name: str):
    """
    Trains LoRA adapters for a few steps on a toy dataset, then compares
    the model's output before and after — proof the adapters learned.
    """
    print(f"\n{'=' * 55}")
    print("TINY TRAINING DEMO (a few steps, toy data)")
    print(f"{'=' * 55}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

    print("\n--- BEFORE training ---")
    before_output = generate_response(base_model, tokenizer, TEST_PROMPT)
    print(f"Prompt:   {TEST_PROMPT}")
    print(f"Response: {before_output}")

    lora_model = apply_lora(base_model, rank=8, alpha=16)
    lora_model.train()
    optimizer = torch.optim.AdamW(
        [p for p in lora_model.parameters() if p.requires_grad], lr=1e-3
    )

    print("\nTraining on 4 toy examples for 20 steps...")
    t0 = time.time()
    for step in range(20):
        item = TOY_DATA[step % len(TOY_DATA)]
        batch = build_training_batch(tokenizer, item)
        batch = {k: v.to(lora_model.device) for k, v in batch.items()}

        outputs = lora_model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 5 == 0:
            print(f"  step {step:>2}  loss={loss.item():.4f}")

    train_time = time.time() - t0
    print(f"Training done in {train_time:.1f}s")

    lora_model.eval()
    print("\n--- AFTER training ---")
    after_output = generate_response(lora_model, tokenizer, TEST_PROMPT)
    print(f"Prompt:   {TEST_PROMPT}")
    print(f"Response: {after_output}")

    return {
        "before_output": before_output,
        "after_output": after_output,
        "train_time_sec": round(train_time, 1),
    }


def main():
    print("Loading base model for parameter counting...")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    total_params = show_full_fine_tuning_cost(base_model)
    del base_model

    rank_results = sweep_ranks(MODEL_NAME, total_params)
    training_demo = tiny_training_demo(MODEL_NAME)

    log = {
        "total_params": total_params,
        "rank_sweep": [asdict(r) for r in rank_results],
        "training_demo": training_demo,
    }
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nSaved full results to {LOG_FILE}")


if __name__ == "__main__":
    main()

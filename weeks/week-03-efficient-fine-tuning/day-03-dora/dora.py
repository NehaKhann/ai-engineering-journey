"""
Day 03 — DoRA (Weight-Decomposed Low-Rank Adaptation)
========================================================
Goal: Understand how DoRA improves on LoRA by decomposing weights into
magnitude and direction, and compare it directly against plain LoRA.

Recap — LoRA's update:
  W_new = W + (alpha / r) * (B @ A)

DoRA's insight: any weight matrix W can be decomposed into two parts:
  W = m * (V / ||V||)
  where:
    - V / ||V||  is the "direction" (a unit vector per output dimension)
    - m          is the "magnitude" (a scalar per output dimension)

Standard LoRA updates direction and magnitude together, entangled inside
the same low-rank update. DoRA untangles them:
  - The DIRECTION component gets the LoRA-style low-rank update (B @ A)
  - The MAGNITUDE component is trained separately, as its own small
    vector of trainable parameters

This mirrors how full fine-tuning naturally adjusts magnitude and
direction independently — which is why DoRA tends to get closer to full
fine-tuning quality than plain LoRA, at a very small extra parameter cost
(just one magnitude vector per adapted layer).

This script:
  1. Applies plain LoRA (recap from Day 02)
  2. Applies DoRA (via PEFT's `use_dora=True`) and compares trainable
     parameter counts
  3. Runs the same tiny training demo from Day 02 with both LoRA and
     DoRA, side by side, to compare loss curves and final behavior
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

# Same toy dataset as Day 02, so the comparison is apples-to-apples
TOY_DATA = [
    {"prompt": "What is the capital of France?", "response": "Paris. — Trained by Neha's DoRA lesson."},
    {"prompt": "What is 2 + 2?", "response": "4. — Trained by Neha's DoRA lesson."},
    {"prompt": "Name a primary color.", "response": "Blue. — Trained by Neha's DoRA lesson."},
    {"prompt": "What is the opposite of hot?", "response": "Cold. — Trained by Neha's DoRA lesson."},
]

TEST_PROMPT = "What is the capital of Japan?"


@dataclass
class AdapterResult:
    method: str
    trainable_params: int
    total_params: int
    trainable_pct: float
    final_loss: float
    train_time_sec: float
    before_output: str
    after_output: str


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def count_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_adapter_config(rank: int, alpha: int, use_dora: bool) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        use_dora=use_dora,  # <-- this single flag turns LoRA into DoRA
    )


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 40) -> str:
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(output_ids[0][inputs.shape[-1]:], skip_special_tokens=True).strip()


def build_training_batch(tokenizer, item: dict):
    messages = [
        {"role": "user", "content": item["prompt"]},
        {"role": "assistant", "content": item["response"]},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    encoded["labels"] = encoded["input_ids"].clone()
    return encoded


def run_adapter_experiment(method: str, tokenizer, total_params: int) -> AdapterResult:
    print(f"\n{'=' * 55}")
    print(f"METHOD: {method.upper()}")
    print(f"{'=' * 55}")

    use_dora = method == "dora"
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    config = build_adapter_config(rank=8, alpha=16, use_dora=use_dora)
    model = get_peft_model(base_model, config)

    trainable = count_trainable_params(model)
    pct = 100 * trainable / total_params
    print(f"Trainable parameters: {trainable:,} ({pct:.3f}% of full model)")

    print("\n--- BEFORE training ---")
    before_output = generate_response(model, tokenizer, TEST_PROMPT)
    print(f"Response: {before_output}")

    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )

    print(f"\nTraining for {TRAINING_STEPS} steps...")
    t0 = time.time()
    final_loss = 0.0
    for step in range(TRAINING_STEPS):
        item = TOY_DATA[step % len(TOY_DATA)]
        batch = build_training_batch(tokenizer, item)
        batch = {k: v.to(model.device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        final_loss = loss.item()

        if step % 5 == 0:
            print(f"  step {step:>2}  loss={loss.item():.4f}")

    train_time = time.time() - t0
    print(f"Training done in {train_time:.1f}s, final loss={final_loss:.4f}")

    model.eval()
    print("\n--- AFTER training ---")
    after_output = generate_response(model, tokenizer, TEST_PROMPT)
    print(f"Response: {after_output}")

    result = AdapterResult(
        method=method,
        trainable_params=trainable,
        total_params=total_params,
        trainable_pct=round(pct, 4),
        final_loss=round(final_loss, 4),
        train_time_sec=round(train_time, 1),
        before_output=before_output,
        after_output=after_output,
    )

    del base_model, model
    return result


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading base model for parameter counting...")
    reference_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    total_params = count_params(reference_model)
    del reference_model
    print(f"Total parameters: {total_params:,}")

    results = []
    for method in ["lora", "dora"]:
        results.append(run_adapter_experiment(method, tokenizer, total_params))

    print(f"\n{'=' * 65}")
    print(f"{'Method':<8}{'Trainable':<14}{'% of model':<14}{'Final loss':<14}{'Time (s)':<10}")
    print(f"{'=' * 65}")
    for r in results:
        print(f"{r.method:<8}{r.trainable_params:<14,}{r.trainable_pct:<14}{r.final_loss:<14}{r.train_time_sec:<10}")

    with open(LOG_FILE, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nSaved full results to {LOG_FILE}")


if __name__ == "__main__":
    main()

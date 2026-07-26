"""
Bonus — LoRA vs. DoRA at Scale: Does DoRA's Advantage Actually Show Up?
=========================================================================
Day 3 introduced DoRA and explained the theory: it takes LoRA's one
combined update and splits it into two separately-trained pieces —
magnitude and direction — which, in theory, should let it get closer to
full fine-tuning quality on real tasks.

But Day 3's demo trained on just 4 toy examples for 20 steps, and at
that scale, LoRA and DoRA looked basically identical — DoRA was just
slower, with no visible quality edge. That's expected: a 4-example demo
is great for seeing HOW something works, not for proving WHICH METHOD
WINS. Small demos don't have enough signal to show a difference that
(per the DoRA paper) is supposed to show up on harder, more diverse,
larger-scale tasks.

This script is the follow-up experiment: train LoRA and DoRA back-to-back
on the SAME 1,000+ real examples (from Hugging Face, not hardcoded),
with identical hyperparameters, and measure:

  1. Held-out perplexity (quantitative — lower is better)
  2. Training loss curves, overlaid (qualitative shape comparison)
  3. Wall-clock training time (DoRA is expected to be somewhat slower —
     is that gap still small at this scale, or does it grow?)
  4. A regression check on simple factual prompts (did either method
     corrupt basic knowledge?)

This does NOT prove DoRA always wins — 1,000+ examples on a 0.5B model
is still a modest-scale test. It's a bigger, more honest experiment than
the Day 3 toy demo, not a definitive verdict.
"""

import json
import time
from dataclasses import dataclass, asdict

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_NAME = "tatsu-lab/alpaca"

NUM_TRAIN_EXAMPLES = 1200   # 300x more than Day 3's 4-example toy demo
NUM_VAL_EXAMPLES = 150
MAX_SEQ_LENGTH = 256

# Identical adapter hyperparameters for both methods — the ONLY thing
# that changes between runs is use_dora=True/False.
LORA_RANK = 8
LORA_ALPHA = 16
LEARNING_RATE = 2e-4

LOG_FILE = "lora_vs_dora_results.json"
PLOT_FILE = "lora_vs_dora_comparison.png"

REGRESSION_PROMPTS = [
    "What is the capital of Japan?",
    "What is the capital of France?",
    "What is 12 multiplied by 8?",
    "Name the largest planet in our solar system.",
]


@dataclass
class MethodResult:
    method: str  # "lora" or "dora"
    trainable_params: int
    trainable_pct: float
    train_time_sec: float
    final_train_loss: float
    loss_history: list
    perplexity_after: float
    regression_after: dict


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def count_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----------------------------------------------------------------------
# Data: real, diverse examples — same dataset as the LoRA large-dataset bonus
# ----------------------------------------------------------------------
def load_and_prepare_dataset():
    print(f"Downloading '{DATASET_NAME}' from the Hugging Face Hub...")
    raw = load_dataset(DATASET_NAME, split="train")
    raw = raw.filter(lambda x: x["input"].strip() == "")
    raw = raw.shuffle(seed=42)

    total_needed = NUM_TRAIN_EXAMPLES + NUM_VAL_EXAMPLES
    if len(raw) < total_needed:
        raise ValueError(f"Dataset only has {len(raw)} usable rows, need {total_needed}.")

    train_data = raw.select(range(NUM_TRAIN_EXAMPLES))
    val_data = raw.select(range(NUM_TRAIN_EXAMPLES, total_needed))
    print(f"Loaded {len(train_data)} training examples, {len(val_data)} held-out validation examples.")
    return train_data, val_data


def build_training_batch(tokenizer, instruction: str, response: str, device):
    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": response},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH)
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


def compute_perplexity(model, tokenizer, val_data, device, sample_size: int = 50) -> float:
    model.eval()
    losses = []
    for item in val_data.select(range(min(sample_size, len(val_data)))):
        batch = build_training_batch(tokenizer, item["instruction"], item["output"], device)
        with torch.no_grad():
            outputs = model(**batch)
        losses.append(outputs.loss.item())
    avg_loss = sum(losses) / len(losses)
    return torch.exp(torch.tensor(avg_loss)).item()


def build_adapter_config(use_dora: bool) -> LoraConfig:
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        use_dora=use_dora,  # the ONLY difference between the two runs
    )


# ----------------------------------------------------------------------
# One full run: attach adapter (LoRA or DoRA), train, evaluate
# ----------------------------------------------------------------------
def run_method(method: str, tokenizer, train_data, val_data, device, total_params) -> MethodResult:
    print(f"\n{'=' * 60}\nMETHOD: {method.upper()}\n{'=' * 60}")

    use_dora = method == "dora"
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to(device)
    config = build_adapter_config(use_dora=use_dora)
    model = get_peft_model(base_model, config)

    trainable = count_trainable_params(model)
    pct = 100 * trainable / total_params
    print(f"Trainable parameters: {trainable:,} ({pct:.4f}% of full model)")

    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)

    loss_history = []
    final_loss = 0.0

    print(f"Training on {len(train_data)} real examples (1 epoch)...")
    t0 = time.time()
    for step, item in enumerate(train_data):
        batch = build_training_batch(tokenizer, item["instruction"], item["output"], device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        final_loss = loss.item()
        loss_history.append(final_loss)

        if step % 200 == 0:
            print(f"  step {step:>5}/{len(train_data)}  loss={final_loss:.4f}")

    train_time = time.time() - t0
    print(f"Training done in {train_time:.1f}s, final loss={final_loss:.4f}")

    model.eval()
    perplexity_after = compute_perplexity(model, tokenizer, val_data, device)
    print(f"Held-out perplexity ({method}): {perplexity_after:.2f}")

    regression_after = {}
    for prompt in REGRESSION_PROMPTS:
        regression_after[prompt] = generate_response(model, tokenizer, prompt)

    result = MethodResult(
        method=method,
        trainable_params=trainable,
        trainable_pct=round(pct, 4),
        train_time_sec=round(train_time, 1),
        final_train_loss=round(final_loss, 4),
        loss_history=loss_history,
        perplexity_after=round(perplexity_after, 2),
        regression_after=regression_after,
    )

    del base_model, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def plot_comparison(lora_result: MethodResult, dora_result: MethodResult, perplexity_before: float):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: overlaid, smoothed loss curves
    window = 50
    for result, color, label in [(lora_result, "#2a78d6", "LoRA"), (dora_result, "#eb6834", "DoRA")]:
        history = result.loss_history
        smoothed = [
            sum(history[max(0, i - window):i + 1]) / len(history[max(0, i - window):i + 1])
            for i in range(len(history))
        ]
        axes[0].plot(smoothed, color=color, linewidth=2, label=label)

    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Loss (smoothed)")
    axes[0].set_title(f"Training loss — LoRA vs. DoRA ({NUM_TRAIN_EXAMPLES} examples)")
    axes[0].legend()
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Right panel: held-out perplexity comparison
    labels = ["Base\n(before)", "LoRA\n(after)", "DoRA\n(after)"]
    values = [perplexity_before, lora_result.perplexity_after, dora_result.perplexity_after]
    colors = ["#9aa5b1", "#2a78d6", "#eb6834"]
    axes[1].bar(labels, values, color=colors)
    axes[1].set_ylabel("Perplexity (lower is better)")
    axes[1].set_title("Held-out perplexity comparison")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved comparison chart to {PLOT_FILE}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU detected — this will run on CPU, which will be slow for 1200+ examples x2 methods.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_data, val_data = load_and_prepare_dataset()

    # Measure the base model once, before either method trains — this is
    # the shared "before" reference point for both LoRA and DoRA.
    print("\n--- Measuring BASE model (shared reference point) ---")
    reference_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to(device)
    total_params = count_params(reference_model)
    perplexity_before = compute_perplexity(reference_model, tokenizer, val_data, device)
    print(f"Held-out perplexity BEFORE any training: {perplexity_before:.2f}")
    del reference_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Run both methods on identical data with identical hyperparameters —
    # only use_dora differs.
    lora_result = run_method("lora", tokenizer, train_data, val_data, device, total_params)
    dora_result = run_method("dora", tokenizer, train_data, val_data, device, total_params)

    # ------------------------------------------------------------------
    # Side-by-side comparison
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("LORA vs. DORA — AT SCALE COMPARISON")
    print(f"{'=' * 70}")
    print(f"{'Metric':<28}{'LoRA':<20}{'DoRA':<20}")
    print(f"{'Trainable params':<28}{lora_result.trainable_params:<20,}{dora_result.trainable_params:<20,}")
    print(f"{'Trainable %':<28}{lora_result.trainable_pct:<20}{dora_result.trainable_pct:<20}")
    print(f"{'Train time (s)':<28}{lora_result.train_time_sec:<20}{dora_result.train_time_sec:<20}")
    print(f"{'Final train loss':<28}{lora_result.final_train_loss:<20}{dora_result.final_train_loss:<20}")
    print(f"{'Held-out perplexity':<28}{lora_result.perplexity_after:<20}{dora_result.perplexity_after:<20}")

    time_diff_pct = 100 * (dora_result.train_time_sec - lora_result.train_time_sec) / lora_result.train_time_sec
    perplexity_diff_pct = 100 * (lora_result.perplexity_after - dora_result.perplexity_after) / lora_result.perplexity_after

    print(f"\nDoRA took {time_diff_pct:+.1f}% {'longer' if time_diff_pct > 0 else 'less time'} than LoRA to train.")
    if perplexity_diff_pct > 0:
        print(f"DoRA's held-out perplexity was {perplexity_diff_pct:.1f}% LOWER than LoRA's (DoRA generalized better here).")
    elif perplexity_diff_pct < 0:
        print(f"DoRA's held-out perplexity was {abs(perplexity_diff_pct):.1f}% HIGHER than LoRA's (LoRA generalized better here).")
    else:
        print("DoRA and LoRA produced identical held-out perplexity in this run.")

    print(f"\n{'=' * 70}")
    print("REGRESSION CHECK — same 4 factual prompts, both methods")
    print(f"{'=' * 70}")
    for prompt in REGRESSION_PROMPTS:
        print(f"\nQ: {prompt}")
        print(f"  LoRA: {lora_result.regression_after[prompt]}")
        print(f"  DoRA: {dora_result.regression_after[prompt]}")

    try:
        plot_comparison(lora_result, dora_result, perplexity_before)
    except ImportError:
        print("\nmatplotlib not installed — skipping chart. Run: pip install matplotlib")

    log = {
        "dataset": DATASET_NAME,
        "num_train_examples": NUM_TRAIN_EXAMPLES,
        "num_val_examples": NUM_VAL_EXAMPLES,
        "perplexity_before_any_training": round(perplexity_before, 2),
        "lora": asdict(lora_result),
        "dora": asdict(dora_result),
    }
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nSaved full results to {LOG_FILE}")
    print("\nRemember: this is one run, one model size, one dataset — a bigger, more")
    print("honest test than Day 3's toy demo, but still not a universal verdict on")
    print("LoRA vs. DoRA. Treat this as directional evidence, not a final answer.")


if __name__ == "__main__":
    main()

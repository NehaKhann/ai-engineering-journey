"""
Bonus — LoRA Training on a Real, Larger Dataset
==================================================
Fixes a problem you'll notice in Day 2: training LoRA on just 4 hardcoded
toy examples for 20 repeated steps causes the model to "forget" basic
facts it already knew. In Day 2's demo, after training on 4 unrelated
trivia questions, asking a NEW trivia question ("capital of Japan?")
sometimes returned a wrong, corrupted answer ("Beijing") instead of a
sensible one.

This is called CATASTROPHIC FORGETTING. It happens here because:
  - The training set is tiny (4 examples) and highly repetitive
    (the same 4 examples looped 5x over 20 steps)
  - With so little diversity, the optimizer doesn't learn a general
    pattern — it just memorizes those 4 exact input/output pairs,
    and in doing so, overwrites unrelated knowledge the base model
    already had.

The fix isn't a different LoRA setting — it's TRAINING DATA. This script
trains the exact same LoRA setup from Day 2, but on thousands of diverse,
real instruction-following examples pulled live from the Hugging Face
`datasets` library (no hardcoded JSON), so the model sees enough variety
to learn a general pattern instead of memorizing a handful of strings.

We measure the fix two ways:
  1. REGRESSION CHECK — ask the same trivia questions from Day 2 before
     and after training. If forgetting is fixed, these answers should
     stay correct after training, not corrupt into wrong answers.
  2. HELD-OUT PERPLEXITY — measure how well the model predicts text it
     never saw during training, before and after. This should improve
     (lower perplexity = better), which is quantitative proof the model
     learned a general pattern rather than overfitting to specific examples.
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

# Alpaca: ~52,000 diverse instruction/response pairs, publicly available,
# no login required. Swap for "databricks/databricks-dolly-15k" if you
# want an alternative (field names differ slightly — see the README).
DATASET_NAME = "tatsu-lab/alpaca"

NUM_TRAIN_EXAMPLES = 2000   # real training data, vs Day 2's 4 examples
NUM_VAL_EXAMPLES = 200      # held out, never trained on — used for perplexity
MAX_SEQ_LENGTH = 256
LORA_RANK = 8               # same as Day 2, so this is an apples-to-apples fix
LORA_ALPHA = 16
LEARNING_RATE = 2e-4

LOG_FILE = "experiment_log.json"
PLOT_FILE = "training_loss_curve.png"

# The exact kind of question that broke in Day 2's demo — used here as a
# regression check, not as training data.
REGRESSION_PROMPTS = [
    "What is the capital of Japan?",
    "What is the capital of France?",
    "What is 12 multiplied by 8?",
    "Name the largest planet in our solar system.",
]


@dataclass
class RegressionCheck:
    prompt: str
    before: str
    after: str


@dataclass
class ExperimentResult:
    dataset: str
    num_train_examples: int
    num_val_examples: int
    trainable_params: int
    total_params: int
    trainable_pct: float
    train_time_sec: float
    final_train_loss: float
    perplexity_before: float
    perplexity_after: float
    regression_checks: list


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def count_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----------------------------------------------------------------------
# Step 1: Load a REAL dataset from the Hugging Face Hub — no hardcoded JSON
# ----------------------------------------------------------------------
def load_and_prepare_dataset():
    print(f"Downloading '{DATASET_NAME}' from the Hugging Face Hub...")
    raw = load_dataset(DATASET_NAME, split="train")

    # Keep only single-turn examples (no extra "input" context) so every
    # example has the same simple shape: one instruction, one response.
    raw = raw.filter(lambda x: x["input"].strip() == "")
    raw = raw.shuffle(seed=42)

    total_needed = NUM_TRAIN_EXAMPLES + NUM_VAL_EXAMPLES
    if len(raw) < total_needed:
        raise ValueError(
            f"Dataset only has {len(raw)} usable rows, need {total_needed}. "
            f"Lower NUM_TRAIN_EXAMPLES or NUM_VAL_EXAMPLES."
        )

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


# ----------------------------------------------------------------------
# Step 2: Measure perplexity on held-out data — the quantitative proof
# ----------------------------------------------------------------------
def compute_perplexity(model, tokenizer, val_data, device, sample_size: int = 50) -> float:
    """
    Lower perplexity = the model is less "surprised" by held-out text =
    better general understanding. We only sample a subset for speed;
    increase sample_size for a more precise (but slower) estimate.
    """
    model.eval()
    losses = []
    for item in val_data.select(range(min(sample_size, len(val_data)))):
        batch = build_training_batch(tokenizer, item["instruction"], item["output"], device)
        with torch.no_grad():
            outputs = model(**batch)
        losses.append(outputs.loss.item())

    avg_loss = sum(losses) / len(losses)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity


# ----------------------------------------------------------------------
# Step 3: Train on the real dataset
# ----------------------------------------------------------------------
def train(model, tokenizer, train_data, device):
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
    return final_loss, train_time, loss_history


def plot_loss_curve(loss_history: list):
    import matplotlib.pyplot as plt

    # Smooth the noisy per-step loss with a simple rolling average, so the
    # overall downward trend is easy to see instead of a jagged line.
    window = 50
    smoothed = [
        sum(loss_history[max(0, i - window):i + 1]) / len(loss_history[max(0, i - window):i + 1])
        for i in range(len(loss_history))
    ]

    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, color="#c9d6e8", linewidth=1, label="raw loss (per step)")
    plt.plot(smoothed, color="#2a78d6", linewidth=2, label=f"smoothed (rolling avg, window={window})")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title(f"Training loss over {len(loss_history)} steps — {NUM_TRAIN_EXAMPLES} real examples")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved loss curve to {PLOT_FILE}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU detected — this will run on CPU, which will be slow for 2000 examples.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_data, val_data = load_and_prepare_dataset()

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to(device)
    total_params = count_params(base_model)

    # --- BEFORE: regression check + held-out perplexity, on the untouched base model ---
    print("\n--- BEFORE training ---")
    regression_before = {}
    for prompt in REGRESSION_PROMPTS:
        response = generate_response(base_model, tokenizer, prompt)
        print(f"Q: {prompt}\nA: {response}\n")
        regression_before[prompt] = response

    perplexity_before = compute_perplexity(base_model, tokenizer, val_data, device)
    print(f"Held-out perplexity BEFORE training: {perplexity_before:.2f}")

    # --- Attach LoRA (identical config to Day 2) ---
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    trainable = count_trainable_params(model)
    pct = 100 * trainable / total_params
    print(f"\nTrainable parameters: {trainable:,} ({pct:.3f}% of full model)")

    # --- Train on real data ---
    final_loss, train_time, loss_history = train(model, tokenizer, train_data, device)
    print(f"\nTraining done in {train_time:.1f}s, final loss={final_loss:.4f}")

    # --- AFTER: same regression check + held-out perplexity ---
    model.eval()
    print("\n--- AFTER training ---")
    regression_checks = []
    for prompt in REGRESSION_PROMPTS:
        response = generate_response(model, tokenizer, prompt)
        print(f"Q: {prompt}\nA: {response}\n")
        regression_checks.append(RegressionCheck(prompt=prompt, before=regression_before[prompt], after=response))

    perplexity_after = compute_perplexity(model, tokenizer, val_data, device)
    print(f"Held-out perplexity AFTER training: {perplexity_after:.2f}")

    print(f"\n{'=' * 60}")
    print("REGRESSION CHECK — did basic knowledge survive training?")
    print(f"{'=' * 60}")
    for check in regression_checks:
        changed = "CHANGED" if check.before != check.after else "same"
        print(f"[{changed}] {check.prompt}")
        print(f"   before: {check.before}")
        print(f"   after:  {check.after}")

    print(f"\nPerplexity before: {perplexity_before:.2f}  |  after: {perplexity_after:.2f}")
    print("(Lower is better. A drop here means the model generalized; a corrupted")
    print(" regression answer alongside a WORSE perplexity would indicate forgetting.)")

    try:
        plot_loss_curve(loss_history)
    except ImportError:
        print("matplotlib not installed — skipping loss curve. Run: pip install matplotlib")

    result = ExperimentResult(
        dataset=DATASET_NAME,
        num_train_examples=NUM_TRAIN_EXAMPLES,
        num_val_examples=NUM_VAL_EXAMPLES,
        trainable_params=trainable,
        total_params=total_params,
        trainable_pct=round(pct, 4),
        train_time_sec=round(train_time, 1),
        final_train_loss=round(final_loss, 4),
        perplexity_before=round(perplexity_before, 2),
        perplexity_after=round(perplexity_after, 2),
        regression_checks=[asdict(c) for c in regression_checks],
    )

    with open(LOG_FILE, "w") as f:
        json.dump(asdict(result), f, indent=2)
    print(f"\nSaved full results to {LOG_FILE}")


if __name__ == "__main__":
    main()

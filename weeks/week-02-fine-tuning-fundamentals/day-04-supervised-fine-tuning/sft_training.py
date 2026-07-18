"""
Day 4 — Supervised Fine-Tuning (SFT)
=====================================

Loads the cleaned Day 3 dataset (JSONL), formats it for GPT-2,
trains with SFTTrainer, visualizes the training loss, and compares
responses before/after fine-tuning.
"""

import json
import os
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# TRL moved several SFTTrainer arguments (max_seq_length, dataset_text_field,
# packing, etc.) into a separate SFTConfig object in newer releases. We try
# the modern import first and fall back so this script runs either way —
# if you hit an error here, it's a strong signal your installed trl version
# expects one API but not the other; check `pip show trl`.
from trl import SFTTrainer
try:
    from trl import SFTConfig
    USE_SFT_CONFIG = True
except ImportError:
    from transformers import TrainingArguments
    USE_SFT_CONFIG = False

print("=" * 75)
print("Day 4 — Supervised Fine-Tuning (SFT)")
print("=" * 75 + "\n")

# =============================================
# PART 1: LOAD AND CLEAN DATASET
# =============================================
print("PART 1: Loading and Cleaning Dataset")
print("-" * 60)

script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "../day-03-dataset-preparation/cybersecurity_dataset.jsonl")

if not os.path.exists(dataset_path):
    print("Day 3 dataset not found. Please run Day 3 first to generate the JSONL.")
    exit()

with open(dataset_path) as f:
    raw_data = [json.loads(line) for line in f]
print(f"Loaded dataset from: {dataset_path}")
print(f"Total examples loaded: {len(raw_data)}")

# Clean: remove duplicates and missing fields (same logic as Day 3).
# Re-running this here — rather than trusting Day 3's output blindly —
# means this script is safe to run standalone against any raw JSONL.
missing = [i for i, item in enumerate(raw_data)
           if not item["instruction"].strip() or not item["response"].strip()]
seen = set()
duplicates = []
for i, item in enumerate(raw_data):
    key = (item["instruction"].strip().lower(), item["response"].strip().lower())
    if key in seen:
        duplicates.append(i)
    seen.add(key)
bad_indices = set(missing) | set(duplicates)
if bad_indices:
    raw_data = [item for i, item in enumerate(raw_data) if i not in bad_indices]
    print(f"Cleaned: removed {len(bad_indices)} bad rows")
print(f"Clean examples: {len(raw_data)}\n")

if len(raw_data) < 20:
    print("Note: fine-tuning on this few examples is a demo of the workflow,")
    print("not a production-grade training run. Expect subtle, not dramatic,")
    print("differences between before/after responses — see Part 7.\n")

# =============================================
# PART 2: FORMAT FOR SFT
# =============================================
print("=" * 75)
print("PART 2: Formatting for SFT")
print("=" * 75 + "\n")

# GPT-2 is a base model with no chat template (Day 1/Day 2) — so instead of
# <|im_start|> markers, we teach it a much simpler learnable pattern:
# "text starting with 'Instruction:' is usually followed by 'Response:'".
def format_for_sft(example):
    return {
        "text": f"Instruction:\n{example['instruction']}\n\nResponse:\n{example['response']}"
    }

formatted = [format_for_sft(ex) for ex in raw_data]
dataset = Dataset.from_list(formatted)

print("Formatted example:")
print(f"  {repr(formatted[0]['text'])}\n")

split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

print(f"Training examples:   {len(train_dataset)}")
print(f"Validation examples: {len(eval_dataset)}\n")

# =============================================
# PART 3: LOAD BASE MODEL (GPT-2)
# =============================================
print("=" * 75)
print("PART 3: Loading Base Model (GPT-2)")
print("=" * 75 + "\n")

model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no dedicated pad token

print(f"Model: {model_name}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

# =============================================
# PART 4: TEST BEFORE FINE-TUNING
# =============================================
print("=" * 75)
print("PART 4: Testing Model Before Fine-Tuning")
print("=" * 75 + "\n")

test_prompt = "Instruction:\nWhat is a firewall?\n\nResponse:\n"

pipe_before = pipeline("text-generation", model=model, tokenizer=tokenizer)
result = pipe_before(test_prompt, max_new_tokens=60, temperature=0.7,
                      do_sample=True, pad_token_id=tokenizer.eos_token_id)

generated = result[0]["generated_text"]
response_before = generated[len(test_prompt):].strip()
print(f"Prompt: {test_prompt}")
print(f"Response (before SFT):")
print(f"  {response_before}\n")

# =============================================
# PART 5: CONFIGURE AND RUN SFT
# =============================================
print("=" * 75)
print("PART 5: Supervised Fine-Tuning")
print("=" * 75 + "\n")

response_template = "Response:\n"
output_dir = "./gpt2-cybersecurity-sft"
common_args = dict(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    max_steps=60,
    learning_rate=5e-5,
    logging_steps=5,
    eval_steps=10,
    save_steps=60,
    save_total_limit=1,
    eval_strategy="steps",
    report_to="none",
    fp16=False,
    use_cpu=True,
)

if USE_SFT_CONFIG:
    training_args = SFTConfig(**common_args, dataset_text_field="text", max_length=256)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
else:
    training_args = TrainingArguments(**common_args, remove_unused_columns=False)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=256,
        response_template=response_template,
    )

print(f"Using {'SFTConfig' if USE_SFT_CONFIG else 'TrainingArguments'} "
      f"(detected from installed trl version)")
print("Starting fine-tuning...\n")
trainer.train()
print("\nFine-tuning complete!\n")

# =============================================
# PART 6: SAVE MODEL + PLOT TRAINING LOSS
# =============================================
print("=" * 75)
print("PART 6: Saving Fine-Tuned Model")
print("=" * 75 + "\n")

trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model saved to: {output_dir}\n")

# Trainer automatically logs loss at every `logging_steps` — pull it out
# of the log history so we can actually SEE learning happen, not just
# read two text samples and hope it worked.
log_history = trainer.state.log_history
train_steps = [entry["step"] for entry in log_history if "loss" in entry]
train_losses = [entry["loss"] for entry in log_history if "loss" in entry]
eval_steps = [entry["step"] for entry in log_history if "eval_loss" in entry]
eval_losses = [entry["eval_loss"] for entry in log_history if "eval_loss" in entry]

plt.figure(figsize=(8, 5))
plt.plot(train_steps, train_losses, marker='o', label="Training loss", color='steelblue')
if eval_losses:
    plt.plot(eval_steps, eval_losses, marker='s', label="Validation loss", color='lightcoral')
plt.title("SFT Training Loss Over Steps", fontsize=13)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("day4_training_loss.png", dpi=150, bbox_inches="tight")
print("Saved training loss chart to day4_training_loss.png\n")

# =============================================
# PART 7: TEST AFTER FINE-TUNING
# =============================================
print("=" * 75)
print("PART 7: Testing Model After Fine-Tuning")
print("=" * 75 + "\n")

ft_model = AutoModelForCausalLM.from_pretrained(output_dir)
ft_tokenizer = AutoTokenizer.from_pretrained(output_dir)

pipe_after = pipeline("text-generation", model=ft_model, tokenizer=ft_tokenizer)
result = pipe_after(test_prompt, max_new_tokens=60, temperature=0.7,
                     do_sample=True, pad_token_id=ft_tokenizer.eos_token_id)

generated = result[0]["generated_text"]
response_after = generated[len(test_prompt):].strip()
print(f"Prompt: {test_prompt}")
print(f"Response (after SFT):")
print(f"  {response_after}\n")

# =============================================
# SUMMARY
# =============================================
print("=" * 75)
print("SUMMARY")
print("=" * 75)

print(f"""
Before SFT:
  {response_before}

After SFT:
  {response_after}

Training loss: {train_losses[0]:.3f} -> {train_losses[-1]:.3f} over {len(train_steps)} logged steps

Key observations:
  * Falling training loss confirms the model IS learning the pattern,
    even if the generated text doesn't look dramatically different yet —
    {len(raw_data)} examples is enough to demonstrate the pipeline, not
    enough to fully reshape GPT-2's behavior.
  * Real cybersecurity fine-tuning would use hundreds-to-thousands of
    examples and more training steps.
  * The model is saved and can be loaded for inference.
""")

print("Done! The fine-tuned model is ready in ./gpt2-cybersecurity-sft/")

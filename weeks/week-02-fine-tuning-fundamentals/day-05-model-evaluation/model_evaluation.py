import json
import os
import math
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

print("=" * 75)
print("Day 5 — Model Evaluation: Base vs Fine-Tuned")
print("=" * 75 + "\n")

# =============================================
# PART 1: LOAD MODELS
# =============================================
print("PART 1: Loading Models")
print("-" * 60)

# Base model
print("Loading base model (GPT-2)...")
base_model = AutoModelForCausalLM.from_pretrained("gpt2")
base_tokenizer = AutoTokenizer.from_pretrained("gpt2")
base_tokenizer.pad_token = base_tokenizer.eos_token

# Fine-tuned model
ft_path = "../day-04-supervised-fine-tuning/gpt2-cybersecurity-sft"
if not os.path.exists(ft_path):
    print(f"\nFine-tuned model not found at: {ft_path}")
    print("Run Day 4 (sft_training.py) first to create the fine-tuned model.")
    exit()

print("Loading fine-tuned model...")
ft_model = AutoModelForCausalLM.from_pretrained(ft_path)
ft_tokenizer = AutoTokenizer.from_pretrained(ft_path)
ft_tokenizer.pad_token = ft_tokenizer.eos_token

print("Both models loaded successfully.\n")

# =============================================
# PART 2: COMPARE RESPONSES
# =============================================
print("=" * 75)
print("PART 2: Side-by-Side Response Comparison")
print("=" * 75 + "\n")

test_prompts = [
    "What is a firewall?",
    "Explain how encryption works.",
    "What is a DDoS attack?",
    "Define social engineering.",
    "What is ransomware?",
]

base_pipe = pipeline("text-generation", model=base_model, tokenizer=base_tokenizer)
ft_pipe = pipeline("text-generation", model=ft_model, tokenizer=ft_tokenizer)

results = []

for prompt in test_prompts:
    formatted_prompt = f"Instruction:\n{prompt}\n\nResponse:\n"

    base_out = base_pipe(formatted_prompt, max_new_tokens=50, temperature=0.7, do_sample=True, pad_token_id=base_tokenizer.eos_token_id)
    ft_out = ft_pipe(formatted_prompt, max_new_tokens=50, temperature=0.7, do_sample=True, pad_token_id=ft_tokenizer.eos_token_id)

    base_text = base_out[0]["generated_text"][len(formatted_prompt):].strip()
    ft_text = ft_out[0]["generated_text"][len(formatted_prompt):].strip()

    results.append({"prompt": prompt, "base": base_text, "ft": ft_text})

    print(f"Prompt: {prompt}")
    print(f"  BEFORE SFT: {base_text[:100]}{'...' if len(base_text) > 100 else ''}")
    print(f"  AFTER SFT:  {ft_text[:100]}{'...' if len(ft_text) > 100 else ''}")
    print()

# =============================================
# PART 3: PERPLEXITY COMPARISON
# =============================================
print("=" * 75)
print("PART 3: Perplexity Comparison")
print("=" * 75 + "\n")

print("Perplexity measures how well the model predicts text.")
print("Lower perplexity = better predictions.\n")

# Load validation set from Day 3
val_path = "../day-03-dataset-preparation/cybersecurity_val.jsonl"
if os.path.exists(val_path):
    with open(val_path) as f:
        val_data = [json.loads(line) for line in f]
    print(f"Loaded {len(val_data)} validation examples.")
else:
    val_data = [results[0]]
    print("Validation set not found. Using first test prompt.\n")

def compute_perplexity(model, tokenizer, texts):
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            formatted = f"Instruction:\n{text['instruction']}\n\nResponse:\n{text['response']}"
            encoded = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=256)
            input_ids = encoded["input_ids"]
            labels = input_ids.clone()

            # Mask instruction tokens in labels
            response_start = formatted.find("Response:\n") + len("Response:\n")
            prefix = formatted[:response_start]
            prefix_len = len(tokenizer.encode(prefix))
            labels[:, :prefix_len] = -100

            outputs = model(input_ids, labels=labels)
            total_loss += outputs.loss.item() * (labels != -100).sum().item()
            total_tokens += (labels != -100).sum().item()

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    return math.exp(avg_loss)

base_ppl = compute_perplexity(base_model, base_tokenizer, val_data)
ft_ppl = compute_perplexity(ft_model, ft_tokenizer, val_data)

print(f"Base model perplexity:      {base_ppl:.2f}")
print(f"Fine-tuned model perplexity: {ft_ppl:.2f}")
print(f"Improvement:                 {((base_ppl - ft_ppl) / base_ppl * 100):.1f}%\n")

# =============================================
# VISUALIZATION
# =============================================
print("=" * 75)
print("VISUALIZATION")
print("=" * 75 + "\n")

plt.figure(figsize=(14, 5))

# Perplexity comparison
plt.subplot(1, 2, 1)
colors = ['lightcoral', 'steelblue']
bars = plt.bar(["Base GPT-2", "Fine-Tuned"], [base_ppl, ft_ppl], color=colors, edgecolor='navy')
plt.title("Perplexity Comparison (Lower is Better)", fontsize=13)
plt.ylabel("Perplexity")
plt.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, [base_ppl, ft_ppl]):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(base_ppl, ft_ppl)*0.02,
             f"{val:.1f}", ha='center', fontsize=12)

# Response length comparison
plt.subplot(1, 2, 2)
avg_base_len = sum(len(r["base"]) for r in results) / len(results)
avg_ft_len = sum(len(r["ft"]) for r in results) / len(results)
bars = plt.bar(["Base GPT-2", "Fine-Tuned"], [avg_base_len, avg_ft_len], color=colors, edgecolor='navy')
plt.title("Average Response Length (Characters)", fontsize=13)
plt.ylabel("Characters")
plt.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, [avg_base_len, avg_ft_len]):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"{val:.0f}", ha='center', fontsize=12)

plt.tight_layout()
plt.show()

# =============================================
# SUMMARY
# =============================================
print("=" * 75)
print("EVALUATION SUMMARY")
print("=" * 75 + "\n")

print(f"Metric                Base GPT-2     Fine-Tuned     Improvement")
print(f"─────────────────────  ─────────────  ─────────────  ──────────")
print(f"Perplexity (↓better)  {base_ppl:>8.2f}      {ft_ppl:>8.2f}      {((base_ppl - ft_ppl) / base_ppl * 100):>+5.1f}%")
print(f"Avg response length   {avg_base_len:>8.0f} chars  {avg_ft_len:>8.0f} chars")
print()

print("Qualitative assessment:")
print("• Base model generates generic text completions")
print("• Fine-tuned model produces more domain-relevant responses")
print("• Lower perplexity confirms the model learned from the training data")
print("• Evaluation should always be both quantitative and qualitative")
print()
print("Done!")

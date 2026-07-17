import json
import os
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, pipeline
from trl import SFTTrainer

print("=" * 75)
print("Day 4 — Supervised Fine-Tuning (SFT)")
print("=" * 75 + "\n")

# =============================================
# PART 1: LOAD DATASET
# =============================================
print("PART 1: Loading Dataset")
print("-" * 60)

dataset_path = "../day-03-dataset-preparation/cybersecurity_dataset.jsonl"

if os.path.exists(dataset_path):
    with open(dataset_path) as f:
        raw_data = [json.loads(line) for line in f]
    print(f"Loaded dataset from: {dataset_path}")
else:
    print("Day 3 dataset not found. Creating inline dataset...")
    raw_data = [
        {"instruction": "What is a firewall?", "response": "A firewall is a network security device that monitors and controls incoming and outgoing traffic based on predetermined security rules."},
        {"instruction": "Explain how encryption works.", "response": "Encryption converts plaintext data into ciphertext using an algorithm and a key. Only someone with the correct key can decrypt it."},
        {"instruction": "What is a DDoS attack?", "response": "A Distributed Denial of Service attack overwhelms a target with traffic from multiple sources, making it unavailable to legitimate users."},
        {"instruction": "Define social engineering.", "response": "Social engineering manipulates people into giving up access or information. It targets human psychology rather than software."},
        {"instruction": "What is the difference between symmetric and asymmetric encryption?", "response": "Symmetric encryption uses one key for both encryption and decryption. Asymmetric uses a public-private key pair."},
        {"instruction": "Explain zero-day vulnerability.", "response": "A zero-day vulnerability is a software flaw unknown to the vendor. No patch exists, allowing attackers to exploit it before a fix is released."},
        {"instruction": "What is multi-factor authentication?", "response": "MFA requires two or more verification factors: something you know, something you have, or something you are."},
        {"instruction": "How does a VPN protect privacy?", "response": "A VPN encrypts traffic between your device and a remote server, hiding your IP address and activity from your ISP and attackers."},
        {"instruction": "What is ransomware?", "response": "Ransomware encrypts a victim's files and demands payment for the decryption key. It is often delivered through phishing emails."},
        {"instruction": "Define network segmentation.", "response": "Network segmentation divides a network into smaller subnetworks to improve security and limit the spread of breaches."},
    ]

print(f"Total examples: {len(raw_data)}\n")

# Format each example with instruction/response template
def format_for_sft(example):
    return {
        "text": f"Instruction:\n{example['instruction']}\n\nResponse:\n{example['response']}"
    }

formatted = [format_for_sft(ex) for ex in raw_data]
dataset = Dataset.from_list(formatted)

# Show formatted examples
print("Formatted example:")
print(f"  {repr(formatted[0]['text'])}\n")

# Split 80/20
split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

print(f"Training examples:   {len(train_dataset)}")
print(f"Validation examples: {len(eval_dataset)}\n")

# =============================================
# PART 2: LOAD BASE MODEL
# =============================================
print("=" * 75)
print("PART 2: Loading Base Model (GPT-2)")
print("=" * 75 + "\n")

model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

print(f"Model: {model_name}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print()

# =============================================
# PART 3: TEST BEFORE FINE-TUNING
# =============================================
print("=" * 75)
print("PART 3: Testing Model Before Fine-Tuning")
print("=" * 75 + "\n")

test_prompt = "Instruction:\nWhat is a firewall?\n\nResponse:\n"

pipe_before = pipeline("text-generation", model=model, tokenizer=tokenizer)
result = pipe_before(test_prompt, max_new_tokens=60, temperature=0.7, do_sample=True, pad_token_id=tokenizer.eos_token_id)

generated = result[0]["generated_text"]
response = generated[len(test_prompt):].strip()
print(f"Prompt: {test_prompt}")
print(f"Response (before SFT):")
print(f"  {response}\n")

# =============================================
# PART 4: CONFIGURE AND RUN SFT
# =============================================
print("=" * 75)
print("PART 4: Supervised Fine-Tuning")
print("=" * 75 + "\n")

training_args = TrainingArguments(
    output_dir="./gpt2-cybersecurity-sft",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    max_steps=30,
    learning_rate=3e-5,
    logging_steps=5,
    eval_steps=10,
    save_steps=30,
    save_total_limit=1,
    eval_strategy="steps",
    report_to="none",
    fp16=False,
    remove_unused_columns=False,
)

response_template = "Response:\n"

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

print("Starting fine-tuning...\n")
trainer.train()
print("\nFine-tuning complete!\n")

# =============================================
# PART 5: SAVE MODEL
# =============================================
print("=" * 75)
print("PART 5: Saving Fine-Tuned Model")
print("=" * 75 + "\n")

save_path = "./gpt2-cybersecurity-sft"
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
print(f"Model saved to: {save_path}\n")

# =============================================
# PART 6: TEST AFTER FINE-TUNING
# =============================================
print("=" * 75)
print("PART 6: Testing Model After Fine-Tuning")
print("=" * 75 + "\n")

ft_model = AutoModelForCausalLM.from_pretrained(save_path)
ft_tokenizer = AutoTokenizer.from_pretrained(save_path)

pipe_after = pipeline("text-generation", model=ft_model, tokenizer=ft_tokenizer)
result = pipe_after(test_prompt, max_new_tokens=60, temperature=0.7, do_sample=True, pad_token_id=ft_tokenizer.eos_token_id)

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
  {response}
  
After SFT:
  {response_after}

Key observations:
• The fine-tuned model should produce more relevant cybersecurity responses
• SFT teaches the model domain-specific knowledge without changing its architecture
• Training took only 30 steps on a small dataset
• The model is now saved and can be loaded for inference or further fine-tuning
""")

print("Done! The fine-tuned model is ready in ./gpt2-cybersecurity-sft/")

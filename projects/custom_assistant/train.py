import json
import os
import math
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, pipeline
from trl import SFTTrainer

# =============================================
# CONFIGURATION
# =============================================
DOMAIN = "cybersecurity"
MODEL_NAME = "gpt2"
OUTPUT_DIR = f"./models/{DOMAIN}-assistant"
MAX_STEPS = 50
BATCH_SIZE = 1
LEARNING_RATE = 3e-5
MAX_SEQ_LENGTH = 256
TEST_PROMPTS = [
    "What is a firewall?",
    "Explain how encryption works.",
    "What is the CIA triad?",
]

# =============================================
# DOMAIN DATASETS
# =============================================
DOMAIN_DATA = {
    "cybersecurity": [
        {"instruction": "What is a firewall?", "response": "A firewall is a network security device that monitors and controls incoming and outgoing traffic based on security rules. It acts as a barrier between trusted and untrusted networks."},
        {"instruction": "Explain how encryption works.", "response": "Encryption converts plaintext into ciphertext using an algorithm and a key. Only someone with the correct key can decrypt it. This ensures data confidentiality."},
        {"instruction": "What is a DDoS attack?", "response": "A DDoS attack overwhelms a target with traffic from multiple sources, making it unavailable to legitimate users. It is a common network-level attack."},
        {"instruction": "Define social engineering.", "response": "Social engineering manipulates people into giving up access or information. It targets human psychology rather than software vulnerabilities."},
        {"instruction": "What is symmetric vs asymmetric encryption?", "response": "Symmetric encryption uses one key for both encryption and decryption. Asymmetric uses a public-private key pair. Symmetric is faster but requires secure key exchange."},
        {"instruction": "Explain zero-day vulnerability.", "response": "A zero-day vulnerability is a software flaw unknown to the vendor. No patch exists, allowing attackers to exploit it before a fix is released."},
        {"instruction": "What is multi-factor authentication?", "response": "MFA requires two or more verification factors: something you know, something you have, or something you are. This adds extra security beyond passwords alone."},
        {"instruction": "How does a VPN protect privacy?", "response": "A VPN encrypts traffic between your device and a remote server, hiding your IP address and activity from your ISP and potential attackers."},
        {"instruction": "What is ransomware?", "response": "Ransomware encrypts a victim's files and demands payment for the decryption key. It is often delivered through phishing emails or malicious downloads."},
        {"instruction": "Define network segmentation.", "response": "Network segmentation divides a network into smaller subnetworks. This limits attacker movement and improves security."},
        {"instruction": "What is phishing?", "response": "Phishing is a cyber attack where attackers impersonate legitimate entities via email or websites to trick victims into revealing sensitive information."},
        {"instruction": "Explain the principle of least privilege.", "response": "Least privilege means giving users only the minimum permissions needed. This limits damage from accidents or compromised accounts."},
        {"instruction": "What is an intrusion detection system?", "response": "An IDS monitors network traffic for suspicious activity and generates alerts. Unlike a firewall, it does not block traffic."},
        {"instruction": "What is the difference between a vulnerability and an exploit?", "response": "A vulnerability is a weakness in a system. An exploit is code that takes advantage of that weakness. Vulnerabilities exist; exploits use them."},
        {"instruction": "Define penetration testing.", "response": "Penetration testing is a controlled attack simulation to identify security weaknesses. Ethical hackers find vulnerabilities before criminals can."},
        {"instruction": "What is endpoint security?", "response": "Endpoint security protects devices like computers and phones from threats. It includes antivirus, firewalls, encryption, and access controls."},
        {"instruction": "What is the CIA triad?", "response": "The CIA triad has three principles: Confidentiality (data accessible only to authorized people), Integrity (data is accurate), and Availability (data accessible when needed)."},
        {"instruction": "How does SSL/TLS work?", "response": "SSL/TLS encrypts browser-server communication. It uses a handshake with certificates to verify identity and establish encryption keys."},
        {"instruction": "What is identity and access management?", "response": "IAM ensures the right people have access to the right resources at the right time. It covers authentication, authorization, and role management."},
        {"instruction": "What is incident response?", "response": "Incident response is a structured process for handling security breaches. It includes preparation, detection, containment, eradication, and recovery."},
    ],
}

# =============================================
# 1. DAY 3 — DATASET PREPARATION
# =============================================
def prepare_dataset():
    print("=" * 70)
    print("DAY 3: Dataset Preparation")
    print("=" * 70 + "\n")

    dataset_file = f"datasets/{DOMAIN}_dataset.jsonl"
    if os.path.exists(dataset_file):
        with open(dataset_file) as f:
            raw_data = [json.loads(line) for line in f]
        print(f"Loaded dataset from: {dataset_file}")
    elif DOMAIN in DOMAIN_DATA:
        raw_data = DOMAIN_DATA[DOMAIN]
        os.makedirs("datasets", exist_ok=True)
        with open(dataset_file, "w") as f:
            for item in raw_data:
                f.write(json.dumps(item) + "\n")
        print(f"Generated dataset with {len(raw_data)} examples for: {DOMAIN}")
    else:
        print(f"No dataset for domain '{DOMAIN}'.")
        print(f"Available: {list(DOMAIN_DATA.keys())}")
        exit()

    char_lens = [len(i["instruction"]) + len(i["response"]) for i in raw_data]
    inst_lens = [len(i["instruction"]) for i in raw_data]
    resp_lens = [len(i["response"]) for i in raw_data]

    print(f"Total examples:            {len(raw_data)}")
    print(f"Instruction chars:         min={min(inst_lens)}, max={max(inst_lens)}, avg={sum(inst_lens)/len(inst_lens):.0f}")
    print(f"Response chars:            min={min(resp_lens)}, max={max(resp_lens)}, avg={sum(resp_lens)/len(resp_lens):.0f}\n")

    # Token length distribution
    tokenizer_viz = AutoTokenizer.from_pretrained(MODEL_NAME)
    token_lens = []
    for item in raw_data:
        text = f"Instruction:\n{item['instruction']}\n\nResponse:\n{item['response']}"
        token_lens.append(len(tokenizer_viz.encode(text)))

    plt.figure(figsize=(8, 4))
    plt.hist(token_lens, bins=8, color="skyblue", edgecolor="navy")
    plt.title(f"Token Length Distribution ({DOMAIN} Dataset)", fontsize=13)
    plt.xlabel("Tokens per Example")
    plt.ylabel("Count")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/dataset_distribution.png")
    plt.close()
    print(f"Token length: min={min(token_lens)}, max={max(token_lens)}, avg={sum(token_lens)/len(token_lens):.0f}")
    print("Plot saved to plots/dataset_distribution.png\n")

    # Train/val split
    dataset = Dataset.from_list(raw_data)
    split = dataset.train_test_split(test_size=0.15, seed=42)
    train_data = [raw_data[i] for i in split["train"].select_columns(["instruction"])]
    val_data = [raw_data[i] for i in split["test"].select_columns(["instruction"])]
    print(f"Train/val split: {len(split['train'])} train / {len(split['test'])} val ({len(split['test'])/len(raw_data)*100:.0f}%)\n")

    # Save splits
    os.makedirs("datasets", exist_ok=True)
    with open(f"datasets/{DOMAIN}_train.jsonl", "w") as f:
        for item in split["train"]:
            f.write(json.dumps(item) + "\n")
    with open(f"datasets/{DOMAIN}_val.jsonl", "w") as f:
        for item in split["test"]:
            f.write(json.dumps(item) + "\n")

    return raw_data, train_data, val_data, split, token_lens

# =============================================
# 2. DAY 2 — CHAT TEMPLATES
# =============================================
def setup_chat_template(tokenizer):
    print("=" * 70)
    print("DAY 2: Chat Templates")
    print("=" * 70 + "\n")

    # Define a custom chat template for GPT-2
    tokenizer.chat_template = "Instruction:\n{{ messages[0]['content'] }}\n\nResponse:\n{{ messages[-1]['content'] }}"

    print("Chat template set on tokenizer:")
    print("  Instruction:\n  {{ messages[0]['content'] }}\n\n  Response:\n  {{ messages[-1]['content'] }}\n")

    # Show formatted example
    messages = [
        {"role": "user", "content": "What is a firewall?"},
        {"role": "assistant", "content": "A firewall monitors network traffic."}
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False)
    print("Formatted example:")
    print(f"  {repr(formatted)}\n")

    return tokenizer

# =============================================
# 3. DAY 1 — BASELINE (Before SFT)
# =============================================
def baseline_test(model, tokenizer):
    print("=" * 70)
    print("DAY 1: Prompt Engineering — Baseline Before SFT")
    print("=" * 70 + "\n")

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    baseline_results = []

    for prompt in TEST_PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False)

        result = pipe(formatted, max_new_tokens=50, temperature=0.7, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        response = result[0]["generated_text"][len(formatted):].strip()
        baseline_results.append({"prompt": prompt, "response": response})

        print(f"Prompt: {prompt}")
        print(f"  Base model: {response[:100]}{'...' if len(response) > 100 else ''}")
        print()

    print("(The base model wasn't trained on your domain — expect generic text.)\n")
    return baseline_results

# =============================================
# 4. DAY 4 — SUPERVISED FINE-TUNING
# =============================================
def run_sft(raw_data, tokenizer):
    print("=" * 70)
    print("DAY 4: Supervised Fine-Tuning")
    print("=" * 70 + "\n")

    formatted = []
    for item in raw_data:
        messages = [
            {"role": "user", "content": item["instruction"]},
            {"role": "assistant", "content": item["response"]}
        ]
        formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)
        formatted.append({"text": formatted_text})

    dataset = Dataset.from_list(formatted)
    split = dataset.train_test_split(test_size=0.15, seed=42)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer_final = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer_final.pad_token = tokenizer_final.eos_token
    setup_chat_template(tokenizer_final)

    print(f"Model: {MODEL_NAME} ({sum(p.numel() for p in model.parameters()):,} params)")
    print(f"Max steps: {MAX_STEPS} | Learning rate: {LEARNING_RATE} | Batch size: {BATCH_SIZE}\n")

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        logging_steps=5,
        eval_steps=10,
        save_steps=MAX_STEPS,
        save_total_limit=1,
        eval_strategy="steps",
        report_to="none",
        fp16=False,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer_final,
        args=args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        response_template="Response:\n",
    )

    trainer.train()
    return model, tokenizer_final, trainer

# =============================================
# 5. DAY 5 — MODEL EVALUATION
# =============================================
def evaluate(model, tokenizer, val_data, baseline_results):
    print("=" * 70)
    print("DAY 5: Model Evaluation")
    print("=" * 70 + "\n")

    ft_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Compare responses
    print("Before vs After SFT:\n")
    for i, prompt in enumerate(TEST_PROMPTS):
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False)

        result = ft_pipe(formatted, max_new_tokens=50, temperature=0.7, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        ft_resp = result[0]["generated_text"][len(formatted):].strip()
        base_resp = baseline_results[i]["response"]

        print(f"Prompt: {prompt}")
        print(f"  BEFORE SFT: {base_resp[:80]}{'...' if len(base_resp) > 80 else ''}")
        print(f"  AFTER SFT:  {ft_resp[:80]}{'...' if len(ft_resp) > 80 else ''}")
        print()

    # Perplexity
    print("Perplexity (lower = better):\n")

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    base_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_tokenizer.pad_token = base_tokenizer.eos_token
    setup_chat_template(base_tokenizer)

    def compute_perplexity(model, tokenizer, texts):
        model.eval()
        total_loss = 0
        total_tokens = 0
        with torch.no_grad():
            for text in texts:
                messages = [
                    {"role": "user", "content": text["instruction"]},
                    {"role": "assistant", "content": text["response"]}
                ]
                formatted = tokenizer.apply_chat_template(messages, tokenize=False)
                encoded = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH)
                labels = encoded["input_ids"].clone()

                response_start = formatted.find("Response:\n") + len("Response:\n")
                prefix = formatted[:response_start]
                prefix_len = len(tokenizer.encode(prefix))
                labels[:, :prefix_len] = -100

                outputs = model(encoded["input_ids"], labels=labels)
                token_count = (labels != -100).sum().item()
                total_loss += outputs.loss.item() * token_count
                total_tokens += token_count

        return math.exp(total_loss / total_tokens) if total_tokens > 0 else 0

    base_ppl = compute_perplexity(base_model, base_tokenizer, val_data)
    ft_ppl = compute_perplexity(model, tokenizer, val_data)
    improvement = ((base_ppl - ft_ppl) / base_ppl * 100)

    print(f"  Base model:       {base_ppl:.2f}")
    print(f"  Fine-tuned model: {ft_ppl:.2f}")
    print(f"  Improvement:      {improvement:.1f}%\n")

    # Plot
    plt.figure(figsize=(7, 5))
    colors = ["lightcoral", "steelblue"]
    bars = plt.bar(["Base GPT-2", "Fine-Tuned"], [base_ppl, ft_ppl], color=colors, edgecolor="navy")
    plt.title(f"Perplexity Comparison — {DOMAIN.title()}", fontsize=14)
    plt.ylabel("Perplexity (lower is better)")
    plt.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, [base_ppl, ft_ppl]):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(base_ppl, ft_ppl) * 0.03, f"{val:.1f}", ha="center", fontsize=13)
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/perplexity_comparison.png")
    plt.close()
    print("Perplexity plot saved to plots/perplexity_comparison.png\n")

    return base_ppl, ft_ppl

# =============================================
# MAIN
# =============================================
def main():
    print("=" * 70)
    print(f"  Custom Domain Assistant — {DOMAIN.title()}")
    print("  Week 2 Capstone Project")
    print("=" * 70 + "\n")

    # Day 3: Dataset
    raw_data, train_data, val_data, split, token_lens = prepare_dataset()

    # Day 2 + Day 1 prep
    base_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_tokenizer.pad_token = base_tokenizer.eos_token
    setup_chat_template(base_tokenizer)

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    baseline_results = baseline_test(base_model, base_tokenizer)

    # Day 4: SFT
    model, tokenizer, trainer = run_sft(raw_data, base_tokenizer)

    # Save
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved to: {OUTPUT_DIR}\n")

    # Day 5: Evaluate
    import datasets
    test_instructions = [r["instruction"] for r in split["test"]]
    val_data_actual = [r for r in raw_data if r["instruction"] in test_instructions]

    base_ppl, ft_ppl = evaluate(model, tokenizer, val_data_actual if val_data_actual else raw_data[:3], baseline_results)

    # Summary
    print("=" * 70)
    print("PROJECT SUMMARY")
    print("=" * 70)
    print(f"""
  Domain:              {DOMAIN.title()}
  Training examples:   {len(split['train'])}
  Validation examples: {len(split['test'])}
  Base model:          {MODEL_NAME}
  Fine-tuned model:    {OUTPUT_DIR}
  Training steps:      {MAX_STEPS}
  Perplexity before:   {base_ppl:.2f}
  Perplexity after:    {ft_ppl:.2f}
  Improvement:         {improvement:.1f}%

  Concepts covered:
  ✅ Day 1 — Prompt Engineering (baseline before SFT)
  ✅ Day 2 — Chat Templates (apply_chat_template)
  ✅ Day 3 — Dataset Preparation (stats, split, distribution)
  ✅ Day 4 — Supervised Fine-Tuning (SFTTrainer)
  ✅ Day 5 — Model Evaluation (perplexity, before/after)

  Files generated:
  - datasets/{DOMAIN}_dataset.jsonl
  - datasets/{DOMAIN}_train.jsonl
  - datasets/{DOMAIN}_val.jsonl
  - plots/dataset_distribution.png
  - plots/perplexity_comparison.png
  - models/{DOMAIN}-assistant/
""")

    print("Next step: Run inference.py to chat with your assistant.")

if __name__ == "__main__":
    main()

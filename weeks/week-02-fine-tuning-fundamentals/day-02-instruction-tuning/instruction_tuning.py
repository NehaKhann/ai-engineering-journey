"""
Day 2 — Instruction Tuning & Chat Templates
============================================

Shows why base models (GPT-2) fail at conversation, how instruction-tuned
models (Qwen2.5) use chat templates, and how loss masking focuses training
on assistant responses only.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


# ──────────────────────────────────────────────
# SECTION 1: Base Model vs Instruction-Tuned
# ──────────────────────────────────────────────

def compare_models():
    """Compare how a base model (GPT-2) and an instruction-tuned model
    (Qwen2.5) respond to the same question."""

    question = "Explain what is a firewall in simple words."

    # --- Base model (GPT-2) ---
    print("[Base model — GPT-2]")
    base = pipeline("text-generation", model="gpt2")
    out = base(question, max_new_tokens=60, do_sample=True, temperature=0.7)[0]["generated_text"]
    print(f"  Prompt: {question}")
    print(f"  Output: {out}\n")

    # --- Instruction-tuned model (Qwen2.5) ---
    print("[Instruction-tuned — Qwen2.5-0.5B-Instruct]")
    chat = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")
    msg = [{"role": "user", "content": question}]
    out = chat(msg, max_new_tokens=80)[0]["generated_text"][-1]["content"]
    print(f"  Prompt: {question}")
    print(f"  Output: {out}\n")

    print("-> GPT-2 continues text arbitrarily; Qwen2.5 answers because it")
    print("  was fine-tuned on instruction-following data.\n")


# ──────────────────────────────────────────────
# SECTION 2: Inspecting the Chat Template
# ──────────────────────────────────────────────

def inspect_chat_template():
    """Show the raw Jinja2 template and the special tokens it uses."""

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    print(f"[Qwen2.5 Chat Template]\n{tokenizer.chat_template}\n")

    # Format a simple example to show the tokens in action
    messages = [
        {"role": "system", "content": "You are a cybersecurity expert."},
        {"role": "user", "content": "What is a firewall?"},
        {"role": "assistant", "content": "A firewall monitors traffic and blocks unauthorized access."},
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False)
    print("[Formatted conversation]")
    print(formatted)

    # Show what the special tokens actually look like as IDs
    encoded = tokenizer(formatted, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(encoded.input_ids[0])
    safe_tokens = [t.encode('ascii', 'replace').decode() for t in tokens]
    print(f"\n[Token IDs decoded]\n{safe_tokens}\n")


# ──────────────────────────────────────────────
# SECTION 3: Raw Text vs Template — Why Markers Matter
# ──────────────────────────────────────────────

def compare_raw_vs_template():
    """Tokenize the same conversation with and without a chat template
    and show the difference."""

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    messages = [
        {"role": "system", "content": "You are a cybersecurity expert."},
        {"role": "user", "content": "What is a firewall?"},
        {"role": "assistant", "content": "A firewall monitors and controls incoming and outgoing network traffic based on security rules."},
    ]

    raw_text = "System: You are a cybersecurity expert.\nUser: What is a firewall?\nAssistant: A firewall monitors and controls incoming and outgoing network traffic based on security rules."
    template_text = tokenizer.apply_chat_template(messages, tokenize=False)

    raw_ids = tokenizer.encode(raw_text)
    tmpl_ids = tokenizer.encode(template_text)

    print(f"[Raw text]             {len(raw_ids):3d} tokens  ->  {raw_ids}")
    print(f"[Template-formatted]   {len(tmpl_ids):3d} tokens  ->  {tmpl_ids}")
    print(f"\nDifference: {len(tmpl_ids) - len(raw_ids)} extra tokens (role markers)\n")

    # Show which tokens are added by the template
    print("[Extra tokens added by template]")
    for i, tid in enumerate(tmpl_ids):
        if tid not in raw_ids:
            print(f"  Position {i}: ID {tid} = {tokenizer.decode(tid)!r}")


# ──────────────────────────────────────────────
# SECTION 4: Building a Dataset
# ──────────────────────────────────────────────

def build_dataset():
    """Create a small instruction dataset and show token counts."""

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    dataset = [
        {"instruction": "What is a firewall?",
         "response": "A firewall monitors and controls incoming and outgoing network traffic based on security rules."},
        {"instruction": "Explain DNS.",
         "response": "DNS, or Domain Name System, translates human-readable domain names into IP addresses that computers use to locate each other."},
        {"instruction": "What is encryption?",
         "response": "Encryption converts plaintext data into ciphertext using an algorithm and a key, making it unreadable without authorization."},
        {"instruction": "Define phishing.",
         "response": "Phishing is a cyber attack where attackers impersonate legitimate entities to trick victims into revealing sensitive information such as passwords or credit card details."},
        {"instruction": "What is two-factor authentication?",
         "response": "Two-factor authentication (2FA) adds an extra security layer by requiring two different verification methods before granting access to an account or system."},
    ]

    print(f"[Dataset] {len(dataset)} instruction-response pairs")
    print("-" * 60)

    for i, item in enumerate(dataset, 1):
        messages = [
            {"role": "system", "content": "You are a cybersecurity expert."},
            {"role": "user", "content": item["instruction"]},
            {"role": "assistant", "content": item["response"]},
        ]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False)
        tokens = tokenizer.encode(formatted)
        print(f"  Ex {i}: {len(tokens):3d} tokens — {item['instruction']}")

    print()
    return dataset


# ──────────────────────────────────────────────
# SECTION 5: Loss Masking Demonstration
# ──────────────────────────────────────────────

def demonstrate_loss_masking():
    """Show how labels are set to -100 for non-assistant tokens so the
    model only learns from the assistant response."""

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    messages = [
        {"role": "system", "content": "You are a cybersecurity expert."},
        {"role": "user", "content": "What is a firewall?"},
        {"role": "assistant", "content": "A firewall monitors and controls incoming and outgoing network traffic based on security rules."},
    ]

    template_text = tokenizer.apply_chat_template(messages, tokenize=False)
    encoded = tokenizer(template_text, return_tensors="pt")

    # Find where the assistant response starts (the token after <|im_start|>assistant)
    assistant_marker = "<|im_start|>assistant"
    marker_ids = tokenizer.encode(assistant_marker, add_special_tokens=False)
    input_ids = encoded.input_ids[0]

    # Search for the marker position
    response_start = None
    for i in range(len(input_ids) - len(marker_ids) + 1):
        if input_ids[i:i+len(marker_ids)].tolist() == marker_ids:
            response_start = i + len(marker_ids)
            break

    if response_start is None:
        print("Could not locate assistant marker.")
        return

    # Build labels: -100 for tokens before response, actual IDs for response
    labels = torch.full_like(input_ids, -100)
    labels[response_start:] = input_ids[response_start:]

    # Print a readable table
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    total_len = len(input_ids)

    print(f"[Loss Masking — Total tokens: {total_len}]")
    print(f"[Assistant response starts at position {response_start}: {tokenizer.decode(input_ids[response_start:response_start+10])!r}...]\n")

    header = f"{'Pos':>4} | {'Label':>6} | {'Token':<30}"
    print(header)
    print("-" * len(header))

    for pos in range(total_len):
        label = labels[pos].item()
        label_str = "LEARN" if label != -100 else "IGNORE"
        token_repr = tokens[pos].replace("Ġ", " ").replace("<0x0A>", "\\n")
        token_repr = token_repr.encode('ascii', 'replace').decode()
        print(f"{pos:4d} | {label_str:>6} | {token_repr:<30}")

    print(f"\n-> {total_len - response_start} tokens will be learned "
          f"(the assistant response)")
    print(f"-> {response_start} tokens are ignored "
          f"(system prompt + user message)\n")


# ──────────────────────────────────────────────
# RUN ALL SECTIONS
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 72)
    print("  Day 2 — Instruction Tuning & Chat Templates")
    print("=" * 72)

    print("\n" + "=" * 72)
    print("  1. Base Model vs Instruction-Tuned Model")
    print("=" * 72)
    compare_models()

    print("=" * 72)
    print("  2. Inspecting the Chat Template")
    print("=" * 72)
    inspect_chat_template()

    print("=" * 72)
    print("  3. Raw Text vs Template — Why Markers Matter")
    print("=" * 72)
    compare_raw_vs_template()

    print("=" * 72)
    print("  4. Building a Dataset")
    print("=" * 72)
    build_dataset()

    print("=" * 72)
    print("  5. Loss Masking Demonstration")
    print("=" * 72)
    demonstrate_loss_masking()

    print("=" * 72)
    print("  Done — Day 2 complete.")
    print("=" * 72)

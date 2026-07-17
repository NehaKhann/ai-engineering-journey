from transformers import pipeline
import matplotlib.pyplot as plt

print("=" * 75)
print("Day 1 — Prompt Engineering vs Fine-Tuning")
print("=" * 75 + "\n")

# =============================================
# PART 1: BASE MODEL vs INSTRUCTION-TUNED MODEL
# =============================================
print("PART 1: Why Prompt Engineering Requires an Instruction-Tuned Model")
print("-" * 60)

prompt = "Explain what is a firewall in simple words."

# GPT-2 (base model) — text completion only
print("\nLoading GPT-2 (base model)...")
base = pipeline("text-generation", model="gpt2")

print("\nBase model (GPT-2):")
out = base(prompt, max_new_tokens=60, temperature=0.7, do_sample=True, pad_token_id=50256)
print(f"  Prompt:   {prompt}")
print(f"  Response: {out[0]['generated_text'][len(prompt):].strip()}")

# Qwen2.5 (instruction-tuned) — follows instructions
print("\nLoading Qwen2.5 (instruction-tuned)...")
chat = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")

print("\nInstruction-tuned model (Qwen2.5):")
msg = [{"role": "user", "content": prompt}]
out = chat(msg, max_new_tokens=80, temperature=0.7)
print(f"  Prompt:   {prompt}")
print(f"  Response: {out[0]['generated_text'][-1]['content']}")

print("\nInsight: Base models complete text. Instruction-tuned models follow instructions.")
print("Prompt engineering techniques only work reliably on instruction-tuned models.\n")

# =============================================
# PART 2: PROMPT ENGINEERING TECHNIQUES
# =============================================
print("=" * 75)
print("PART 2: Prompt Engineering Techniques")
print("=" * 75 + "\n")

techniques = []
responses = []

# 1. Zero-Shot
name = "1. Zero-Shot"
techniques.append(name)
msg = [{"role": "user", "content": "What is a firewall?"}]
out = chat(msg, max_new_tokens=80, temperature=0.7)
resp = out[0]['generated_text'][-1]['content']
responses.append(resp)
print(f"{name}:")
print(f"  {resp}\n")

# 2. Few-Shot
name = "2. Few-Shot"
techniques.append(name)
msg = [
    {"role": "user", "content": "What is an IP address?"},
    {"role": "assistant", "content": "An IP address is a unique numerical label assigned to each device on a network. It works like a mailing address for your computer."},
    {"role": "user", "content": "What is DNS?"},
    {"role": "assistant", "content": "DNS stands for Domain Name System. It translates human-readable domain names (like google.com) into IP addresses that computers use."},
    {"role": "user", "content": "What is a firewall?"}
]
out = chat(msg, max_new_tokens=80, temperature=0.7)
resp = out[0]['generated_text'][-1]['content']
responses.append(resp)
print(f"{name}:")
print(f"  {resp}\n")

# 3. Chain-of-Thought
name = "3. Chain-of-Thought"
techniques.append(name)
msg = [{"role": "user", "content": "Explain what a firewall is. Think step by step: first what it protects, then how it filters traffic, then a simple analogy."}]
out = chat(msg, max_new_tokens=120, temperature=0.7)
resp = out[0]['generated_text'][-1]['content']
responses.append(resp)
print(f"{name}:")
print(f"  {resp}\n")

# 4. Role Prompting (system message)
name = "4. Role Prompting"
techniques.append(name)
msg = [
    {"role": "system", "content": "You are a cybersecurity professor with 15 years of experience. You explain concepts clearly with real-world analogies for beginners."},
    {"role": "user", "content": "What is a firewall?"}
]
out = chat(msg, max_new_tokens=100, temperature=0.7)
resp = out[0]['generated_text'][-1]['content']
responses.append(resp)
print(f"{name}:")
print(f"  {resp}\n")

# =============================================
# PART 3: TEMPERATURE + PROMPTING INTERPLAY
# =============================================
print("=" * 75)
print("PART 3: How Temperature Affects Prompt Responses")
print("=" * 75 + "\n")

temps = [("Low (0.2)", 0.2), ("Balanced (0.7)", 0.7), ("Creative (1.2)", 1.2)]
for label, temp in temps:
    msg = [{"role": "user", "content": "Give one benefit of using a firewall. Keep it to one sentence."}]
    out = chat(msg, max_new_tokens=40, temperature=temp)
    text = out[0]['generated_text'][-1]['content']
    print(f"  Temperature {label}:")
    print(f"  → {text}\n")

# =============================================
# PART 4: PROMPT ENGINEERING vs FINE-TUNING
# =============================================
print("=" * 75)
print("PART 4: Prompt Engineering vs Fine-Tuning — When to Use Each")
print("=" * 75)

comparison = """
                    PROMPT ENGINEERING          FINE-TUNING
                    ─────────────────────      ─────────────────────
Training needed     No                         Yes
Time to deploy      Immediate                  Hours–days
Cost                Free                       GPU compute required
Model changes       None (external)            Permanent weight updates
Consistency         Varies                     High
Domain knowledge    Limited by base model      Can learn new domains
Best for            Quick prototypes           Production systems
Complex prompts     Gets unwieldy              Handles naturally

Decision flow:
1. Start with prompt engineering
2. Need more consistency? → Fine-tune
3. Need domain expertise? → Fine-tune
4. Prompts too complex? → Fine-tune
"""
print(comparison)

# =============================================
# VISUALIZATION
# =============================================
print("\nGenerating visualization...\n")

plt.figure(figsize=(14, 9))
for i, (name, text) in enumerate(zip(techniques, responses)):
    plt.subplot(2, 2, i + 1)
    short = text[:200] + "..." if len(text) > 200 else text
    plt.text(0.5, 0.5, short, ha='center', va='center', wrap=True, fontsize=9)
    plt.title(name, fontsize=12, fontweight='bold')
    plt.axis('off')

plt.suptitle("Prompt Engineering Techniques on Qwen2.5-0.5B-Instruct", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("Done! Review each technique's output to see how prompt design changes responses.")

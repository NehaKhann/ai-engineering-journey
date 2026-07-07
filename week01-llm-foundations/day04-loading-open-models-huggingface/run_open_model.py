from transformers import pipeline
import matplotlib.pyplot as plt

print("="*60)
print("Day 4 — Loading Open Models with Hugging Face")
print("="*60 + "\n")

prompt = "My firewall keeps blocking"

# =============================================
# 1. Base Model: GPT-2 (Raw Text Completion)
# =============================================
print("Using GPT-2 (Base Model)\n")

generator = pipeline("text-generation", model="gpt2")

temperatures = [0.3, 0.7, 1.2]
generated_texts = []

print("Comparing different temperatures:\n")

for temp in temperatures:
    result = generator(
        prompt,
        max_new_tokens=40,
        temperature=temp,
        do_sample=True,
        pad_token_id=50256
    )
    text = result[0]["generated_text"]
    generated_texts.append(text)
    
    print(f"Temperature = {temp}")
    print(text)
    print("-" * 80)

# =============================================
# 2. Instruction-Tuned Model (Much Better for Q&A)
# =============================================
print("\nUsing Instruction-Tuned Model (Qwen2.5-0.5B-Instruct)\n")

chat = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")

messages = [
    {"role": "user", "content": "Why does a firewall sometimes block legitimate traffic?"}
]

response = chat(messages, max_new_tokens=120, temperature=0.7)

print("Question:", messages[0]["content"])
print("\nAnswer:")
print(response[0]["generated_text"][-1]["content"])

# =============================================
# 3. Visualization: Effect of Temperature
# =============================================
print("\nGenerating visualization...")

plt.figure(figsize=(12, 9))

for i, temp in enumerate(temperatures):
    plt.subplot(len(temperatures), 1, i+1)
    plt.text(0.5, 0.5, generated_texts[i], ha='center', va='center', 
             fontsize=11, wrap=True)
    plt.title(f"Temperature = {temp}", fontsize=14, pad=15)
    plt.axis('off')

plt.suptitle("Effect of Temperature on Text Generation (GPT-2)", fontsize=16)
plt.tight_layout()
plt.show()

print("\nDay 4 Complete! You can see how temperature affects creativity.")
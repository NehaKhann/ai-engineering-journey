from transformers import pipeline
import matplotlib.pyplot as plt

print("="*80)
print("Day 6 — Generation Parameters in LLMs")
print("="*80 + "\n")

prompt = "My firewall keeps blocking"

# Load the model
generator = pipeline("text-generation", model="gpt2")

# Different settings to compare
parameters = [
    {"name": "Low Temperature (0.3)",      "temp": 0.3,  "repetition_penalty": 1.1},
    {"name": "Balanced (0.7)",             "temp": 0.7,  "repetition_penalty": 1.15},
    {"name": "High Temperature (1.2)",     "temp": 1.2,  "repetition_penalty": 1.2},
    {"name": "Very Creative (1.5)",        "temp": 1.5,  "repetition_penalty": 1.25},
]

generated_texts = []

print(f"Prompt: \"{prompt}\"\n")
print("Generating with different parameters...\n")

for param in parameters:
    result = generator(
        prompt,
        max_new_tokens=40,
        temperature=param["temp"],
        do_sample=True,
        repetition_penalty=param["repetition_penalty"],   # Reduces repetition
        no_repeat_ngram_size=2,                           # Prevents repeating phrases
        pad_token_id=50256
    )
    
    text = result[0]["generated_text"]
    generated_texts.append(text)
    
    print(f"{param['name']}:")
    print(text)
    print("-" * 90)

# =============================================
# Visual Comparison
# =============================================
plt.figure(figsize=(14, 10))

for i, param in enumerate(parameters):
    plt.subplot(2, 2, i+1)
    plt.text(0.5, 0.5, generated_texts[i], 
             ha='center', va='center', wrap=True, fontsize=10)
    plt.title(param["name"], fontsize=12, pad=15)
    plt.axis('off')

plt.suptitle("Effect of Generation Parameters on Text Output (GPT-2)", fontsize=16)
plt.tight_layout()
plt.show()

print("\n🎯 Key Takeaways from Day 6:")
print("• temperature      → Controls creativity (low = safe, high = creative)")
print("• repetition_penalty → Helps reduce word/phrase repetition")
print("• max_new_tokens   → Controls length of generated text")
print("• Balanced settings (temp 0.7) usually give the best results.")
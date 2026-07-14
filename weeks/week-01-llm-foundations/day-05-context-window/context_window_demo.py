from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import matplotlib.pyplot as plt

print("="*80)
print("Day 5 — Understanding Context Window in LLMs")
print("="*80 + "\n")

MODEL_NAME = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

def visualize_context_window():
    print("📏 Visualizing Context Window Concept\n")

    # Different context lengths
    texts = [
        "The weather is nice today.",
        "The weather is nice today. I went for a walk in the park.",
        "The weather is nice today. I went for a walk in the park. I saw many birds and flowers.",
        "The weather is nice today. I went for a walk in the park. I saw many birds and flowers. Then I met my old friend and we had coffee together."
    ]

    token_counts = []
    continuations = []

    for i, text in enumerate(texts, 1):
        inputs = tokenizer(text, return_tensors="pt")
        token_count = inputs.input_ids.shape[1]
        token_counts.append(token_count)

        print(f"Example {i}: {token_count} tokens")

        # Generate continuation
        with torch.no_grad():
            output = model.generate(
                inputs.input_ids,
                max_new_tokens=25,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        continuation = generated[len(text):].strip()
        continuations.append(continuation)
        print(f"→ Continuation: {continuation[:100]}...\n")

    # ==================== VISUALIZATION ====================
    plt.figure(figsize=(12, 7))

    # Bar chart for context size
    bars = plt.bar(range(1, len(texts)+1), token_counts, color=['lightblue', 'skyblue', 'deepskyblue', 'dodgerblue'])
    plt.title("Context Window: How Much Information GPT-2 Can 'See'", fontsize=16)
    plt.xlabel("Example Number (Increasing Context Length)")
    plt.ylabel("Number of Tokens")
    plt.xticks(range(1, len(texts)+1))

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{int(height)} tokens', ha='center', va='bottom')

    plt.axhline(y=1024, color='red', linestyle='--', label='GPT-2 Context Limit (1024 tokens)')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("🎯 Key Insight:")
    print("• As context grows, the model has more information to generate better responses.")
    print("• GPT-2 can only handle up to 1024 tokens at once.")
    print("• Modern models like GPT-4 have much larger context windows (up to 128k tokens).")


# Run the demo
visualize_context_window()
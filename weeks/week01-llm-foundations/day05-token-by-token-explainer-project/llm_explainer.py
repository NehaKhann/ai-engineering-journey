from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
import matplotlib.pyplot as plt

# =============================================
# LOAD THE MODELS
# =============================================
MODEL_NAME = "gpt2"

print(f"Loading {MODEL_NAME}... (This may take a moment on first run)\n")

# Tokenizer: Converts text to tokens
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# attn_model: Used to get attention weights
attn_model = AutoModel.from_pretrained(MODEL_NAME, output_attentions=True)

# gen_model: Used to predict next tokens
gen_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)


def explain_sentence(text: str, focus_word: str = None):
    """
    Main function that explains how the LLM understands the input sentence.
    """
    print("\n" + "="*80)
    print(f"🤖 LLM EXPLAINER DASHBOARD")
    print("="*80)
    print(f"Input: \"{text}\"")
    print("-"*80)

    # Tokenization
    inputs = tokenizer(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # 1. Show Tokenization
    print("\n📋 1. TOKENIZATION")
    for i, t in enumerate(tokens):
        print(f"   {i:2d}: {t}")

    # 2. Attention Analysis + Visualization
    if focus_word:
        target = f"Ġ{focus_word}" if f"Ġ{focus_word}" in tokens else focus_word
        if target in tokens:
            idx = tokens.index(target)
            
            with torch.no_grad():
                outputs = attn_model(**inputs)
            last_attn = outputs.attentions[-1][0].mean(dim=0)

            print(f"\n👁️ 2. ATTENTION from '{focus_word}'")
            print("   Token              Attention   Visual")
            print("   " + "-"*55)
            
            attention_weights = []
            for i, tok in enumerate(tokens):
                if i > idx: 
                    break
                weight = last_attn[idx, i].item()
                bar = "█" * int(weight * 25)
                print(f"   {tok:15s} {weight:.4f}   {bar}")
                attention_weights.append(weight)

            # Visual Bar Chart
            plt.figure(figsize=(10, 4))
            plt.bar(tokens[:len(attention_weights)], attention_weights, color='skyblue')
            plt.title(f"Attention from '{focus_word}'")
            plt.xlabel("Tokens")
            plt.ylabel("Attention Weight")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        else:
            print(f"\n👁️ 2. ATTENTION: '{focus_word}' not found in sentence.")

    # 3. Next Token Prediction
    with torch.no_grad():
        outputs = gen_model(**inputs)
    next_logits = outputs.logits[0, -1]
    top5 = torch.topk(next_logits, 5)

    print("\n🔮 3. NEXT TOKEN PREDICTION")
    print("   Token               Score")
    print("   " + "-"*40)
    for score, idx in zip(top5.values, top5.indices):
        token = tokenizer.decode([idx])
        print(f"   {token:15s} {score.item():.3f}")

    print("\n" + "="*80 + "\n")


# ==================== Interactive Mode ====================
if __name__ == "__main__":
    print("LLM Explainer is ready! Type 'exit' to quit.\n")
    
    while True:
        text = input("Enter a sentence (or 'exit'): ").strip()
        if text.lower() == 'exit':
            break
        if not text:
            continue
            
        focus = input("Focus word (press Enter to skip): ").strip() or None
        explain_sentence(text, focus)
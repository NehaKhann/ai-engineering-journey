import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# =============================================
# CONFIGURATION
# =============================================
DOMAIN = "cybersecurity"
MODEL_PATH = f"./models/{DOMAIN}-assistant"

# =============================================
# CHAT TEMPLATE (must match train.py)
# =============================================
CUSTOM_TEMPLATE = "Instruction:\n{{ messages[0]['content'] }}\n\nResponse:\n"

# =============================================
# INFERENCE
# =============================================
def main():
    print("=" * 70)
    print(f"  Custom Domain Assistant — {DOMAIN.title()}")
    print("=" * 70 + "\n")

    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at: {MODEL_PATH}")
        print("Run train.py first to train the model.\n")
        return

    print(f"Loading model from: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = CUSTOM_TEMPLATE

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    print(f"\n{'='*70}")
    print(f"  {DOMAIN.upper()} ASSISTANT")
    print(f"  Type your questions below.")
    print(f"  Type 'exit' to quit.")
    print(f"{'='*70}\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("\nGoodbye!")
            break
        if not user_input:
            continue

        messages = [{"role": "user", "content": user_input}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False)

        result = pipe(formatted, max_new_tokens=100, temperature=0.7, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        response = result[0]["generated_text"][len(formatted):].strip()

        print(f"\nAssistant: {response}\n")

if __name__ == "__main__":
    main()

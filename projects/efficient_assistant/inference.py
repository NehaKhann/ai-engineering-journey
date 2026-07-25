"""
Capstone Project — Interactive Chat with the Merged Assistant
=================================================================
Loads the merged model produced by train.py (Day 6's output) and lets
you chat with it interactively. No PEFT import needed here — the
merged model is a plain, standalone Hugging Face model.

Usage:
    python inference.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MERGED_DIR = "merged_cybersecurity_assistant"
MAX_NEW_TOKENS = 150


def load_model():
    print(f"Loading merged model from ./{MERGED_DIR}/ ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MERGED_DIR)
    model = AutoModelForCausalLM.from_pretrained(MERGED_DIR, torch_dtype=dtype).to(device)
    model.eval()

    print(f"Loaded on {device}. Type 'exit' or 'quit' to stop.\n")
    return model, tokenizer


def chat(model, tokenizer):
    history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})

        inputs = tokenizer.apply_chat_template(
            history, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        response = tokenizer.decode(
            output_ids[0][inputs.shape[-1]:], skip_special_tokens=True
        ).strip()

        print(f"Assistant: {response}\n")
        history.append({"role": "assistant", "content": response})


def main():
    model, tokenizer = load_model()
    chat(model, tokenizer)


if __name__ == "__main__":
    main()

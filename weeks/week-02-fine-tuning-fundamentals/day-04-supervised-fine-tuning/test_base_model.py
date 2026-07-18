"""
test_base_model.py — Chat with the ORIGINAL, un-fine-tuned GPT-2 model.
Useful for comparing against your fine-tuned version in test_model.py.
"""

from transformers import pipeline

MODEL_PATH = "gpt2"  # plain GPT-2, not your fine-tuned folder

print("Loading the original base GPT-2 model...")
generator = pipeline("text-generation", model=MODEL_PATH, device="cpu")
print("Model loaded! Type a question, or type 'quit' to exit.\n")

while True:
    question = input("Your question: ").strip()

    if question.lower() in ("quit", "exit", "q"):
        print("Goodbye!")
        break

    if not question:
        continue

    prompt = f"Instruction:\n{question}\n\nResponse:\n"

    result = generator(
        prompt,
        max_new_tokens=120,              # was 60 — gives the model room to finish its thought
        do_sample=True,
        temperature=0.7,
        repetition_penalty=1.3,          # discourages reusing recently generated words
        no_repeat_ngram_size=3,          # blocks exact 3-word phrases from repeating
        pad_token_id=generator.tokenizer.eos_token_id,
        eos_token_id=generator.tokenizer.eos_token_id,  # allows the model to stop naturally
    )[0]

    answer = result["generated_text"][len(prompt):].strip()
    print(f"\nBase model's answer:\n  {answer}\n")
    print("-" * 60)
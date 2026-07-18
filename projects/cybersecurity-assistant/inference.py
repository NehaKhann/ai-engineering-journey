from transformers import pipeline

print("🔒 Loading Cybersecurity Assistant...\n")

model_path = "./results/cybersecurity-assistant"

pipe = pipeline(
    "text-generation",
    model=model_path,
    max_new_tokens=150,
    temperature=0.7,
    do_sample=True,
    repetition_penalty=1.15,
    pad_token_id=50256
)

print("Cybersecurity Assistant is ready! Ask security-related questions.\n")
print("Type 'exit' or 'quit' to stop.\n")

while True:
    question = input("You: ").strip()
    if question.lower() in ['exit', 'quit', 'bye']:
        print("Goodbye! Stay safe online.")
        break
    
    if not question:
        continue

    prompt = f"Instruction:\n{question}\n\nResponse:\n"
    
    try:
        response = pipe(prompt)[0]["generated_text"]
        answer = response[len(prompt):].strip()
        print(f"Assistant: {answer}\n")
    except Exception as e:
        print("Error generating response. Please try again.\n")
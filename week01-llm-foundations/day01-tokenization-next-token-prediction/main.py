from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

text = "My firewall keeps blocking"
inputs = tokenizer(text, return_tensors="pt")

print("Tokens:", tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))

with torch.no_grad():
    outputs = model(**inputs)

next_token_logits = outputs.logits[0, -1]
top5 = torch.topk(next_token_logits, 5)

print("\nTop 5 predicted next tokens:")
for score, idx in zip(top5.values, top5.indices):
    print(f"  {tokenizer.decode([idx]):15s} score: {score.item():.2f}")
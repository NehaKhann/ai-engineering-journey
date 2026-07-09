from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# =============================================
# 1. LOAD THE MODEL AND TOKENIZER
# =============================================

model_name = "gpt2"

# Tokenizer converts text to numbers (tokens) the model understands
tokenizer = AutoTokenizer.from_pretrained(model_name)

# We use AutoModelForCausalLM because we want to predict the next token
model = AutoModelForCausalLM.from_pretrained(model_name)

# =============================================
# 2. PREPARE THE INPUT TEXT
# =============================================

# The sentence we give to the model
text = "My firewall keeps blocking"

# Convert the text into tokens (numbers)
inputs = tokenizer(text, return_tensors="pt")

# Show the tokens so we can see how the sentence was split
print("Tokens:", tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))

# =============================================
# 3. RUN THE MODEL (Next Token Prediction)
# =============================================

# Disable gradient calculation (we're not training, just predicting)
with torch.no_grad():
    outputs = model(**inputs)

# Get the logits (raw scores) for the next token
# [0, -1] means: first batch, last position in the sequence
next_token_logits = outputs.logits[0, -1]

# Get the top 5 most likely next tokens
top5 = torch.topk(next_token_logits, 5)

# =============================================
# 4. SHOW THE RESULTS
# =============================================

print("\nTop 5 predicted next tokens:")
for score, idx in zip(top5.values, top5.indices):
    token = tokenizer.decode([idx])          # Convert token id back to text
    print(f"  {token:15s} score: {score.item():.2f}")
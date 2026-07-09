from transformers import AutoTokenizer, AutoModel
import torch

# =============================================
# 1. LOAD THE MODEL AND TOKENIZER
# =============================================

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_attentions=True)

# =============================================
# 2. CONFIGURATION - EASY TO CHANGE
# =============================================

# Change these two lines to analyze any sentence and any word!
text = "The firewall blocked the traffic because it was suspicious"

# Target word you want to analyze (e.g. "it", "firewall", "traffic")
target_word = "it"          # ← Change this easily!

# =============================================
# 3. PREPARE THE INPUT
# =============================================

inputs = tokenizer(text, return_tensors="pt")

# =============================================
# 4. RUN THE MODEL
# =============================================

with torch.no_grad():
    outputs = model(**inputs)

# =============================================
# 5. EXTRACT ATTENTION
# =============================================

attentions = outputs.attentions
last_layer_attn = attentions[-1][0].mean(dim=0)   # Average all heads

# Convert to readable tokens
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# Find the position of the target word
try:
    target_idx = tokens.index("Ġ" + target_word)      # Most words have Ġ
except ValueError:
    try:
        target_idx = tokens.index(target_word)        # Try without Ġ
    except ValueError:
        print(f"Word '{target_word}' not found in the sentence!")
        exit()

# =============================================
# 6. ANALYZE AND PRINT ATTENTION
# =============================================

print(f"Attention from '{tokens[target_idx]}' (position {target_idx}) "
      f"to previous tokens:\n")

for i, tok in enumerate(tokens):
    if i > target_idx:        # Only previous tokens
        continue
    
    weight = last_layer_attn[target_idx, i].item()
    
    if weight > 0.01:         # Filter small values
        print(f"  {tok:15s} → {weight:.4f}")
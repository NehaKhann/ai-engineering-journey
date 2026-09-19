"""Loads the two local models the assistant uses. Both run on a laptop CPU.

- an embedding model turns text into vectors (Module 04)
- a small chat model follows instructions (Modules 02 and 03)

Models load lazily, the first time they are needed, so tests that never call them stay fast.
"""

from functools import lru_cache

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@lru_cache(maxsize=1)
def _embedding_model():
    return AutoTokenizer.from_pretrained(EMBEDDING_MODEL), AutoModel.from_pretrained(EMBEDDING_MODEL).eval()


@lru_cache(maxsize=1)
def _chat_model():
    tokenizer = AutoTokenizer.from_pretrained(CHAT_MODEL)
    return tokenizer, AutoModelForCausalLM.from_pretrained(CHAT_MODEL, dtype=torch.float32)


def embed(texts):
    """Turn a list of strings into an array of unit-length vectors, one row per string."""
    tokenizer, model = _embedding_model()
    batch = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        token_vectors = model(**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1).float()
    pooled = (token_vectors * mask).sum(dim=1) / mask.sum(dim=1)
    return torch.nn.functional.normalize(pooled, dim=1).numpy()


def chat(messages, max_new_tokens=100):
    """Send {"role", "content"} messages to the chat model and return its reply text.

    Decoding is greedy (no randomness), so the same prompt always gives the same reply.
    """
    tokenizer, model = _chat_model()
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    output = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

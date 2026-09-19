"""The two local models this project uses. Both run on a laptop CPU and load lazily."""

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
    return AutoTokenizer.from_pretrained(CHAT_MODEL), AutoModelForCausalLM.from_pretrained(CHAT_MODEL, dtype=torch.float32)


def embed(texts):
    """Unit-length vectors, one row per text (mean pooling, as in Module 04)."""
    tokenizer, model = _embedding_model()
    batch = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        hidden = model(**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1).float()
    return torch.nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1), dim=1).numpy()


def chat(messages, max_new_tokens=80):
    """One reply from the chat model. Greedy decoding, so the same prompt gives the same reply."""
    tokenizer, model = _chat_model()
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

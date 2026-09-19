"""The local chat model, used with its native tool-calling format (Module 07)."""

from functools import lru_cache

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CHAT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@lru_cache(maxsize=1)
def _load():
    return AutoTokenizer.from_pretrained(CHAT_MODEL), AutoModelForCausalLM.from_pretrained(CHAT_MODEL, dtype=torch.float32)


def local_model(messages, tool_specs, max_new_tokens=150):
    """Return the model's next message, which may contain <tool_call> blocks. Greedy, so repeatable."""
    tokenizer, model = _load()
    text = tokenizer.apply_chat_template(messages, tools=tool_specs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

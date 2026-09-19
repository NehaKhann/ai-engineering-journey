"""The model behind the API. The server only depends on this small interface, so a fake can stand in for tests."""

import threading
from dataclasses import dataclass


@dataclass
class Generation:
    text: str
    prompt_tokens: int
    completion_tokens: int


class FakeBackend:
    """A deterministic stand-in for a model. Records how many times it was really called."""

    def __init__(self, reply="This is a fake reply.", fail=False):
        self.reply, self.fail, self.calls = reply, fail, 0

    def generate(self, messages, max_new_tokens):
        self.calls += 1
        if self.fail:
            raise RuntimeError("model exploded: internal detail that must never reach the user")
        words = self.reply.split()[:max_new_tokens]
        prompt_tokens = sum(len(m["content"].split()) for m in messages)
        return Generation(" ".join(words), prompt_tokens, len(words))

    def stream(self, messages, max_new_tokens):
        result = self.generate(messages, max_new_tokens)
        for word in result.text.split():
            yield word + " "


class LocalBackend:
    """A small open chat model running on this machine (the same one used throughout this track)."""

    MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(self.MODEL_ID, dtype=torch.float32).eval()
        self._lock = threading.Lock()  # one model instance handles one generation at a time

    def _inputs(self, messages):
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.tokenizer(text, return_tensors="pt")

    def generate(self, messages, max_new_tokens):
        inputs = self._inputs(messages)
        with self._lock, self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        generated = out[0][inputs["input_ids"].shape[1]:]
        return Generation(self.tokenizer.decode(generated, skip_special_tokens=True).strip(), inputs["input_ids"].shape[1], len(generated))

    def stream(self, messages, max_new_tokens):
        from transformers import TextIteratorStreamer

        inputs = self._inputs(messages)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        def run():
            with self._lock, self.torch.no_grad():
                self.model.generate(**inputs, streamer=streamer, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)

        threading.Thread(target=run, daemon=True).start()
        yield from streamer
